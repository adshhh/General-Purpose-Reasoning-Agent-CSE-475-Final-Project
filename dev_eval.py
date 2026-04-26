#!/usr/bin/env python3
# Owner: Person B (Soul) - dev-set evaluation harness for cs + fp solvers
"""Run Person B's solvers on the labeled dev set; report per-domain accuracy + call usage."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# .env loader (runs before utils import)
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# Side-effect: log every LLM call to call_log.jsonl.
import call_logger  # noqa: F401

import ast
import re

import common_sense
import future_prediction
from evaluator import grade_exact, grade_llm_judge, _normalize
from utils import get_per_question_calls, reset_per_question_counter


# The dev file is a labelled set with `input`, `output`, `domain` per record.
# It may live in a few places depending on how the user obtained it -- search
# in priority order and use the first match.
_PROJECT_ROOT = Path(__file__).resolve().parent
_DEV_NAME = "cse476_final_project_dev_data.json"
_DEV_CANDIDATES = [
    _PROJECT_ROOT / "data" / _DEV_NAME,                                # preferred
    _PROJECT_ROOT / "final_project_tutorial_and_dev_data" / _DEV_NAME, # nested (Soul's layout)
    _PROJECT_ROOT.parent / "final_project_tutorial_and_dev_data" / _DEV_NAME,
]


def _resolve_dev_path() -> Path:
    for p in _DEV_CANDIDATES:
        if p.exists():
            return p
    # Fall back to the first candidate so the error message points users
    # at the canonical place to put the file.
    return _DEV_CANDIDATES[0]


DEV_PATH = _resolve_dev_path()
RESULTS_OUT = _PROJECT_ROOT / "dev_eval_results.json"


# ---------------------------------------------------------------------------
# Domain-aware exact-match grader for future_prediction
# ---------------------------------------------------------------------------

# Inner of the LAST \boxed{...} -- this is what the prompt tells the model
# to wrap its answer in, so it's the part we should compare.
_BOXED_INNER = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _strip_boxed(s: str) -> str:
    """Return the inner text of the last \\boxed{...} in s, or s unchanged."""
    if not s:
        return ""
    matches = _BOXED_INNER.findall(s)
    return matches[-1].strip() if matches else s.strip()


def _expected_choices(expected: str) -> List[str]:
    """
    Parse a future_prediction expected output. The dataset stores expected as
    a stringified Python list, e.g. "['No']" or "[265.0]" or
    "['option1', 'option2']". Return the list elements as plain strings.
    Falls back to [expected] if it doesn't look like a list literal.
    """
    s = (expected or "").strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return [s]


def _grade_fp_exact(expected: str, got: str) -> bool:
    """
    Lenient exact-ish grader for future_prediction in --no-judge mode.
    Strategy:
      1. Strip \\boxed{...} from `got`.
      2. Parse `expected` as a list literal.
      3. For numeric answers: compare numbers extracted from each side.
      4. For text answers: check normalized containment in either direction
         for ANY expected element.
    Reports True on a match.
    """
    inner = _strip_boxed(got)
    if not inner:
        return False

    choices = _expected_choices(expected)
    if not choices:
        return False

    inner_norm = _normalize(inner)

    # Numeric path: when ANY choice parses as a number, compare numerically.
    inner_nums = _NUMBER_RE.findall(inner)
    for choice in choices:
        choice_nums = _NUMBER_RE.findall(choice)
        if choice_nums and inner_nums:
            try:
                if abs(float(inner_nums[0]) - float(choice_nums[0])) < 1e-3:
                    return True
            except ValueError:
                pass

    # Text path: lenient containment in either direction.
    for choice in choices:
        c_norm = _normalize(choice)
        if not c_norm:
            continue
        if c_norm == inner_norm:
            return True
        if c_norm in inner_norm or inner_norm in c_norm:
            return True
    return False


# ---------------------------------------------------------------------------
# Lenient common_sense grader (eval-only; does NOT change agent output)
# ---------------------------------------------------------------------------

# Honorific titles to strip when comparing person names.
_TITLES_RE = re.compile(
    r"\b(president|vice[- ]president|prime minister|senator|governor|"
    r"mayor|king|queen|prince|princess|duke|duchess|lord|lady|"
    r"sir|dame|dr|doctor|prof|professor|mr|mrs|ms|miss|mx|"
    r"saint|st|reverend|rev|father|sister|brother|judge|justice|"
    r"general|colonel|major|captain|lieutenant|sergeant|admiral)\b\.?",
    re.IGNORECASE,
)

# Unit normalizations: map full forms to canonical short form.
_UNIT_NORMALIZATIONS = [
    (r"\bkilometers?\b", "km"),
    (r"\bkilometre?s?\b", "km"),
    (r"\bmetres?\b", "m"),
    (r"\bmeters?\b", "m"),
    (r"\bcentimet(?:re|er)s?\b", "cm"),
    (r"\bmillimet(?:re|er)s?\b", "mm"),
    (r"\bmiles?\b", "mi"),
    (r"\binches?\b", "in"),
    (r"\bfeet\b", "ft"),
    (r"\bfoot\b", "ft"),
    (r"\byards?\b", "yd"),
    (r"\bkilograms?\b", "kg"),
    (r"\bgrams?\b", "g"),
    (r"\bpounds?\b", "lb"),
    (r"\bounces?\b", "oz"),
    (r"\bseconds?\b", "s"),
    (r"\bminutes?\b", "min"),
    (r"\bhours?\b", "h"),
    (r"\bpercent\b", "%"),
]

# Trailing measurement-context words to strip ("6.213 km long" -> "6.213 km").
_TRAILING_MEASURE_WORDS_RE = re.compile(
    r"\s+(long|wide|tall|deep|high|across|in length|in width|"
    r"in height|in depth|away|old)\b",
    re.IGNORECASE,
)

# Generic place-name prefixes that indicate the same place
# (eval-only -- "New Delhi" should match "Delhi", "Greater London" should
# match "London", etc.). One-way: prefix word can be removed but we don't
# add words.
_PLACE_PREFIX_RE = re.compile(
    r"^(new|greater|north|south|east|west|upper|lower|saint|st\.?|el|la|le|los|las)\s+",
    re.IGNORECASE,
)

# Generic suffixes on places ("London City", "Manhattan Borough", etc.)
_PLACE_SUFFIX_RE = re.compile(
    r"\s+(city|borough|county|harbour|harbor|district|region|province|"
    r"state|country|island|peninsula|territory|metropolitan area)$",
    re.IGNORECASE,
)


def _lenient_normalise(s: str) -> str:
    """Aggressive normalization: lowercase, strip titles, normalize units,
    strip trailing measurement context, drop articles."""
    if not s:
        return ""
    out = s.strip().lower()
    # Strip honorific titles (eg "President Richard Nixon" -> "Richard Nixon").
    out = _TITLES_RE.sub("", out)
    # Normalize units (kilometers -> km, etc.).
    for pat, rep in _UNIT_NORMALIZATIONS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    # Strip trailing measurement-context words ("long", "wide", "old").
    out = _TRAILING_MEASURE_WORDS_RE.sub("", out)
    # Drop leading articles.
    out = re.sub(r"^(the|a|an)\s+", "", out)
    # Drop apostrophes entirely so "sophie's" matches "sophies".
    out = out.replace("'", "").replace("’", "")
    # Strip in-number commas BEFORE general punctuation removal so "1,234"
    # collapses to "1234" rather than "1 234".
    out = re.sub(r"(?<=\d),(?=\d)", "", out)
    # Strip punctuation that doesn't carry meaning.
    out = re.sub(r"[^\w\s\-]", " ", out)
    # Collapse whitespace.
    out = re.sub(r"\s+", " ", out).strip()
    # Re-collapse a stray possessive "s" left over from inputs like
    # "sophie s choice" (where the dataset already pre-stripped the apostrophe).
    out = re.sub(r"\b(\w+)\s+s\b", r"\1s", out)
    return out


def _place_variants(s: str) -> List[str]:
    """Generate accepted variants of a place name for matching."""
    if not s:
        return []
    base = _lenient_normalise(s)
    variants = {base}
    # Strip one place-prefix ("new delhi" -> "delhi").
    stripped_prefix = _PLACE_PREFIX_RE.sub("", base, count=1)
    if stripped_prefix and stripped_prefix != base:
        variants.add(stripped_prefix.strip())
    # Strip one place-suffix ("london city" -> "london").
    stripped_suffix = _PLACE_SUFFIX_RE.sub("", base, count=1)
    if stripped_suffix and stripped_suffix != base:
        variants.add(stripped_suffix.strip())
    # Both directions ("greater london city" -> "london").
    both = _PLACE_SUFFIX_RE.sub("", _PLACE_PREFIX_RE.sub("", base, count=1), count=1)
    if both and both != base:
        variants.add(both.strip())
    return [v for v in variants if v]


def _grade_cs_lenient(expected: str, got: str) -> bool:
    """
    Eval-only lenient grader for common_sense. Catches the 4 documented
    failure modes (titles, place prefixes, units, modifier words) without
    changing what the agent outputs.

    Match levels (return True on first hit):
      1. Both normalised forms are equal.
      2. Either is a substring of the other after normalisation
         (handles 'ethyl alcohol' vs 'alcohol').
      3. Place-variant cross-match (handles 'New Delhi' vs 'Delhi').
      4. Token-set: every token in expected appears in got's normalised text
         (handles 'Bo Donaldson and The Heywoods' vs 'bo donaldson heywoods').

    Crucially conservative: does NOT match unrelated entities. 'Mumbai' will
    NOT match 'Delhi' even though both are cities.
    """
    e_norm = _lenient_normalise(expected)
    g_norm = _lenient_normalise(got)
    if not e_norm or not g_norm:
        return False

    # 1. Direct equality after normalization.
    if e_norm == g_norm:
        return True

    # 2. Substring containment (in either direction). Conservative: only
    # accept if the shorter side is at least 3 chars (avoids accepting "a"
    # as a substring of any answer).
    if len(e_norm) >= 3 and e_norm in g_norm:
        return True
    if len(g_norm) >= 3 and g_norm in e_norm:
        return True

    # 3. Place-variant cross-match.
    e_vars = set(_place_variants(expected))
    g_vars = set(_place_variants(got))
    if e_vars & g_vars:
        return True

    # 4. Token-set: do all expected tokens appear in got?
    # Only when expected has >= 2 tokens (single-token answers should already
    # match via #1 or #2; using #4 on a single token risks false positives).
    e_tokens = set(e_norm.split())
    g_tokens = set(g_norm.split())
    if len(e_tokens) >= 2 and e_tokens.issubset(g_tokens):
        return True
    # And the reverse (got tokens are a subset of expected).
    if len(g_tokens) >= 2 and g_tokens.issubset(e_tokens):
        return True

    return False


def _grade(domain: str, question: str, expected: str, got: str, use_judge: bool) -> bool:
    """Dispatch grader. Honours --judge for both domains; otherwise uses
    the appropriate exact-match grader per domain."""
    if not got:
        return False
    if use_judge:
        return grade_llm_judge(question, expected, got)
    if domain == "future_prediction":
        return _grade_fp_exact(expected, got)
    if domain == "common_sense":
        # Try strict exact-match first (fastest), then lenient on miss.
        if grade_exact(expected, got):
            return True
        return _grade_cs_lenient(expected, got)
    return grade_exact(expected, got)


# ---------------------------------------------------------------------------
# Dev-data loading + filtering
# ---------------------------------------------------------------------------

def _load_dev() -> List[Dict]:
    if not DEV_PATH.exists():
        print(f"ERROR: dev data not found.")
        print(f"Looked in:")
        for p in _DEV_CANDIDATES:
            print(f"  - {p}  ({'EXISTS' if p.exists() else 'missing'})")
        print(f"Drop the file in any of those locations (data/ is preferred).")
        sys.exit(1)
    try:
        with DEV_PATH.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except json.JSONDecodeError as e:
        print(f"ERROR: dev data is not valid JSON: {e}")
        sys.exit(1)
    if not isinstance(data, list):
        print("ERROR: dev data must be a list of records.")
        sys.exit(1)
    return data


def _select(records: List[Dict], domain: str, n: int) -> List[Tuple[int, Dict]]:
    """First n records whose `domain` field equals `domain`. Returns (idx, record)."""
    out = []
    for i, r in enumerate(records):
        if r.get("domain") == domain:
            out.append((i, r))
            if len(out) >= n:
                break
    return out


# ---------------------------------------------------------------------------
# Atomic checkpoint write + summary builder (so we can save mid-run)
# ---------------------------------------------------------------------------

def _save_atomic(payload: Dict, path: Path) -> None:
    """Write JSON to path atomically: write .tmp, fsync, replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _build_summary(rows: List[Dict]) -> Dict:
    """Pure data builder -- same shape as _report but does not print."""
    by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    summary: Dict = {"by_domain": {}}
    for d, drows in by_domain.items():
        n = len(drows)
        correct = sum(1 for r in drows if r["correct"])
        total_calls = sum(r["calls"] for r in drows)
        summary["by_domain"][d] = {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n if n else 0, 4),
            "avg_calls": round(total_calls / n if n else 0, 2),
            "max_calls": max((r["calls"] for r in drows), default=0),
            "over_cap": sum(1 for r in drows if r["calls"] > 18),
            "errors": sum(1 for r in drows if r["error"]),
        }
    n_all = len(rows)
    correct_all = sum(1 for r in rows if r["correct"])
    summary["overall"] = {
        "n": n_all,
        "correct": correct_all,
        "accuracy": round(correct_all / n_all if n_all else 0, 4),
        "over_cap": sum(1 for r in rows if r["calls"] > 18),
    }
    return summary


def _load_existing(path: Path) -> List[Dict]:
    """Load previously-saved results for --resume. Empty list if file missing/bad."""
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict) and isinstance(data.get("results"), list):
            return data["results"]
    except (json.JSONDecodeError, OSError):
        pass
    return []


# ---------------------------------------------------------------------------
# Run one batch
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _process_one_question(
    i: int,
    total: int,
    domain: str,
    solver_fn,
    idx: int,
    rec: Dict,
    use_judge: bool,
) -> Dict:
    """
    Solve + grade ONE question. Designed to run in its own thread so the
    per-question call counter (a contextvars cell installed below) is
    isolated from concurrently-running questions.
    """
    question = rec.get("input", "")
    expected = str(rec.get("output", ""))

    # Install a fresh per-question counter cell IN THIS THREAD'S CONTEXT.
    # Sub-pools spawned by parallel.call_llm_concurrent will copy this
    # context and increment the SAME cell.
    reset_per_question_counter()

    t0 = time.time()
    try:
        answer = solver_fn(question)
        err = None
    except Exception as e:
        answer = ""
        err = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    calls = get_per_question_calls()

    if err:
        correct = False
    else:
        correct = _grade(domain, question, expected, answer, use_judge)

    # Build the per-question report lines as one string so concurrent
    # workers don't interleave their output mid-line.
    q_preview = (question or "").replace("\n", " ")
    if len(q_preview) > 90:
        q_preview = q_preview[:87] + "..."
    ans_preview = (answer or "").replace("\n", " ")
    if len(ans_preview) > 90:
        ans_preview = ans_preview[:87] + "..."
    exp_preview = (expected or "").replace("\n", " ")
    if len(exp_preview) > 60:
        exp_preview = exp_preview[:57] + "..."

    flag = "OK" if correct else "  "
    if err:
        flag = "ER"
    elif calls > 19:
        flag = "!!"

    block = (
        f"\n[{i:>3}/{total}] {flag} idx={idx} calls={calls} "
        f"time={elapsed:.1f}s correct={correct}\n"
        f"      Q: {q_preview}\n"
        f"      EXPECTED: {exp_preview}\n"
        f"      GOT     : {ans_preview!r}"
    )
    if err:
        block += f"\n      ERROR   : {err}"

    with _print_lock:
        print(block, flush=True)

    return {
        "domain": domain,
        "dev_index": idx,
        "question": question,
        "expected": expected,
        "answer": answer,
        "correct": correct,
        "calls": calls,
        "elapsed_sec": round(elapsed, 2),
        "error": err,
    }


def _run_one_domain(
    label: str,
    domain: str,
    solver_fn,
    items: List[Tuple[int, Dict]],
    use_judge: bool,
    workers: int,
    on_row_done=None,
) -> List[Dict]:
    """
    Run all `items` of one domain through `solver_fn`. When workers > 1, runs
    questions concurrently via ThreadPoolExecutor; per-question contextvars
    keep each question's call counter isolated.
    """
    with _print_lock:
        print(f"\n{'=' * 78}")
        print(f"{label}  ({len(items)} questions, {workers} workers)")
        print('=' * 78, flush=True)

    rows: List[Dict] = []

    if workers <= 1 or len(items) <= 1:
        # Sequential path -- preserves prior behaviour exactly.
        for i, (idx, rec) in enumerate(items, 1):
            row = _process_one_question(i, len(items), domain, solver_fn, idx, rec, use_judge)
            rows.append(row)
            if on_row_done is not None:
                on_row_done(row)
        return rows

    # Parallel path. Each worker thread starts in a freshly-copied context,
    # then immediately installs its own per-question counter cell. Sub-pools
    # spawned inside the solver inherit that cell.
    from parallel import submit_in_context  # local import to avoid cycles
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for i, (idx, rec) in enumerate(items, 1):
            fut = submit_in_context(
                pool,
                _process_one_question,
                i, len(items), domain, solver_fn, idx, rec, use_judge,
            )
            futs[fut] = (i, idx)

        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
            if on_row_done is not None:
                on_row_done(row)

    # Re-sort rows by their position in `items` so the saved JSON has a
    # predictable order regardless of completion order.
    item_order = {idx: i for i, (idx, _) in enumerate(items)}
    rows.sort(key=lambda r: item_order.get(r["dev_index"], 1_000_000))
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report(rows: List[Dict], use_judge: bool) -> Dict:
    """Print human-readable report and return the same summary that's saved to disk."""
    summary = _build_summary(rows)

    print("\n" + "=" * 78)
    print(f"DEV-SET RESULTS  (grader = {'LLM-as-judge' if use_judge else 'exact match'})")
    print("=" * 78)
    print(f"  {'Domain':<22}{'N':>5}{'Correct':>10}{'Acc':>8}"
          f"{'AvgCalls':>10}{'MaxCalls':>10}{'OverCap':>9}{'Errs':>6}")
    print("  " + "-" * 76)

    for d, s in sorted(summary["by_domain"].items()):
        print(f"  {d:<22}{s['n']:>5}{s['correct']:>10}{s['accuracy']:>7.1%}"
              f"{s['avg_calls']:>10.2f}{s['max_calls']:>10}{s['over_cap']:>9}{s['errors']:>6}")

    o = summary["overall"]
    total_calls = sum(r["calls"] for r in rows)
    max_calls = max((r["calls"] for r in rows), default=0)
    errs_all = sum(1 for r in rows if r["error"])
    print("  " + "-" * 76)
    print(f"  {'OVERALL':<22}{o['n']:>5}{o['correct']:>10}{o['accuracy']:>7.1%}"
          f"{(total_calls / o['n'] if o['n'] else 0):>10.2f}"
          f"{max_calls:>10}{o['over_cap']:>9}{errs_all:>6}")

    if o["over_cap"]:
        print("\n  BUDGET VIOLATIONS (calls > 18):")
        for r in rows:
            if r["calls"] > 18:
                print(f"     - {r['domain']} idx={r['dev_index']} calls={r['calls']}")
    else:
        print("\n  No budget violations (all questions <= 18 calls).")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cs", type=int, default=50,
                    help="Common-sense items (default 50). Each = ~5 solver calls.")
    ap.add_argument("--fp", type=int, default=50,
                    help="Future-prediction items (default 50). Each = ~4 solver calls.")
    ap.add_argument("--judge", action="store_true",
                    help="Use LLM-as-judge to grade (more accurate for open-ended). "
                         "Adds +1 call per graded question.")
    ap.add_argument("--no-save", action="store_true",
                    help="Don't write dev_eval_results.json.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions already present in dev_eval_results.json. "
                         "Use after a crash / VPN drop / Ctrl-C to pick up where you left off.")
    ap.add_argument("--workers", type=int, default=3,
                    help="Number of questions to solve concurrently. Default 3. "
                         "Each worker has its own per-question call counter; the "
                         "global API semaphore (parallel.DEFAULT_API_CONCURRENCY) "
                         "caps total in-flight calls so we don't get throttled.")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Paste your key into .env.")
        return 1

    records = _load_dev()
    print(f"Loaded {len(records)} dev records from:")
    print(f"  {DEV_PATH}")
    if args.judge:
        print("  Grader: LLM-as-judge  (+1 call per graded question)")
    else:
        print("  Grader: exact-then-LENIENT (cs) + boxed-list match (fp), no extra calls")

    # ----- Resume support: load any prior results and skip those (domain, idx) pairs.
    all_rows: List[Dict] = []
    done_keys: set = set()
    if args.resume:
        prior = _load_existing(RESULTS_OUT)
        if prior:
            all_rows = prior
            done_keys = {(r["domain"], r["dev_index"]) for r in prior}
            print(f"  Resume: loaded {len(prior)} prior rows from {RESULTS_OUT.name}; "
                  f"will skip those.")
        else:
            print(f"  Resume: no prior results at {RESULTS_OUT.name}; starting fresh.")
    elif RESULTS_OUT.exists() and not args.no_save:
        # Safety: warn before overwriting an existing result file without --resume.
        print(f"  WARNING: {RESULTS_OUT.name} exists and will be OVERWRITTEN. "
              f"Re-run with --resume to keep prior rows. Continuing in 3s...")
        time.sleep(3)

    cs_items = _select(records, "common_sense", args.cs)
    fp_items = _select(records, "future_prediction", args.fp)
    if len(cs_items) < args.cs:
        print(f"  NOTE: only {len(cs_items)} common_sense items in dev set.")
    if len(fp_items) < args.fp:
        print(f"  NOTE: only {len(fp_items)} future_prediction items in dev set.")

    # Filter out items already done (resume).
    cs_remaining = [(i, r) for (i, r) in cs_items if ("common_sense", i) not in done_keys]
    fp_remaining = [(i, r) for (i, r) in fp_items if ("future_prediction", i) not in done_keys]
    skipped_cs = len(cs_items) - len(cs_remaining)
    skipped_fp = len(fp_items) - len(fp_remaining)
    if skipped_cs or skipped_fp:
        print(f"  Skipping {skipped_cs} cs + {skipped_fp} fp items already done.")

    solver_calls = len(cs_remaining) * 5 + len(fp_remaining) * 4
    judge_calls = (len(cs_remaining) + len(fp_remaining)) if args.judge else 0
    print(f"\nPlanned cost (this invocation):")
    print(f"  solver calls : ~{solver_calls}")
    print(f"  judge calls  : {judge_calls}  ({'on' if args.judge else 'off'})")
    print(f"  TOTAL        : ~{solver_calls + judge_calls} LLM calls")
    if not args.no_save:
        print(f"  Checkpoint   : {RESULTS_OUT.name} written after every question")
    try:
        from parallel import api_concurrency_limit
        print(f"  Concurrency  : {args.workers} questions in parallel; "
              f"global API cap {api_concurrency_limit()} in-flight calls")
    except Exception:
        pass
    print()

    # ----- Per-row save closure: thread-safe append + atomic file write.
    checkpoint_lock = threading.Lock()

    def _on_row(row: Dict) -> None:
        with checkpoint_lock:
            all_rows.append(row)
            if args.no_save:
                return
            payload = {"summary": _build_summary(all_rows), "results": all_rows}
            try:
                _save_atomic(payload, RESULTS_OUT)
            except OSError as e:
                print(f"  [checkpoint warning] could not write {RESULTS_OUT.name}: {e}")

    interrupted = False
    wall_t0 = time.time()
    try:
        if cs_remaining:
            _run_one_domain(
                "common_sense  (Step-Back + RECITE + Adaptive SC + USC)",
                "common_sense", common_sense.solve, cs_remaining, args.judge,
                workers=args.workers,
                on_row_done=_on_row,
            )
        if fp_remaining:
            _run_one_domain(
                "future_prediction  (Plan-and-Solve + Self-Refine + Ensemble)",
                "future_prediction", future_prediction.solve, fp_remaining, args.judge,
                workers=args.workers,
                on_row_done=_on_row,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[interrupted] Ctrl-C received. Partial results were checkpointed "
              f"to {RESULTS_OUT.name}.")
        print("Re-run with --resume to continue.")

    wall_elapsed = time.time() - wall_t0
    summary = _report(all_rows, args.judge)

    # Final save (no-op if checkpoint already wrote everything).
    if not args.no_save and all_rows:
        _save_atomic({"summary": summary, "results": all_rows}, RESULTS_OUT)
        print(f"\nFull per-question results saved to {RESULTS_OUT.name} "
              f"({len(all_rows)} rows).")

    new_rows = [r for r in all_rows if (r["domain"], r["dev_index"]) not in done_keys]
    if new_rows:
        avg_q = wall_elapsed / len(new_rows)
        print(f"\nWall time: {wall_elapsed:.1f}s for {len(new_rows)} new questions "
              f"({avg_q:.1f}s/question wall, {args.workers} workers).")

    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
