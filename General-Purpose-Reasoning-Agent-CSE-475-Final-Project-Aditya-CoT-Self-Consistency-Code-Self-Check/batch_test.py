#!/usr/bin/env python3
"""Run domain-filtered test items through the solvers and print a results table for review."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# .env loader (same as test_person_b.py -- runs before utils import)
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

# Project modules.
import common_sense
import future_prediction
from router import classify_by_keywords
from utils import get_per_question_calls, reset_per_question_counter


TEST_DATA = Path(__file__).resolve().parent / "data" / "cse_476_final_project_test_data.json"
RESULTS_OUT = Path(__file__).resolve().parent / "batch_results.json"


# ---------------------------------------------------------------------------
# Pick questions by domain using the free keyword classifier
# ---------------------------------------------------------------------------

def _pick_by_domain(
    questions: List[Dict],
    target_domain: str,
    n: int,
    offset: int,
) -> List[Tuple[int, str]]:
    """
    Walk the test set in order, classify each item by keywords (0 LLM calls),
    return the first `n` items whose classification matches `target_domain`,
    starting from `offset`.

    Returns list of (original_index, question_text).
    """
    found: List[Tuple[int, str]] = []
    skipped = 0
    for idx, q in enumerate(questions):
        text = q.get("input", "")
        if classify_by_keywords(text) != target_domain:
            continue
        if skipped < offset:
            skipped += 1
            continue
        found.append((idx, text))
        if len(found) >= n:
            break
    return found


# ---------------------------------------------------------------------------
# Run a batch through one solver
# ---------------------------------------------------------------------------

def _run_batch(
    label: str,
    solver_fn,
    items: List[Tuple[int, str]],
    expected_calls: int,
) -> List[Dict]:
    print(f"\n{'=' * 78}")
    print(f"{label}")
    print(f"  {len(items)} questions × ~{expected_calls} calls = ~{len(items) * expected_calls} calls expected")
    print('=' * 78)

    results: List[Dict] = []
    for i, (orig_idx, q) in enumerate(items, 1):
        reset_per_question_counter()
        t0 = time.time()
        try:
            answer = solver_fn(q)
            err = None
        except Exception as e:
            answer = ""
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0
        calls = get_per_question_calls()

        # Compact human-readable line.
        q_preview = q.replace("\n", " ")
        if len(q_preview) > 110:
            q_preview = q_preview[:107] + "..."
        ans_preview = (answer or "").replace("\n", " ")
        if len(ans_preview) > 110:
            ans_preview = ans_preview[:107] + "..."

        flag = "  "
        if err:
            flag = "ER"
        elif calls > 18:
            flag = "!!"
        elif not answer:
            flag = "??"

        print(f"\n[{i:>3}/{len(items)}] {flag} idx={orig_idx} calls={calls} time={elapsed:.1f}s")
        print(f"      Q: {q_preview}")
        print(f"      A: {ans_preview!r}")
        if err:
            print(f"      ERROR: {err}")

        results.append({
            "domain_label": label,
            "test_index": orig_idx,
            "question": q,
            "answer": answer,
            "calls": calls,
            "elapsed_sec": round(elapsed, 2),
            "error": err,
        })
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary(results: List[Dict]) -> None:
    if not results:
        return
    n = len(results)
    n_err = sum(1 for r in results if r["error"])
    n_empty = sum(1 for r in results if not r["answer"] and not r["error"])
    total_calls = sum(r["calls"] for r in results)
    max_calls = max(r["calls"] for r in results)
    over_cap = sum(1 for r in results if r["calls"] > 18)

    avg = total_calls / n if n else 0
    total_t = sum(r["elapsed_sec"] for r in results)

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    print(f"  total questions    : {n}")
    print(f"  total LLM calls    : {total_calls}")
    print(f"  avg calls/question : {avg:.1f}")
    print(f"  max calls/question : {max_calls}  (hard cap = 20)")
    print(f"  over-cap questions : {over_cap}")
    print(f"  errored questions  : {n_err}")
    print(f"  empty answers      : {n_empty}")
    print(f"  total wall time    : {total_t:.1f}s ({total_t / 60:.1f} min)")
    print(f"\n  Full results saved to {RESULTS_OUT.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cs", type=int, default=20,
                    help="Common-sense items to run (default 20).")
    ap.add_argument("--fp", type=int, default=10,
                    help="Future-prediction items to run (default 10).")
    ap.add_argument("--offset", type=int, default=0,
                    help="Skip the first N matches per domain (for sampling further "
                         "into the dataset). Default 0.")
    ap.add_argument("--no-save", action="store_true",
                    help="Don't write batch_results.json.")
    args = ap.parse_args()

    if not TEST_DATA.exists():
        print(f"ERROR: {TEST_DATA} not found.")
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Paste your key into .env.")
        return 1

    with TEST_DATA.open("r", encoding="utf-8") as fp:
        questions = json.load(fp)
    print(f"Loaded {len(questions)} test questions from {TEST_DATA.name}.")

    # Pre-flight: tell the user the expected cost BEFORE we start spending.
    expected_total = args.cs * 5 + args.fp * 4
    print(f"\nPlanned run:")
    print(f"  common_sense       : {args.cs}  questions × 5 calls = {args.cs * 5} calls")
    print(f"  future_prediction  : {args.fp}  questions × 4 calls = {args.fp * 4} calls")
    print(f"  ESTIMATED TOTAL    : ~{expected_total} LLM calls")
    print(f"  (offset = {args.offset})\n")

    all_results: List[Dict] = []

    if args.cs > 0:
        cs_items = _pick_by_domain(questions, "common_sense", args.cs, args.offset)
        if len(cs_items) < args.cs:
            print(f"  NOTE: only found {len(cs_items)} common_sense items "
                  f"(asked for {args.cs}).")
        all_results += _run_batch(
            "common_sense  (Self-Consistency, 5 samples @ T=0.7)",
            common_sense.solve,
            cs_items,
            expected_calls=5,
        )

    if args.fp > 0:
        fp_items = _pick_by_domain(questions, "future_prediction", args.fp, args.offset)
        if len(fp_items) < args.fp:
            print(f"  NOTE: only found {len(fp_items)} future_prediction items "
                  f"(asked for {args.fp}).")
        all_results += _run_batch(
            "future_prediction  (Plan-and-Solve + Self-Refine)",
            future_prediction.solve,
            fp_items,
            expected_calls=4,
        )

    _summary(all_results)

    if not args.no_save and all_results:
        with RESULTS_OUT.open("w", encoding="utf-8") as fp:
            json.dump(all_results, fp, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
