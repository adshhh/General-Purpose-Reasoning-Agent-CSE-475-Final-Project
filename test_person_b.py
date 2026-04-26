#!/usr/bin/env python3
# Owner: Person B (Soul) - smoke test for cs + fp solvers
"""Smoke test the two solvers on a few sample questions; --extract-only skips API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tiny .env loader -- runs BEFORE we import utils, so utils.py picks up the
# key. Avoids needing python-dotenv as a dependency.
# Reads `KEY=VALUE` lines from a .env file next to this script. Lines that
# start with `#` or are blank are ignored. Existing env vars take priority.
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

# Side-effect import: wraps utils.call_llm so every call is logged to
# call_log.jsonl. Run `python monitor.py` in another terminal (or after
# the run with `python monitor.py --summary`) to inspect calls.
import call_logger  # noqa: F401

# Local modules from the project root. Imported AFTER _load_dotenv and
# AFTER call_logger so utils.call_llm is already wrapped when these
# modules do `from utils import call_llm`.
import common_sense
import future_prediction
from agent import run_agent
from utils import get_per_question_calls, reset_per_question_counter


# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

# Representative common-sense items (multiple-choice + short trivia).
COMMON_SENSE_QUESTIONS = [
    "A student walks to school one morning and notices the grass is wet "
    "but the streets are dry. Which of these processes most likely caused "
    "the grass to be wet? A. condensation B. erosion C. evaporation "
    "D. precipitation",
    "If you put a metal spoon in a cup of hot tea, the spoon gets hot. "
    "What property of metal does this demonstrate? "
    "A. magnetism B. conductivity C. flexibility D. transparency",
]

# Representative future-prediction items. The dataset uses \boxed{} answers
# and an instruction to never refuse, so the prompt below mirrors that style.
FUTURE_PREDICTION_QUESTIONS = [
    "Predict whether global average surface temperature in 2030 will be "
    "above or below the 2020 value. IMPORTANT: Your final answer MUST end "
    "with \\boxed{above} or \\boxed{below}. Do not refuse to make a "
    "prediction.",
    "Predict the most likely winner of a hypothetical match between a "
    "current top-10 ATP tennis player and an average club-level player "
    "over best-of-three sets. IMPORTANT: Your final answer MUST end with "
    "\\boxed{top-10 player} or \\boxed{club player}. Do not refuse to "
    "make a prediction.",
]


# ---------------------------------------------------------------------------
# Offline tests for the extractors (no LLM calls)
# ---------------------------------------------------------------------------

def test_extractors_offline() -> None:
    """Sanity-check the two extract_final_answer helpers without any API."""
    print("=== extract_final_answer (common_sense) ===")
    cases = [
        ("The grass is wet because of dew at night. Final answer: A", "A"),
        ("Step 1... Step 2...\n\nAnswer: (C) condensation", "C"),
        ("...so the result is \\boxed{42}.", "42"),
        ("Therefore the species is\nVan Morrison", "Van Morrison"),
        ("Final answer: 0.75", "0.75"),
    ]
    for raw, expected in cases:
        got = common_sense.extract_final_answer(raw)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {raw!r:<60} -> {got!r}  "
              f"(expected {expected!r})")

    print("\n=== extract_final_answer (future_prediction) ===")
    fp_cases = [
        # Boxed answer present -- must be preserved.
        ("Reasoning... Final answer: \\boxed{above}",
         "\\boxed{above}"),
        # Multiple boxed -- last one wins.
        ("First guess \\boxed{below}. After review: \\boxed{above}",
         "\\boxed{above}"),
        # No boxed, but Final answer: tag -- should be re-wrapped.
        ("Reasoning...\nFinal answer: top-10 player",
         "\\boxed{top-10 player}"),
        # No tag at all -- last line, wrapped.
        ("Lots of text.\nProbably above.",
         "\\boxed{Probably above}"),
    ]
    for raw, expected in fp_cases:
        got = future_prediction.extract_final_answer(raw)
        ok = got == expected
        print(f"  [{'OK' if ok else 'FAIL'}] {raw!r:<70} -> {got!r}  "
              f"(expected {expected!r})")


# ---------------------------------------------------------------------------
# Online tests (require OPENAI_API_KEY + VPN)
# ---------------------------------------------------------------------------

def _run_solver(name: str, solver_fn, questions) -> None:
    print(f"\n=== {name} ===")
    for i, q in enumerate(questions, 1):
        reset_per_question_counter()
        try:
            answer = solver_fn(q)
        except Exception as e:
            print(f"  [{i}] CRASHED: {e}")
            continue
        calls = get_per_question_calls()
        preview = (q[:80] + "...") if len(q) > 80 else q
        print(f"  [{i}] q: {preview}")
        print(f"      answer: {answer!r}")
        print(f"      LLM calls used: {calls}")


def _run_full_router(questions) -> None:
    """End-to-end check: question goes through the real agent.run_agent."""
    print("\n=== full router (agent.run_agent) ===")
    for i, q in enumerate(questions, 1):
        try:
            answer = run_agent(q)
        except Exception as e:
            print(f"  [{i}] CRASHED: {e}")
            continue
        preview = (q[:80] + "...") if len(q) > 80 else q
        print(f"  [{i}] q: {preview}")
        print(f"      answer: {answer!r}")


def test_online(cs_n: int, fp_n: int, run_router: bool) -> None:
    """Run the LLM-backed checks. Caller controls how many questions per pipeline."""
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set -- skipping online tests.")
        print("Paste your key into the .env file in this folder, then "
              "re-run.")
        return

    if cs_n > 0:
        _run_solver(
            f"common_sense.solve  (Self-Consistency, 5 samples @ T=0.7)  "
            f"-- {cs_n} q x ~5 calls = ~{cs_n * 5} calls",
            common_sense.solve,
            COMMON_SENSE_QUESTIONS[:cs_n],
        )

    if fp_n > 0:
        _run_solver(
            f"future_prediction.solve  (Plan-and-Solve + Self-Refine)  "
            f"-- {fp_n} q x ~4 calls = ~{fp_n * 4} calls",
            future_prediction.solve,
            FUTURE_PREDICTION_QUESTIONS[:fp_n],
        )

    if run_router:
        _run_full_router(
            COMMON_SENSE_QUESTIONS[:1] + FUTURE_PREDICTION_QUESTIONS[:1]
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Smoke-test Person B's solvers cheaply. Default = 1 common_sense "
            "question (~5 calls) + 1 future_prediction question (~4 calls) "
            "= ~9 LLM calls total."
        )
    )
    ap.add_argument(
        "--extract-only",
        action="store_true",
        help="Run only the offline extractor tests. ZERO LLM calls.",
    )
    ap.add_argument(
        "--cs",
        type=int,
        default=1,
        help="How many common_sense questions to run (each = ~5 calls). "
             "Set 0 to skip. Default 1.",
    )
    ap.add_argument(
        "--fp",
        type=int,
        default=1,
        help="How many future_prediction questions to run (each = ~4 calls). "
             "Set 0 to skip. Default 1.",
    )
    ap.add_argument(
        "--router",
        action="store_true",
        help="Also run questions through agent.run_agent end-to-end. "
             "Adds another ~9 calls. Off by default.",
    )
    args = ap.parse_args()

    # Clamp counts to what we actually have.
    cs_n = max(0, min(args.cs, len(COMMON_SENSE_QUESTIONS)))
    fp_n = max(0, min(args.fp, len(FUTURE_PREDICTION_QUESTIONS)))

    test_extractors_offline()
    if not args.extract_only:
        test_online(cs_n=cs_n, fp_n=fp_n, run_router=args.router)
    return 0


if __name__ == "__main__":
    sys.exit(main())
