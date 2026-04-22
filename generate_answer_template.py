#!/usr/bin/env python3
"""
Generate the final answer file for the CSE 476 auto-grader.

Reads cse_476_final_project_test_data.json from ./data, runs each question
through the agent (agent.run_agent), and writes cse_476_final_project_answers.json.

Usage:
    python generate_answer_template.py                 # process all ~6200 questions
    python generate_answer_template.py --n 50          # first 50 only (smoke test)
    python generate_answer_template.py --resume        # skip questions already answered
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from agent import run_agent
from utils import get_call_count, reset_call_count


INPUT_PATH = Path("data/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("data/cse_476_final_project_answers.json")


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def load_existing_answers(path: Path) -> List[Dict[str, str]]:
    """Load a partial answers file if it exists (for --resume)."""
    if not path.exists():
        return []
    try:
        with path.open("r") as fp:
            data = json.load(fp)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def build_answers(
    questions: List[Dict[str, Any]],
    existing: List[Dict[str, str]],
    resume: bool,
) -> List[Dict[str, str]]:
    answers: List[Dict[str, str]] = list(existing) if resume else []
    start_idx = len(answers) if resume else 0
    if resume and start_idx:
        print(f"Resuming from question {start_idx + 1} (already have {start_idx}).")

    reset_call_count()
    t0 = time.time()
    for idx in range(start_idx, len(questions)):
        q = questions[idx]
        try:
            answer = run_agent(q["input"])
        except Exception as e:
            print(f"[{idx + 1}] crashed: {e}")
            answer = ""
        answers.append({"output": answer})

        if (idx + 1) % 10 == 0 or idx == len(questions) - 1:
            elapsed = time.time() - t0
            rate = (idx + 1 - start_idx) / max(1, elapsed)
            calls = get_call_count()
            print(
                f"[{idx + 1}/{len(questions)}] "
                f"calls={calls} "
                f"avg={calls / max(1, idx + 1 - start_idx):.1f}/q "
                f"rate={rate:.2f} q/s",
                flush=True,
            )
            # Periodic checkpoint so a crash doesn't lose everything.
            _write(OUTPUT_PATH, answers)

    return answers


def validate_results(
    questions: List[Dict[str, Any]], answers: List[Dict[str, Any]]
) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, answer in enumerate(answers):
        if "output" not in answer:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(answer["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(answer['output'])}"
            )
        if len(answer["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(answer['output'])} chars)."
            )


def _write(path: Path, answers: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit number of questions")
    ap.add_argument("--resume", action="store_true", help="skip questions already answered")
    args = ap.parse_args()

    questions = load_questions(INPUT_PATH)
    if args.n:
        questions = questions[: args.n]

    existing = load_existing_answers(OUTPUT_PATH) if args.resume else []
    answers = build_answers(questions, existing, resume=args.resume)

    _write(OUTPUT_PATH, answers)
    with OUTPUT_PATH.open("r") as fp:
        saved = json.load(fp)
    validate_results(questions, saved)
    print(
        f"\nWrote {len(answers)} answers to {OUTPUT_PATH}. "
        f"Total LLM calls: {get_call_count()}."
    )


if __name__ == "__main__":
    main()