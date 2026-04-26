# Ismail - shared evaluation harness
"""
Evaluation harness for CSE 476 Final Project.

Runs the agent on the 1,000-example dev set and reports:
  - overall accuracy
  - per-domain accuracy (math, coding, common_sense, future_prediction, planning)
  - total LLM calls used
  - average calls per question

Usage:
    python evaluator.py --n 20                  # quick smoke test (20 examples)
    python evaluator.py --domains planning      # only planning examples
    python evaluator.py                         # full 1,000-example run
    python evaluator.py --no-judge              # skip LLM-as-judge (cheaper)

Grading:
  - math   : numeric extraction + float compare
  - coding : substring containment + LLM-as-judge fallback
  - others : LLM-as-judge (uses one extra LLM call per graded answer)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent import run_agent
from utils import call_llm, get_call_count, reset_call_count


# Dev data lives in final_project_tutorial_and_dev_data/ in this repo layout.
# Falls back to data/ if a teammate dropped it there instead.
_HERE = Path(__file__).resolve().parent
_DEV_CANDIDATES = [
    _HERE / "final_project_tutorial_and_dev_data" / "cse476_final_project_dev_data.json",
    _HERE / "data" / "cse476_final_project_dev_data.json",
]
DEV_PATH = next((p for p in _DEV_CANDIDATES if p.exists()), _DEV_CANDIDATES[0])


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_number(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return m.group(0) if m else None


def grade_math(expected: str, got: str) -> bool:
    en, gn = _extract_number(expected), _extract_number(got)
    if en is not None and gn is not None:
        try:
            return abs(float(en) - float(gn)) < 1e-6
        except ValueError:
            pass
    return _normalize(got) == _normalize(expected)


def grade_exact(expected: str, got: str) -> bool:
    return _normalize(got) == _normalize(expected)


def grade_llm_judge(question: str, expected: str, got: str) -> bool:
    """LLM-as-judge. Uses one extra LLM call."""
    system = (
        "You are a strict grader. Reply with exactly 'True' or 'False' - "
        "no punctuation, no explanation."
    )
    prompt = (
        f"QUESTION:\n{question[:2000]}\n\n"
        f"EXPECTED:\n{expected[:500]}\n\n"
        f"PREDICTION:\n{got[:2000]}\n\n"
        "Would the prediction be accepted as correct? Reply True or False."
    )
    reply = (call_llm(prompt, system=system, temperature=0.0, max_tokens=5) or "").strip().lower()
    if reply.startswith("true"):
        return True
    if reply.startswith("false"):
        return False
    return _normalize(got) == _normalize(expected)


def grade(domain: str, question: str, expected: str, got: str, use_judge: bool) -> bool:
    if domain == "math":
        return grade_math(expected, got)
    if domain == "coding":
        if _normalize(expected) in _normalize(got) or _normalize(got) in _normalize(expected):
            return True
        return grade_llm_judge(question, expected, got) if use_judge else False
    if use_judge:
        return grade_llm_judge(question, expected, got)
    return grade_exact(expected, got)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_eval(
    n: Optional[int] = None,
    domains: Optional[List[str]] = None,
    use_judge: bool = True,
) -> Dict[str, Any]:
    with DEV_PATH.open() as fp:
        data = json.load(fp)

    if domains:
        data = [d for d in data if d["domain"] in domains]
    if n is not None:
        data = data[:n]

    reset_call_count()
    start = time.time()

    per_domain_correct: Dict[str, int] = defaultdict(int)
    per_domain_total: Dict[str, int] = defaultdict(int)
    errors: List[Dict[str, Any]] = []

    for i, ex in enumerate(data):
        domain = ex["domain"]
        question = ex["input"]
        expected = str(ex["output"])

        try:
            got = run_agent(question)
        except Exception as e:
            got = ""
            errors.append({"idx": i, "domain": domain, "error": str(e)})

        is_correct = grade(domain, question, expected, got, use_judge)
        per_domain_total[domain] += 1
        if is_correct:
            per_domain_correct[domain] += 1

        calls_so_far = get_call_count()
        if i % 10 == 0 or i == len(data) - 1:
            print(
                f"[{i + 1:4d}/{len(data)}] {domain:<18} "
                f"correct={is_correct} total_calls={calls_so_far}",
                flush=True,
            )

    elapsed = time.time() - start
    total_calls = get_call_count()

    overall_total = sum(per_domain_total.values())
    overall_correct = sum(per_domain_correct.values())
    report = {
        "total_examples": len(data),
        "total_llm_calls": total_calls,
        "avg_calls_per_question": round(total_calls / max(1, len(data)), 2),
        "elapsed_sec": round(elapsed, 1),
        "overall_accuracy": round(overall_correct / max(1, overall_total), 4),
        "per_domain": {
            d: {
                "correct": per_domain_correct[d],
                "total": per_domain_total[d],
                "accuracy": round(per_domain_correct[d] / max(1, per_domain_total[d]), 4),
            }
            for d in per_domain_total
        },
        "error_count": len(errors),
    }

    print("\n==== EVALUATION SUMMARY ====")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit number of examples")
    ap.add_argument(
        "--domains",
        nargs="+",
        default=None,
        choices=["math", "coding", "common_sense", "future_prediction", "planning"],
    )
    ap.add_argument("--no-judge", action="store_true", help="skip LLM-as-judge grading")
    args = ap.parse_args()

    run_eval(n=args.n, domains=args.domains, use_judge=not args.no_judge)
