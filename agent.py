#Ismail - shared agent entry point
"""
Main agent entry point for CSE 476 Final Project.

`run_agent(question: str) -> str` is the single function the rest of the
codebase (generate_answer_template.py, evaluator.py) uses. It wraps the
per-question LLM budget reset and delegates to the router.
"""

from __future__ import annotations

from router import route_and_solve
from utils import (
    CallBudgetExceeded,
    get_per_question_calls,
    reset_per_question_counter,
)


def run_agent(question: str) -> str:
    """
    Solve one question and return the final answer string.

    Resets the per-question call counter, then routes the question to the
    right domain-specific solver. If the solver tries to exceed our
    18-call-per-question budget, we return whatever partial answer we have
    (or an empty string) rather than crashing the entire run.
    """
    reset_per_question_counter()
    try:
        answer = route_and_solve(question)
    except CallBudgetExceeded:
        # Hard budget hit -- return a safe empty string so the run continues.
        answer = ""

    # Enforce the <5000 character auto-grader limit.
    if answer is None:
        answer = ""
    answer = str(answer).strip()
    if len(answer) > 4900:
        answer = answer[:4900]

    # Helpful visibility when running the eval loop.
    print(f"  [calls used: {get_per_question_calls()}]")
    return answer
