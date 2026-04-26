# Owner: Person C (Ismail) - domain router / dispatcher
"""
Domain router for CSE 476 Final Project.

Given a raw question, decides which domain it belongs to (math, coding,
common_sense, future_prediction, planning) and dispatches to the domain's
solver function.

Uses a cheap keyword-based classifier first (free, 0 LLM calls). Only falls
back to an LLM classifier when the keywords are inconclusive.

Domain modules each export a `solve(question: str) -> str` function.
This file is the single place to register them.
"""

from __future__ import annotations

import re
from typing import Callable, Dict

from utils import call_llm


# ---------------------------------------------------------------------------
# Lazy imports so the router doesn't crash if a teammate's module is missing.
# ---------------------------------------------------------------------------

def _solve_math(q: str) -> str:
    from math_solver import solve as f
    return f(q)


def _solve_coding(q: str) -> str:
    from coding import solve as f
    return f(q)


def _solve_common_sense(q: str) -> str:
    from common_sense import solve as f
    return f(q)


def _solve_future_prediction(q: str) -> str:
    from future_prediction import solve as f
    return f(q)


def _solve_planning(q: str) -> str:
    from planning import solve as f
    return f(q)


SOLVERS: Dict[str, Callable[[str], str]] = {
    "math": _solve_math,
    "coding": _solve_coding,
    "common_sense": _solve_common_sense,
    "future_prediction": _solve_future_prediction,
    "planning": _solve_planning,
}


# ---------------------------------------------------------------------------
# Classifier 1: keyword heuristic (0 LLM calls)
# ---------------------------------------------------------------------------

# Strong signals. Ordered: more specific first.
_PLANNING_HINTS = [
    "I have to plan",
    "I am playing with a set of objects",
    "Here are the actions I can do",
    "[STATEMENT]",
    "My goal is to have",
    "restrictions on my actions",
    "PDDL",
]

_CODING_HINTS = [
    "The function should output with:",
    "def task_func",
    "import ",
    "You should write self-contained code",
    "Retrieves the",  # BigCodeBench style docstrings
]

_FUTURE_HINTS = [
    "predict future",
    "future event",
    "\\boxed{",
    "IMPORTANT: Your final answer MUST end with",
    "Do not refuse to make a prediction",
]

# Math: LaTeX-heavy text or "Find the..." style prompts.
_MATH_RE = re.compile(
    r"(\\frac|\\sqrt|\\boxed|\\sum|\\int|\\pi|\$[^$]+\$|Find the|Compute|"
    r"What is the value|evaluate|determine the)",
    re.IGNORECASE,
)


def classify_by_keywords(question: str) -> str:
    """Return a domain string or 'unknown' if no confident match."""
    q = question

    for hint in _PLANNING_HINTS:
        if hint in q:
            return "planning"

    for hint in _FUTURE_HINTS:
        if hint in q:
            return "future_prediction"

    for hint in _CODING_HINTS:
        if hint in q:
            return "coding"

    if _MATH_RE.search(q):
        return "math"

    # Short factual questions tend to be common sense / trivia.
    if len(q) < 300 and "?" in q:
        return "common_sense"

    return "unknown"


# ---------------------------------------------------------------------------
# Classifier 2: LLM fallback (1 LLM call)
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a strict classifier. Given a user question, reply with ONE word "
    "from this list: math, coding, common_sense, future_prediction, planning. "
    "No punctuation, no explanation."
)


def classify_by_llm(question: str) -> str:
    """One-shot LLM classifier. Returns a valid domain or 'common_sense' default."""
    # Only show the first 1500 chars to keep the classifier cheap.
    snippet = question[:1500]
    reply = call_llm(snippet, system=_CLASSIFIER_SYSTEM, temperature=0.0, max_tokens=10)
    reply = (reply or "").strip().lower()
    for d in SOLVERS:
        if d in reply:
            return d
    return "common_sense"


def classify_domain(question: str) -> str:
    """Keyword first, LLM fallback. Returns one of the 5 domain strings."""
    d = classify_by_keywords(question)
    if d != "unknown":
        return d
    return classify_by_llm(question)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def route_and_solve(question: str) -> str:
    """
    Classify the question and call the matching domain solver. Returns the
    solver's answer string (never None). Falls back to a direct LLM call if
    the chosen solver is not yet implemented.
    """
    domain = classify_domain(question)
    solver = SOLVERS.get(domain)
    if solver is None:
        return _fallback_direct(question)
    try:
        return solver(question)
    except ImportError:
        # The teammate's module isn't written yet -- use a direct LLM call.
        return _fallback_direct(question)
    except Exception as e:
        # Any other solver crash -- don't fail the whole run, fall back.
        print(f"[router] solver for {domain} crashed: {e}. Falling back.")
        return _fallback_direct(question)


def _fallback_direct(question: str) -> str:
    """Cheap 1-call fallback for any domain without a dedicated solver yet."""
    system = (
        "You are a careful problem solver. Read the question, reason briefly, "
        "and reply with ONLY the final answer -- no explanation, no preamble."
    )
    return (call_llm(question, system=system, temperature=0.0) or "").strip()
