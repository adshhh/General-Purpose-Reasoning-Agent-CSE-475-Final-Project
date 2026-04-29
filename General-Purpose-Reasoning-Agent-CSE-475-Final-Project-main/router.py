import re
import sys
from utils import call_llm, CallBudgetExceeded


def _solve_math(q):
    from math_solver import solve as f
    return f(q)


def _solve_coding(q):
    from coding import solve as f
    return f(q)


def _solve_common_sense(q):
    from common_sense import solve as f
    return f(q)


def _solve_future_prediction(q):
    from future_prediction import solve as f
    return f(q)


def _solve_planning(q):
    from planning import solve as f
    return f(q)


SOLVERS = {
    "math": _solve_math,
    "coding": _solve_coding,
    "common_sense": _solve_common_sense,
    "future_prediction": _solve_future_prediction,
    "planning": _solve_planning,
}


_PLANNING_HINTS = [
    "I have to plan",
    "playing with a set of objects",
    "PDDL",
    "[STATEMENT]",
    "Here are the actions I can do",
    "[PLAN]",
]
_CODING_HINTS = [
    "def task_func",
    "write self-contained code",
    "Write a function",
    "import ",
    "Write a Python function",
]
_FUTURE_HINTS = [
    "predict future",
    "predict future events",
    r"\boxed{",
    "MUST end with this exact format",
    "agent that can predict",
]
_MATH_RE = re.compile(
    r"(\\frac|\\sqrt|\\boxed|\\sum|\\int|\\binom|\\pi|"
    r"\\sin|\\cos|\\tan|\\angle|\\triangle|\\circ|\\geq|\\leq|"
    r"\\cdot|\\times|\\equiv|\\pmod|\\ldots|\\dots|"
    r"Find the (area|value|sum|number|smallest|largest|maximum|minimum|"
    r"radius|perimeter|product|remainder)|"
    r"\bCompute\b|What is the value of|Solve for|"
    r"How many (positive |distinct |different |even |odd |)"
    r"(integers|values|ways|pairs|triples|points|solutions|"
    r"polygons|numbers|divisors|digits)|"
    r"the probability that|"
    r"smallest positive integer|largest positive integer)",
    re.IGNORECASE,
)
# LaTeX dollar-math like $ABCD$ or $\angle ABC$ is a strong math signal.
_LATEX_DOLLAR_RE = re.compile(r"\$[^$\n]+\$")
# Reading-comprehension questions need to win over math regex.
_COMMON_SENSE_HINTS = [
    "Answer the question using the context",
    "using the context",
]


def classify_by_keywords(question):
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
    for hint in _COMMON_SENSE_HINTS:
        if hint in q:
            return "common_sense"
    if _MATH_RE.search(q):
        return "math"
    if _LATEX_DOLLAR_RE.search(q):
        return "math"
    if len(q) < 400 and ("?" in q or any(
            w in q.lower() for w in ["which", "what", "who", "where", "why", "how"])):
        return "common_sense"
    return "unknown"


_CLASSIFY_SYS = (
    "Classify the user's question into exactly one of these five categories: "
    "math, coding, common_sense, future_prediction, planning. "
    "Reply with only the single category word, nothing else."
)


def classify_by_llm(question):
    try:
        reply = call_llm(question[:1500], system=_CLASSIFY_SYS, temperature=0.0,
                         max_tokens=10)
    except CallBudgetExceeded:
        raise
    reply = (reply or "").lower().strip()
    for domain in SOLVERS:
        if domain in reply:
            return domain
    # common_sense is the largest dev bucket, so it's a safe default.
    return "common_sense"


def classify_domain(question):
    d = classify_by_keywords(question)
    if d != "unknown":
        return d
    return classify_by_llm(question)


def route_and_solve(question):
    try:
        domain = classify_domain(question)
    except CallBudgetExceeded:
        raise

    solver = SOLVERS.get(domain)
    if solver is None:
        return _fallback_direct(question)

    try:
        return solver(question)
    except CallBudgetExceeded:
        raise
    except ImportError:
        # Teammate module missing -- direct LLM call instead.
        print(f"[router] {domain} solver unavailable; falling back",
              file=sys.stderr, flush=True)
        return _fallback_direct(question)
    except Exception as e:
        print(f"[router] {domain} solver crashed ({type(e).__name__}: {e}); falling back",
              file=sys.stderr, flush=True)
        try:
            return _fallback_direct(question)
        except CallBudgetExceeded:
            raise


def _fallback_direct(question):
    system = "Solve the following question. Reply with ONLY the final answer, no reasoning, no explanation."
    return (call_llm(question, system=system) or "").strip()
