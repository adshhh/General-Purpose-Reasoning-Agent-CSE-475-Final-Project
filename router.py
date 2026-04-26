import re
from utils import call_llm

# Code doesn't break if a teammate's file is missing
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

# Heuristic keywords to save API calls
_PLANNING_HINTS = ["I have to plan", "playing with a set of objects", "PDDL", "[STATEMENT]"]
_CODING_HINTS = ["def task_func", "import ", "write self-contained code"]
_FUTURE_HINTS = ["predict future", "\\boxed{", "MUST end with"]
_MATH_RE = re.compile(r"(\\frac|\\sqrt|\\boxed|\\sum|Find the|Compute|What is the value)", re.IGNORECASE)

def classify_by_keywords(question):
    q = question
    for hint in _PLANNING_HINTS:
        if hint in q: return "planning"
    for hint in _FUTURE_HINTS:
        if hint in q: return "future_prediction"
    for hint in _CODING_HINTS:
        if hint in q: return "coding"
    if _MATH_RE.search(q):
        return "math"
    if len(q) < 300 and "?" in q:
        return "common_sense"
    return "unknown"

def classify_by_llm(question):
    # Only use if keywords fail
    system = "Classify as one word: math, coding, common_sense, future_prediction, planning."
    reply = call_llm(question[:1000], system=system, temperature=0.0)
    reply = reply.lower()
    for domain in SOLVERS:
        if domain in reply:
            return domain
    return "common_sense"

def classify_domain(question):
    d = classify_by_keywords(question)
    if d != "unknown":
        return d
    return classify_by_llm(question)

def route_and_solve(question):
    # Main logic: find domain then call the right solver
    domain = classify_domain(question)
    solver = SOLVERS.get(domain)
    
    if solver is None:
        return _fallback_direct(question)
    
    try:
        return solver(question)
    except (ImportError, Exception) as e:
        # Fallback if teammate's code isn't ready or crashes
        print(f"Routing to fallback for {domain} due to: {e}")
        return _fallback_direct(question)

def _fallback_direct(question):
    # Simple 1-call fallback
    system = "Solve the question and give only the final answer."
    return call_llm(question, system=system).strip()