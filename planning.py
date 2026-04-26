# Ismail - planning domain solver
"""
Planning domain solver for CSE 476 Final Project.

Handles PDDL-style planning questions (Blocksworld, Logistics, Mystery, etc.)
where the expected output is a sequence of action lines like:
    (feast b d)
    (succumb b)
    (attack a)

Implements three inference-time techniques. Router calls `solve()`.

  1. plan_and_solve()    -- 2 LLM calls  (cheap, usually good enough)
  2. least_to_most()     -- 2 LLM calls  (decompose into sub-goals, solve)
  3. tree_of_thoughts()  -- 6 LLM calls  (diversity + self-scoring, best quality)

Total per-question budget stays <= 6 calls, well under the project's 20-call
limit. Default strategy is least_to_most (best quality/cost ratio on dev).
"""

from __future__ import annotations

import re
from typing import List

from utils import call_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Matches parenthesized actions in any case, e.g. "(feast b d)", "(Attack A)"
# "(drive truck2 depot2 depot0)".
_ACTION_RE = re.compile(r"\(\s*[a-zA-Z][a-zA-Z0-9_\- ]*\)")


def _extract_plan(text: str) -> str:
    """
    Extract action lines from raw LLM output.
    Strategy:
      1. Pull all parenthesized tokens -- lowercased so output is consistent.
      2. If nothing found, return the raw text stripped of markdown fences
         (the model may have answered without parens, e.g. 'attack a').
    """
    if not text:
        return ""
    actions: List[str] = []
    for match in _ACTION_RE.findall(text):
        action = re.sub(r"\s+", " ", match.strip()).lower()
        actions.append(action)
    if actions:
        return "\n".join(actions)
    # Fallback: strip markdown fences and return raw text
    cleaned = re.sub(r"```[a-z]*", "", text).strip()
    return cleaned


def _score_plan(problem: str, plan: str) -> float:
    """Ask the LLM to rate a candidate plan 0-10. One cheap call."""
    system = (
        "You are a strict planning validator. Given a planning problem and a "
        "candidate plan, rate how likely the plan is to reach the goal while "
        "respecting every precondition. Reply with ONLY a single integer 0-10."
    )
    prompt = (
        f"PROBLEM:\n{problem[:3000]}\n\n"
        f"CANDIDATE PLAN:\n{plan}\n\n"
        "Score (0-10):"
    )
    reply = call_llm(prompt, system=system, temperature=0.0, max_tokens=10)
    m = re.search(r"\d+", reply or "")
    if not m:
        return 0.0
    return min(10.0, max(0.0, float(m.group(0))))


# ---------------------------------------------------------------------------
# Technique 1: Plan-and-Solve
# ---------------------------------------------------------------------------

def plan_and_solve(problem: str) -> str:
    """First produce a natural-language outline, then emit action syntax."""
    outline_system = (
        "You are a careful planner. Read the planning problem. First restate "
        "the initial state and goal in one sentence each. Then list the "
        "high-level steps needed to reach the goal. Do not emit action syntax "
        "yet. Keep it under 12 steps."
    )
    outline = call_llm(problem, system=outline_system, temperature=0.0, max_tokens=600)

    emit_system = (
        "Convert a plan outline into a sequence of actions using the EXACT "
        "syntax shown in the problem statement (one action per line, e.g. "
        "'(attack a)' or '(drive truck1 depot0 depot1)'). Output ONLY the "
        "action lines, no commentary, no markdown fences."
    )
    emit_prompt = (
        f"PROBLEM:\n{problem}\n\n"
        f"PLAN OUTLINE:\n{outline}\n\n"
        "Now emit the final action sequence:"
    )
    actions = call_llm(emit_prompt, system=emit_system, temperature=0.0, max_tokens=800)
    return _extract_plan(actions)


# ---------------------------------------------------------------------------
# Technique 2: Least-to-Most
# ---------------------------------------------------------------------------

def least_to_most(problem: str) -> str:
    """Decompose goal into ordered sub-goals, then solve them in one shot."""
    decompose_system = (
        "Given a planning problem, break the final goal into an ordered list "
        "of smaller sub-goals. Each sub-goal should name a concrete predicate "
        "from the goal description. Output one sub-goal per line, numbered."
    )
    subgoals = call_llm(
        problem, system=decompose_system, temperature=0.0, max_tokens=400
    )

    solve_system = (
        "You are a planner. Achieve the listed sub-goals in order, using only "
        "the actions defined in the problem. Respect every precondition. "
        "Output ONLY action lines in the EXACT syntax from the problem "
        "(e.g. '(feast b d)'), one per line, no commentary."
    )
    solve_prompt = (
        f"PROBLEM:\n{problem}\n\n"
        f"ORDERED SUB-GOALS:\n{subgoals}\n\n"
        "Full action sequence to achieve all sub-goals in order:"
    )
    actions = call_llm(solve_prompt, system=solve_system, temperature=0.0, max_tokens=800)
    return _extract_plan(actions)


# ---------------------------------------------------------------------------
# Technique 3: Tree of Thoughts
# ---------------------------------------------------------------------------

def tree_of_thoughts(problem: str, n_candidates: int = 3) -> str:
    """Sample diverse plans, score each, return the best. 2 * n_candidates calls."""
    gen_system = (
        "You are a creative planner. Produce a full plan for the given "
        "problem. Output ONLY the action lines in the EXACT syntax used in "
        "the problem statement, one per line, no commentary."
    )

    candidates: List[str] = []
    for i in range(n_candidates):
        temp = 0.3 + 0.3 * i  # 0.3, 0.6, 0.9 -- increasing diversity
        raw = call_llm(problem, system=gen_system, temperature=temp, max_tokens=800)
        plan = _extract_plan(raw)
        if plan:
            candidates.append(plan)

    if not candidates:
        return ""

    best_plan = candidates[0]
    best_score = -1.0
    for plan in candidates:
        score = _score_plan(problem, plan)
        if score > best_score:
            best_score = score
            best_plan = plan
    return best_plan


# ---------------------------------------------------------------------------
# Main entry point (called by router.py)
# ---------------------------------------------------------------------------

def solve(question: str, strategy: str = "least_to_most", debug: bool = False) -> str:
    """
    Route to one of the three techniques.
    strategy: "plan_and_solve" | "least_to_most" (default) | "tree_of_thoughts"
    debug: print raw API response to diagnose empty results
    """
    if debug:
        raw = call_llm(
            question,
            system="You are a planner. Output ONLY action lines in parenthesis syntax e.g. (attack a), one per line.",
            temperature=0.0,
            max_tokens=800,
        )
        print("[DEBUG] Raw API response:")
        print(raw)
        print("[DEBUG] Extracted:")
        print(_extract_plan(raw))
        return _extract_plan(raw)
    if strategy == "plan_and_solve":
        return plan_and_solve(question)
    if strategy == "tree_of_thoughts":
        return tree_of_thoughts(question, n_candidates=3)
    return least_to_most(question)
