import re
from utils import call_llm


# Strict: the whole line must be a valid PDDL action (lowercase).
_ACTION_RE = re.compile(r"^\([a-z][a-z0-9_\-]*(?:\s+[a-z0-9_\-]+)*\)$")
# Fallback: find any parenthesized token anywhere in a line.
_ACTION_ANYWHERE_RE = re.compile(r"\([a-z][a-z0-9_\-]*(?:\s+[a-z0-9_\-]+)*\)")


def _extract_actions(text):
    if not text:
        return []
    actions = []
    for raw in text.split("\n"):
        line = raw.strip().lower()
        if not line:
            continue
        # Strip leading numbering/bullets.
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[\-\*]\s*", "", line)
        line = line.strip("` ")
        if _ACTION_RE.match(line):
            actions.append(line)

    # Fallback: if strict per-line extraction found nothing, search anywhere
    # in the full text. Handles prose like "Step 1: (pick-up a)".
    if not actions:
        actions = _ACTION_ANYWHERE_RE.findall(text.lower())
    return actions


_PDDL_FORMAT_RULES = """You are a PDDL formatter. Output ONLY valid PDDL actions.

RULES:
1. One action per line.
2. Format: (action-name arg1 arg2 arg3)
3. All lowercase. Use hyphens for multi-word actions.
4. Arguments are object names (lowercase).
5. No explanations, no commentary, no blank lines, no markdown fences.

VALID EXAMPLES:
(feast b d)
(succumb b)
(lift hoist2 crate2 crate1 depot2)
(load hoist2 crate2 truck2 depot2)
(unstack yellow red)
(put-down yellow)
(pick-up yellow)

Start your output immediately with the first action."""


# Technique 9: Plan-and-Solve (2 calls)
def plan_and_solve(question):
    plan_sys = (
        "You are a planning expert. Read the problem carefully and write a "
        "clear, step-by-step plan that respects all the action preconditions "
        "stated in the problem. Be explicit about which arguments each action takes."
    )
    plan = call_llm(question, system=plan_sys, max_tokens=800)

    prompt = f"Problem:\n{question}\n\nPlan:\n{plan}\n\nNow output the PDDL actions."
    pddl = call_llm(prompt, system=_PDDL_FORMAT_RULES, max_tokens=800)
    return pddl


# Technique 10: Least-to-Most Decomposition (2 calls)
def least_to_most(question):
    decomp_sys = (
        "Break this planning problem into the simplest possible sub-problems. "
        "Number them 1, 2, 3, ... in order from easiest to hardest. For each "
        "sub-problem, state which goal predicate it achieves."
    )
    subproblems = call_llm(question, system=decomp_sys, max_tokens=800)

    prompt = (f"Problem:\n{question}\n\nSub-problems:\n{subproblems}\n\n"
              f"Solve each sub-problem in order and output the combined PDDL action sequence.")
    pddl = call_llm(prompt, system=_PDDL_FORMAT_RULES, max_tokens=800)
    return pddl


def _split_approaches(text):
    if not text:
        return ["", "", ""]
    parts = re.split(r"(?i)\bapproach\s*\d+\s*[:.\-]?", text)
    parts = [p.strip() for p in parts if p.strip()]
    while len(parts) < 3:
        parts.append("")
    return parts[:3]


# Technique 11: Tree of Thoughts (6 calls)
def tree_of_thoughts(question):
    gen_sys = (
        "Generate exactly 3 distinct, detailed approaches to solve this "
        "planning problem. Label them clearly as 'Approach 1:', 'Approach 2:', "
        "and 'Approach 3:'. Each approach should be a different strategy."
    )
    approaches = call_llm(question, system=gen_sys, max_tokens=900)

    a1, a2, a3 = _split_approaches(approaches)

    eval_sys = (
        "Briefly analyze this approach. List its strengths, weaknesses, and "
        "whether it will produce a valid plan that respects the problem's "
        "preconditions. Be concise (3-5 sentences)."
    )
    e1 = call_llm(f"Problem:\n{question}\n\nApproach:\n{a1}", system=eval_sys, max_tokens=300)
    e2 = call_llm(f"Problem:\n{question}\n\nApproach:\n{a2}", system=eval_sys, max_tokens=300)
    e3 = call_llm(f"Problem:\n{question}\n\nApproach:\n{a3}", system=eval_sys, max_tokens=300)

    select_sys = (
        "Read the three evaluations and decide which approach is best. "
        "Reply with ONLY a single digit: 1, 2, or 3."
    )
    summary = f"Evaluation 1:\n{e1}\n\nEvaluation 2:\n{e2}\n\nEvaluation 3:\n{e3}"
    pick = call_llm(f"Problem:\n{question}\n\n{summary}", system=select_sys, max_tokens=10).strip()

    chosen = a1
    if pick.startswith("2"):
        chosen = a2
    elif pick.startswith("3"):
        chosen = a3

    prompt = (f"Problem:\n{question}\n\nSelected approach:\n{chosen}\n\n"
              f"Now output the PDDL actions for this approach.")
    pddl = call_llm(prompt, system=_PDDL_FORMAT_RULES, max_tokens=800)
    return pddl


def solve(question, technique="plan_and_solve"):
    if technique == "least_to_most":
        result = least_to_most(question)
    elif technique == "tot":
        result = tree_of_thoughts(question)
    else:
        result = plan_and_solve(question)

    actions = _extract_actions(result)
    if actions:
        return "\n".join(actions)
    return (result or "").strip()
