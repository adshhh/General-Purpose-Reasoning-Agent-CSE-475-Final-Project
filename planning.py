import re
from utils import call_llm

def _extract_actions(text):
    """Extract PDDL actions in (action args) format from text."""
    actions = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        # Match (action arg1 arg2 ...) format
        if line.startswith('(') and line.endswith(')'):
            actions.append(line)
    return actions

def plan_and_solve(question):
    """
    Technique 8: Plan-and-Solve (2 calls)
    Call 1: Generate a high-level plan
    Call 2: Convert plan to PDDL actions in (action args) format
    """
    plan_sys = "You are a planning expert. Create a clear, step-by-step plan to solve this planning problem."
    plan = call_llm(question, system=plan_sys)
    
    pddl_sys = """You are a PDDL formatter. Convert the plan into PDDL action format.

RULES:
1. Output ONLY valid PDDL actions, one per line
2. Format: (action-name arg1 arg2 arg3)
3. All lowercase, use hyphens for multi-word actions
4. Arguments are the object names (also lowercase)
5. No explanations, no text, no blank lines between actions

VALID EXAMPLES:
(feast b d)
(succumb b)
(lift hoist2 crate2 crate1 depot2)
(load hoist2 crate2 truck2 depot2)
(unstack yellow red)
(put-down yellow)
(pick-up yellow)

Start output immediately with the first action."""
    
    prompt = f"Problem: {question}\n\nPlan:\n{plan}"
    pddl_actions = call_llm(prompt, system=pddl_sys)
    
    return pddl_actions

def least_to_most(question):
    """
    Technique 9: Least-to-Most Decomposition (2 calls)
    Call 1: Break the problem into simpler sub-problems
    Call 2: Solve by addressing each sub-problem sequentially
    """
    decomp_sys = "Break this planning problem into the simplest possible sub-problems. Number them in order of difficulty (easy to hard)."
    subproblems = call_llm(question, system=decomp_sys)
    
    solve_sys = """You are a PDDL formatter. Solve each sub-problem and output the PDDL actions.

RULES:
1. Output ONLY valid PDDL actions, one per line
2. Format: (action-name arg1 arg2 arg3)
3. All lowercase, use hyphens for multi-word actions
4. Arguments are object names (also lowercase)
5. No explanations, no text, no blank lines between actions

VALID EXAMPLES:
(feast b d)
(succumb b)
(lift hoist2 crate2 crate1 depot2)
(load hoist2 crate2 truck2 depot2)
(unstack yellow red)
(put-down yellow)

Start immediately with the first action."""
    
    prompt = f"Problem: {question}\n\nSub-problems:\n{subproblems}"
    pddl_actions = call_llm(prompt, system=solve_sys)
    
    return pddl_actions

def tree_of_thoughts(question):
    """
    Technique 10: Tree of Thoughts (6 calls)
    Call 1: Generate 3 distinct solution approaches
    Calls 2-4: Deeply evaluate each approach
    Call 5: Select the best approach
    Call 6: Generate final PDDL actions using the best approach
    """
    gen_sys = "Generate exactly 3 distinct, detailed approaches to solve this planning problem. Label as Approach 1, 2, 3."
    approaches = call_llm(question, system=gen_sys)
    
    eval_sys = "Analyze this approach in detail. What are the strengths, weaknesses, and feasibility? Will it solve the problem?"
    
    parts = approaches.split('Approach 2')
    approach_1 = parts[0]
    approach_2 = parts[1].split('Approach 3')[0] if len(parts) > 1 else ""
    approach_3 = parts[1].split('Approach 3')[1] if 'Approach 3' in approaches else ""
    
    eval_1 = call_llm(f"Problem: {question}\n\nApproach:\n{approach_1}", system=eval_sys)
    eval_2 = call_llm(f"Problem: {question}\n\nApproach:\n{approach_2}", system=eval_sys)
    eval_3 = call_llm(f"Problem: {question}\n\nApproach:\n{approach_3}", system=eval_sys)
    
    select_sys = "Based on the evaluations above, which approach is best? Reply with ONLY the number: 1, 2, or 3."
    evals = f"Evaluation 1:\n{eval_1}\n\nEvaluation 2:\n{eval_2}\n\nEvaluation 3:\n{eval_3}"
    best = call_llm(f"Problem: {question}\n\n{evals}", system=select_sys).strip()
    
    solve_sys = """You are a PDDL formatter. Using the selected approach, generate the PDDL action sequence.

RULES:
1. Output ONLY valid PDDL actions, one per line
2. Format: (action-name arg1 arg2 arg3)
3. All lowercase, use hyphens for multi-word actions
4. Arguments are object names (also lowercase)
5. No explanations, no text, no blank lines between actions

VALID EXAMPLES:
(feast b d)
(succumb b)
(lift hoist2 crate2 crate1 depot2)
(load hoist2 crate2 truck2 depot2)
(unstack yellow red)
(put-down yellow)

Start immediately with the first action."""
    
    prompt = f"Problem: {question}\n\nApproaches:\n{approaches}\n\nBest Approach: {best}"
    pddl_actions = call_llm(prompt, system=solve_sys)
    
    return pddl_actions

def solve(question, technique="plan_and_solve"):
    """
    Main entry point for the planning solver.
    Routes to the selected technique and returns cleaned PDDL actions.
    """
    if technique == "least_to_most":
        result = least_to_most(question)
    elif technique == "tot":
        result = tree_of_thoughts(question)
    else:
        result = plan_and_solve(question)
    
    actions = _extract_actions(result)
    
    return "\n".join(actions) if actions else result