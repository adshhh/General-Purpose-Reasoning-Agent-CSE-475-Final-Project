from __future__ import annotations 
from utils import call_llm
import re
from typing import List
from collections import Counter
import random

COT_PLAN_SYSTEM_PROMPT = """You are an expert at solving math problems, including difficult college-level questions.

Your job is not to write Python code and not to give the final numeric answer.
Your job is to understand the problem and produce a clear mathematical plan in normal text that can later be translated into Python.

When given a problem:
1. Read the problem carefully.
2. Identify the important quantities, variables, and relationships.
3. Reason through the math step by step.
4. Produce a concise text-based plan describing how to solve it.
5. Do not output code.
6. Do not output the final answer.

Your output must follow exactly this format:

Reasoning:
Step 1: ...
Step 2: ...
Step 3: ...

Mathematical Plan:
1. Define the known values and unknowns.
2. State the formulas, equations, or relationships to use.
3. Describe the computation steps in plain English.
4. State what quantity should be computed at the end.

Keep the plan concise, clear, and directly usable for generating Python code.
The answer is always a non-negative integer between 0 and 999.
"""

COT_ANSWER_PROMPT = """You are an expert at solving difficult math problems.
The answer is always a non-negative integer between 0 and 999.
When solving a problem, think step-by-step and keep your reasoning concise.
Your output must follow exactly this format:
Reasoning:
Step 1: ...
Step 2: ...
...
Final Answer: <integer>
"""

PAL_SYSTEM_PROMPT = """You are an expert Python programmer solving math word problems.

You will be given:
- a math question
- a mathematical approach written in normal text

Your job is to translate that mathematical approach into correct Python code.

Rules:
- Output only valid Python code.
- Do not explain your code.
- Do not repeat the question.
- Do not include markdown fences.
- Use the mathematical approach when provided.
- You may use standard Python libraries and sympy when symbolic solving is needed.
- Store the final answer in a variable named result.
- Always assign result as a plain Python number. If using sympy, wrap the final value with float() or int().
- The final result must always be a plain Python integer (not float, not sympy expression). Use int() if needed.

Q: There were nine footballs in the sports room. Five more footballs were added each day, from Monday to Thursday. How many footballs are now in the sports room?

Mathematical approach:
1. Start with the initial number of footballs.
2. Compute how many footballs were added over 4 days.
3. Add the total added footballs to the initial amount.
4. Return the total number of footballs.

# solution in Python:
footballs_initial = 9
footballs_per_day = 5
num_days = 4
footballs_added = footballs_per_day * num_days
footballs_total = footballs_initial + footballs_added
result = footballs_total


Q: Solve for x: x^2 - 5x + 6 = 0. What is the smaller root?

Mathematical approach:
1. Define a symbolic variable x.
2. Solve the quadratic equation x^2 - 5x + 6 = 0.
3. Collect the roots.
4. Return the smaller root.

# solution in Python:
from sympy import symbols, solve
x = symbols('x')
roots = solve(x**2 - 5*x + 6, x)
result = min(roots)


Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

Mathematical approach:
1. Start with Shawn's initial number of toys.
2. Add the toys received from his mom and dad.
3. Add the total received toys to the initial amount.
4. Return the final number of toys.

# solution in Python:
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
result = total_toys


Q: Jason had 20 hotwheels cars. He gave Denny some of the cars. Now Jason has 12 hotwheels cars. How many hotwheels cars did Jason give to Denny?

Mathematical approach:
1. Start with Jason's initial number of cars.
2. Note the number of cars Jason has after giving some away.
3. Subtract the final amount from the initial amount.
4. Return the number of cars Jason gave to Denny.

# solution in Python:
jason_hotwheels_cars_initial = 20
jason_hotwheels_cars_after = 12
denny_hotwheels_cars = jason_hotwheels_cars_initial - jason_hotwheels_cars_after
result = denny_hotwheels_cars
"""

ANS_RE = re.compile(r"^Final Answer:\s*(.*)$", re.IGNORECASE | re.MULTILINE)

# Cleanly get final answer value
def extract_answer(output: str) -> str:
    
    m = ANS_RE.search(output.strip())
    final_answer = m.group(1).strip() if m else output.strip()
    
    return final_answer

# Normalize the value, Helper function for majority_vote().
def normalize_answer(ans: str) -> str:
    
    norm_ans = ans.strip().replace(",", "")
    try:
        f = float(norm_ans)
        return str(int(f)) if f == int(f) else str(f)
    
    except (ValueError, OverflowError):
        return norm_ans

# Calculate votes for each final answers and return the winner.
def majority_vote(answers: List[str]) -> str:
    
    if not answers:
        return ""
    
    n_answers = []
    for i in range(len(answers)): n_answers.append(normalize_answer(answers[i]))
    votes = Counter(n_answers)
    winner = max(votes, key=votes.get)

    return winner

def cot_pal_solve(problem: str, temperature: float = 0.0) -> str:
    result = call_llm(problem, system=COT_PLAN_SYSTEM_PROMPT, temperature=temperature, max_tokens=2000)

    return result

def cot_answer_solve(problem: str, temperature: float = 0.0) -> str:
    result = call_llm(problem, system=COT_ANSWER_PROMPT, temperature=temperature, max_tokens=2000)
    
    return extract_answer(result)

def pal_from_cot(question: str, cot_reasoning: str, temperature: float = 0.0) -> str:
    prompt = (
        f"Q: {question}\n\n"
        f"Mathematical approach:\n{cot_reasoning}\n\n"
        f"# solution in Python:"
    )
    result = call_llm(prompt, system=PAL_SYSTEM_PROMPT, temperature=temperature, max_tokens=1000)
    result = re.sub(r"```\w*\n?", "", result)
    result = re.sub(r"\n?```", "", result)
    result = re.sub(r'(\d+)!', r'math.factorial(\1)', result)

    return result.strip()

def run_code(code: str):
    namespace = {"__builtins__": __builtins__}
    exec("import math", namespace)
    try:
        exec(code, namespace)
        return namespace.get("result", None)
    except Exception as e:
        print(f"[run_code failed] {e}\nCode: {code[:200]}")
        return None

def solve(question: str) -> str:
    plan = cot_pal_solve(question)
    pal_code = pal_from_cot(question, plan)
    result = run_code(pal_code)
    if result is not None:
        try:
            return str(int(result))
        except (TypeError, ValueError):
            return str(result)

    answers = []
    for i in range(3):
        temp = 0.0 if i == 0 else random.uniform(0.3, 0.7)
        answers.append(cot_answer_solve(question, temperature=temp))
    
    return majority_vote(answers)
