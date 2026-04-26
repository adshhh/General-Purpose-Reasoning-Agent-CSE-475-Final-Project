from __future__ import annotations 
from utils import call_llm
import re
from typing import List
from collections import Counter
import random

COT_SYSTEM_PROMPT = """You are an expert at solving math problems. You are capable of solving even difficult college math questions.
When solving for a question, analyze the whole question and think step-by-step. Keep your reasoning concise. 
Only output the key steps that you took. Your output should be at most 500 words.
Your output should have the format:
'Reasoning: 
Step 1: ...
Step 2: ...
...
Final Answer: <value>' where <value> is the final answer to the question, format it as a float rounded to 4 decimal places.
"""

PAL_SYSTEM_PROMPT = """
Q: There were nine footballs in the sports room. Five more footballs were added each day, from Monday to Thursday. How many footballs are now in the sports room?

# solution in Python:
'''There were nine footballs in the sports room. Five more footballs were added each day, from Monday to Thursday. How many footballs are now in the sports room?'''
footballs_initial = 9
footballs_per_day = 5
num_days = 4  # 4 days between Monday and Thursday
footballs_added = footballs_per_day * num_days
footballs_total = footballs_initial + footballs_added
result = footballs_total


Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

# solution in Python:
'''Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?'''
toys_initial = 5
mom_toys = 2
dad_toys = 2
total_received = mom_toys + dad_toys
total_toys = toys_initial + total_received
result = total_toys


Q: Jason had 20 hotwheels cars. He gave Denny some of the cars. Now Jason has 12 hotwheels cars. How many hotwheels cars did Jason give to Denny?

# solution in Python:
'''Jason had 20 hotwheels cars. He gave Denny some of the cars. Now Jason has 12 hotwheels cars. How many hotwheels cars did Jason give to Denny?'''
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

def cot_solve(problem: str, temperature: float = 0.0) -> str:
    result = call_llm(problem, system=COT_SYSTEM_PROMPT, temperature=temperature, max_tokens=2000)

    return result

def pal_from_cot(question: str, cot_reasoning: str, temperature: float = 0.0) -> str:
    prompt = (
        f"Q: {question}\n\n"
        f"Mathematical approach:\n{cot_reasoning}\n\n"
        f"# solution in Python:"
    )
    result = call_llm(prompt, system=PAL_SYSTEM_PROMPT, temperature=temperature, max_tokens=1000)
    result = re.sub(r"```\w*\n?", "", result)
    result = re.sub(r"\n?```", "", result)
    return result.strip()


def self_consistency(problem: str, n: int = 3) -> str:
    answers = []
    for i in range(n): 
        temp = 0.0 if i == 0 else random.uniform(0.0, 0.5)
        result = run_code(pal_solve(problem, temperature=temp))
        if result is not None:
            answers.append(str(result))
    return majority_vote(answers)


def pal_solve(question: str,  temperature: float = 0.0) -> str:
    prompt = f"Q: {question}\n\n# solution in Python:"
    result = call_llm(prompt, system=PAL_SYSTEM_PROMPT, temperature=temperature, max_tokens=1000)
    result = re.sub(r"```\w*\n?", "", result)
    result = re.sub(r"\n?```", "", result)
    return result.strip()


def run_code(code: str):
    namespace = {}
    try:
        exec(code, namespace)
        return namespace.get("result", None)
    except Exception as e:
        print(f"[run_code failed] {e}\nCode: {code[:200]}")
        return None


def solve(question: str) -> str:
    cot_output = cot_solve(question)                    # call 1: full reasoning
    pal_code   = pal_from_cot(question, cot_output)     # call 2: code from reasoning
    result     = run_code(pal_code)                     # local exec, 0 LLM calls

    if result is not None:
        return str(result)

    return extract_answer(cot_output)    


    

    