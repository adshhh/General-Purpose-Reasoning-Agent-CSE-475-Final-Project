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

    return extract_answer(result)

def self_consistency(problem: str, n: int = 3) -> str:
    answers = []
    for i in range(n): 
        temp = 0.0 if i == 0 else random.uniform(0.0, 0.5)
        answers.append(cot_solve(problem, temperature=temp))
    
    return majority_vote(answers)
    
def solve(question: str) -> str:
    return self_consistency(question, n=3)

    

    