# Owner: Person A (adshhh / Aditya) - coding domain solver (plan-then-code)
from __future__ import annotations
from utils import call_llm
import re

CODING_SYSTEM_PROMPT = """
You are an expert software developer capable of coding in any programming language and have years of experience.
Given a coding problem, read the problem carefully and strictly adhere to any constraints mentioned in the problem.
Your output should be in the format: 
'Code:
<code here>'
where <code here> is the code you write as the answer to the problem. Do not include any explanations or comments in your output, just the code.
Refer to the plan given with the question.
"""

REASONING_SYSTEM_PROMPT = """
You are an expert software developer with years of experience. Given a coding problem, you have to understand the question fully and adhere to the rules listed in the question.
Then, devise a plan to solve the problem. The plan should be divided into steps, and each step should be concise and to the point.
"""

CODE_RE = re.compile(r"^Code:\s*\n?(.*)", re.IGNORECASE | re.DOTALL)

def extract_code(output: str) -> str:
    m = CODE_RE.search(output.strip())
    code_output = m.group(1).strip() if m else output.strip()
    
    return code_output

def reason_for_code(question: str) -> str:
    result = call_llm(question, system=REASONING_SYSTEM_PROMPT, temperature=0.0, max_tokens=500)
    
    return result.strip()

def generate_code(problem: str, reasoning: str) -> str:
    prompt_with_plan = (
        f"PROBLEM:\n{problem}\n\n"
        f"PLAN:\n{reasoning}\n\n"
    )
    result = call_llm(prompt_with_plan, system=CODING_SYSTEM_PROMPT, temperature=0.0, max_tokens=1000)
    
    return extract_code(result)

def solve(question: str) -> str:
    reasoning = reason_for_code(question)
    
    return generate_code(question, reasoning)

    #first_check = self_check(question, code)
    #if first_check != code:
   #    return self_check(question, first_check)

    #return first_check
