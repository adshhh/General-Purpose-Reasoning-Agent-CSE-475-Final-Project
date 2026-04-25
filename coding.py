from __future__ import annotations 
from utils import call_llm
import re

CODING_SYSTEM_PROMPT = """
You are an expert software developer capable of coding in any programming language and have years of experience.
given a coding problem, read the problem carefully and strictly adhere to any constraints mentioned in the problem.
Your output should be in the format: 
'Code:
<code here>'
where <code here> is the code you write as the answer to the problem. Do not include any explanations or comments in your output, just the code.
"""

SELF_CHECK_SYSTEM_PROMPT = """
You are an expert software developer capable of coding in any programming language and have years of experience.
Given a coding problem and its answer as only the code (no explanation), read the problem carefully.
Evaluate whether the code correctly solves the given problem and adheres to the constraints mentioned in the problem. 
If the code is the correct answer for the given problem and you find no issues, your output should be: 
'Verdict: No issues found.
Original code:
<code here>' where <code here> is the original code provided.
if the code is not the correct answer for the given problem or you find any issues, your output should be:
'Verdict: Issues found.
Code after corrections:
<corrected code here>' where <corrected code here> is the code you write as the answer to the problem.
Do not include any explanations or comments in your output, just the code.
"""

CODE_RE = re.compile(r"^Code:\s*\n?(.*)", re.IGNORECASE | re.DOTALL)

def extract_code(output: str) -> str:
    m = CODE_RE.search(output.strip())
    code_output = m.group(1).strip() if m else output.strip()
    
    return code_output

def generate_code(problem: str) -> str:
    result = call_llm(problem, system=CODING_SYSTEM_PROMPT, temperature=0.0, max_tokens=1000)
    
    return extract_code(result)

VER_RE = re.compile(r"^Verdict:\s*(No issues found\.\nOriginal code:(.*)|Issues found\.\nCode after corrections:)(.*)", re.IGNORECASE | re.DOTALL | re.MULTILINE)

def self_check(problem: str, code: str) -> str:
    prompt_self_check = (
        f"PROBLEM:\n{problem}\n\n"
        f"CODE:\n{code}\n\n"
    )

    sf_result = call_llm(prompt_self_check, system=SELF_CHECK_SYSTEM_PROMPT, temperature=0.0, max_tokens=1200)
    m = VER_RE.search(sf_result.strip())
    
    if not m: verdict_output = sf_result.strip()
    elif "no issues found." in m.group(1).lower(): verdict_output = m.group(2).strip()
    else: verdict_output = m.group(3).strip()

    return verdict_output

def solve(question: str) -> str:
    code = generate_code(question)
    
    first_check = self_check(question, code)
    if first_check != code:
        return self_check(question, first_check)
    #return code
    return first_check
