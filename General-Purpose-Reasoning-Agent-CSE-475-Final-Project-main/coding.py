#Aditya - coding domain solver (plan-then-code)
from __future__ import annotations
from utils import call_llm
import re

CODING_SYSTEM_PROMPT = """You are an expert software developer.
Given a coding problem and a plan, implement ONLY the body of the task_func function.

Output format:
Code:
    <function body here, each line indented with 4 spaces>

Rules:
- Output ONLY the lines inside the function body, each indented with 4 spaces.
- Do NOT include any import statements.
- Do NOT include the def line (e.g. "def task_func(...):").
- Do NOT use markdown code fences (no triple backticks).
- Do NOT include explanations, docstrings, or comments.
- If the problem includes a starter code block with imports, use ALL those imported libraries — they tell you which patterns to use (e.g. if json is imported, use json.loads(); if collections is imported, use collections.OrderedDict()).
- NEVER use f-strings (f"..."). Always use string concatenation with the + operator instead.
- NEVER create intermediate URL variables. Pass URLs directly to function calls.
- Always put constants AFTER setup code like random.seed(), not before.
- Use try/except blocks when working with database connections or file I/O.
- When the problem provides variable names in its description or examples, use those EXACT names.
- Prefer `variable = value` on separate lines over inline expressions.

Example:
Problem: Write task_func(items) that returns items sorted by their length.
Code:
    return sorted(items, key=len)
"""

REASONING_SYSTEM_PROMPT = """You are an expert software developer with years of experience. Given a coding problem, you have to understand the question fully and adhere to the rules listed in the question.
Then, devise a plan to solve the problem. The plan should be divided into steps, and each step should be concise and to the point.
If the problem provides a starter code block, the implementation must follow that exact structure (same imports, same function name and signature).
Note every library imported in the starter — each one must be used in the implementation.
"""

CODE_RE = re.compile(r"^Code:\s*\n?(.*)", re.IGNORECASE | re.DOTALL | re.MULTILINE)

def extract_code(output: str) -> str:
    m = CODE_RE.search(output.strip())
    code_output = m.group(1).strip() if m else output.strip()
    code_output = re.sub(r"```\w*\n?", "", code_output)
    code_output = re.sub(r"\n?```", "", code_output)
    # Strip any accidentally included import lines and function def lines
    lines = code_output.split("\n")
    body_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if re.match(r"^def \w+\s*\(", stripped):
            continue
        body_lines.append(line)
    return "\n".join(body_lines).strip()

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
