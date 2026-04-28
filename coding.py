#Aditya - coding domain solver (plan-then-code)
from __future__ import annotations
from utils import call_llm
import re

CODING_SYSTEM_PROMPT = """You are an expert software developer with years of experience coding in Python.
Given a coding problem and a plan, provide a code solution that adheres to the plan and all constraints in the problem.

Output format:
Code:
    <function body here>

Rules:
- Output ONLY the lines inside the function body.
- Do NOT include import statements.
- Do NOT include the def line (e.g. "def task_func(...):").
- Do NOT use markdown code fences (no triple backticks).
- Do NOT include explanations, docstrings, or comments.
- Do NOT re-declare constants or variables already defined before the function signature in the starter code.
"""

REASONING_SYSTEM_PROMPT = """You are an expert software developer with years of experience. Given a coding problem, you have to understand the question fully and adhere to the rules listed in the question.
Then, devise a plan to solve the problem. The plan should be divided into steps, and each step should be concise and to the point.
If the problem provides a starter code block, the implementation must follow that exact structure (same imports, same function name and signature).
Note every library imported in the starter, each one must be used in the implementation.
"""

SELF_CHECK_PROMPT = """You are a Python code reviewer.
You will be given a coding problem and a function body implementation.

Your job:
- Check if the implementation is syntactically valid Python
- Fix any syntax errors, broken indentation, or incomplete code blocks
- Ensure the code correctly solves the problem constraints
- Output ONLY the corrected function body — no def line, no imports, no explanations
- If the code is already correct, output it unchanged
"""

CODE_RE = re.compile(r"^Code:\s*\n?(.*)", re.IGNORECASE | re.DOTALL)


def starter_constants(question: str) -> set:
    m = re.search(r'```[^\n]*\n(.*?)```', question, re.DOTALL)
    if not m:
        return set()
    
    return set(re.findall(r'^([A-Z_][A-Z0-9_]+)\s*=', m.group(1), re.MULTILINE))


def extract_code(output: str) -> str:
    m = CODE_RE.search(output.strip())
    code_output = m.group(1).strip() if m else output.strip()
    code_output = re.sub(r"```\w*\n?", "", code_output)
    code_output = re.sub(r"\n?```", "", code_output)
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


def strip_header_constants(code: str, question: str) -> str:
    constants = starter_constants(question)
    if not constants:
        return code
    filtered = []
    for line in code.splitlines():
        m = re.match(r'\s*([A-Z_][A-Z0-9_]+)\s*=', line)
        if m and m.group(1) in constants:
            continue
        filtered.append(line)

    return "\n".join(filtered).strip()


def reason_for_code(question: str) -> str:
    return call_llm(question, system=REASONING_SYSTEM_PROMPT, temperature=0.0, max_tokens=500).strip()


def generate_code(problem: str, reasoning: str) -> str:
    prompt_with_plan = (
        f"PROBLEM:\n{problem}\n\n"
        f"PLAN:\n{reasoning}\n\n"
    )
    result = call_llm(prompt_with_plan, system=CODING_SYSTEM_PROMPT, temperature=0.0, max_tokens=1000)

    return extract_code(result)


def self_check(question: str, code: str) -> str:
    if not code.strip():
        return code
    prompt = (
        f"PROBLEM:\n{question}\n\n"
        f"IMPLEMENTATION:\n{code}\n\n"
        "Review and fix the implementation if needed. Output only the corrected function body."
    )
    result = call_llm(prompt, system=SELF_CHECK_PROMPT, temperature=0.0, max_tokens=1000)
    fixed = extract_code(result)

    return fixed if fixed.strip() else code


def solve(question: str) -> str:
    reasoning = reason_for_code(question)
    code = generate_code(question, reasoning)
    code = strip_header_constants(code, question)

    return self_check(question, code)
