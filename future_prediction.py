# Angel - future-prediction domain solver
"""Future-prediction solver: plan-and-solve + self-refine + 4-draft ensemble vote."""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import List, Optional

from utils import call_llm, get_per_question_calls, PER_QUESTION_CAP
from parallel import call_llm_concurrent


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

PLAN_TEMPERATURE = 0.0
SOLVE_TEMPERATURE = 0.2
REFINE_TEMPERATURE = 0.0
DRAFT_TEMPERATURE = 0.5    # quick drafts get more variance for ensemble diversity

PLAN_MAX_TOKENS = 350
SOLVE_MAX_TOKENS = 600
CRITIQUE_MAX_TOKENS = 350
REFINE_MAX_TOKENS = 500
DRAFT_MAX_TOKENS = 350

NUM_QUICK_DRAFTS = 4

# Self-imposed cap. Leaves 1 call of headroom under the team's 20 in utils.py.
PERSON_B_CALL_CAP = 19


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are a forecasting analyst. The user will give you a future-prediction "
    "question. Do NOT answer it yet. Instead produce a short numbered plan "
    "(3-6 items) describing the facts, base rates, and sub-questions you would "
    "need to weigh in order to answer. Keep each line under 20 words."
)

_SOLVE_SYSTEM = (
    "You are a forecasting analyst. You will be given a question and a plan. "
    "Work through the plan step by step, then commit to a single, concrete "
    "prediction. Do NOT refuse to answer and do not hedge with 'it depends'. "
    "End your response with EXACTLY one line in this format:\n"
    "Final answer: \\boxed{<short answer>}"
)

_CRITIQUE_SYSTEM = (
    "You are a strict reviewer of forecasting answers. Given a question and a "
    "draft answer, list the most important issues with the draft: factual "
    "mistakes, missing considerations, logical jumps, or wrong final format. "
    "If the draft is already strong, say 'No major issues.' Keep the response "
    "under 8 short bullet points. Do not write a new answer."
)

_REFINE_SYSTEM = (
    "You are a forecasting analyst producing a final answer. You will be "
    "given the question, a draft answer, and a critique of that draft. "
    "Use the critique to fix the draft. If the critique says 'No major "
    "issues', keep the draft's conclusion. Output a brief justification "
    "(<= 4 sentences) followed by EXACTLY one line in this format:\n"
    "Final answer: \\boxed{<short answer>}"
)

# Quick draft -- no plan or critique, just commit to a boxed prediction.
# Higher temperature (0.5) gives the ensemble useful diversity.
_DRAFT_SYSTEM = (
    "You are a forecasting analyst. Read the question, think briefly, then "
    "commit to a single, concrete prediction. Do NOT refuse to answer. "
    "End with EXACTLY one line in this format:\n"
    "Final answer: \\boxed{<short answer>}"
)


# ---------------------------------------------------------------------------
# Boxed-answer extraction (also exposed publicly)
# ---------------------------------------------------------------------------

# Allow nested braces one level deep, e.g. \boxed{\$1{,}200}.
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_TAG_RE = re.compile(
    r"(?:final\s*answer|answer)\s*[:\-]\s*(.+?)(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _clean(s: str) -> str:
    s = s.strip().strip("`*\"'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\s.]+$", "", s)
    return s


def extract_final_answer(text: str) -> str:
    """Return the model's committed answer wrapped in \\boxed{...}."""
    if not text:
        return ""
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return f"\\boxed{{{_clean(boxed_matches[-1])}}}"
    tag_match = _TAG_RE.search(text)
    if tag_match:
        return f"\\boxed{{{_clean(tag_match.group(1))}}}"
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return f"\\boxed{{{_clean(lines[-1])}}}"
    return ""


def _inner(boxed_str: str) -> str:
    """Strip \\boxed{...} wrapper, return the inner text. No-op if not boxed."""
    m = _BOXED_RE.findall(boxed_str)
    return _clean(m[-1]) if m else _clean(boxed_str)


def _budget_left() -> int:
    return max(0, PERSON_B_CALL_CAP - get_per_question_calls())


# ---------------------------------------------------------------------------
# Pipeline stages (each = exactly 1 LLM call when invoked)
# ---------------------------------------------------------------------------

def _plan(question: str) -> str:
    return call_llm(question, system=_PLAN_SYSTEM,
                    temperature=PLAN_TEMPERATURE, max_tokens=PLAN_MAX_TOKENS) or ""


def _solve(question: str, plan: str) -> str:
    prompt = (f"QUESTION:\n{question}\n\n"
              f"PLAN (follow this):\n{plan}\n\n"
              "Now execute the plan and give your draft prediction.")
    return call_llm(prompt, system=_SOLVE_SYSTEM,
                    temperature=SOLVE_TEMPERATURE, max_tokens=SOLVE_MAX_TOKENS) or ""


def _critique(question: str, draft: str) -> str:
    prompt = (f"QUESTION:\n{question}\n\n"
              f"DRAFT ANSWER:\n{draft}\n\n"
              "List the issues with this draft.")
    return call_llm(prompt, system=_CRITIQUE_SYSTEM,
                    temperature=0.0, max_tokens=CRITIQUE_MAX_TOKENS) or ""


def _refine(question: str, draft: str, critique: str) -> str:
    prompt = (f"QUESTION:\n{question}\n\n"
              f"DRAFT ANSWER:\n{draft}\n\n"
              f"CRITIQUE OF DRAFT:\n{critique}\n\n"
              "Now produce the final answer.")
    return call_llm(prompt, system=_REFINE_SYSTEM,
                    temperature=REFINE_TEMPERATURE, max_tokens=REFINE_MAX_TOKENS) or ""


def _quick_draft(question: str) -> str:
    return call_llm(question, system=_DRAFT_SYSTEM,
                    temperature=DRAFT_TEMPERATURE, max_tokens=DRAFT_MAX_TOKENS) or ""


# ---------------------------------------------------------------------------
# Ensemble vote across 5 candidate boxed answers
# ---------------------------------------------------------------------------

# MCQ helper: extract single A-E option letters from a candidate inner.
_MCQ_OPT_RE = re.compile(r"\b([A-E])\b")


def _looks_mcq(inners: List[str]) -> bool:
    """True iff every candidate is a list of 1+ A-E option letters and nothing else."""
    if not inners:
        return False
    for s in inners:
        # Strip punctuation/whitespace; allow only letters A-E and separators.
        stripped = re.sub(r"[\s,;\\/&]+", " ", s).strip().upper()
        # Must contain at least one A-E token.
        opts = _MCQ_OPT_RE.findall(stripped)
        if not opts:
            return False
        # The whole content (after removing "or"/"and") must reduce to A-E tokens.
        cleaned = re.sub(r"\b(OR|AND)\b", " ", stripped)
        cleaned = re.sub(r"[A-E]", " ", cleaned)
        if cleaned.strip():  # something other than A-E letters survived
            return False
    return True


def _ensemble_pick(candidates: List[str]) -> str:
    """
    Combine multiple \\boxed{...} candidate answers into one final \\boxed{...}.

    Order of attempts:
      1. MCQ path: if all candidates are A-E option sets, vote on the most-
         common single letter (handles draft-level "C, D" hedging).
      2. Numeric path: if a clear majority parse as numbers, return median.
      3. Text path: majority vote on the normalised inner text.
    """
    inners = [_inner(c) for c in candidates if c]
    inners = [s for s in inners if s]
    if not inners:
        return ""

    # 1) MCQ single-letter consolidation. When every candidate is an A-E
    #    option set, prefer the most-frequent SINGLE letter across all
    #    mentions. Fixes the idx-204 hedging case where a candidate said
    #    "C, D" while the expected answer was just "D".
    if _looks_mcq(inners):
        all_letters: List[str] = []
        for s in inners:
            stripped = re.sub(r"[\s,;\\/&]+", " ", s).upper()
            all_letters.extend(_MCQ_OPT_RE.findall(stripped))
        if all_letters:
            top_letter, _ = Counter(all_letters).most_common(1)[0]
            return f"\\boxed{{{top_letter}}}"

    # 2) Numeric path.
    parsed_numbers: List[float] = []
    for s in inners:
        nums = _NUMBER_RE.findall(s)
        if nums:
            try:
                parsed_numbers.append(float(nums[0]))
            except ValueError:
                pass
    if len(parsed_numbers) >= max(2, len(inners) // 2 + 1):
        med = statistics.median(parsed_numbers)
        if med.is_integer():
            return f"\\boxed{{{int(med)}}}"
        return f"\\boxed{{{med}}}"

    # 3) Text path: majority vote on normalised inner.
    keys = [re.sub(r"\s+", " ", s.lower().strip()) for s in inners]
    most_common_key, _ = Counter(keys).most_common(1)[0]
    for original, key in zip(inners, keys):
        if key == most_common_key:
            return f"\\boxed{{{original}}}"
    return f"\\boxed{{{inners[0]}}}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def plan_solve_refine_with_ensemble(question: str) -> str:
    """
    Full 8-call pipeline:
      4-call PaS+SR  ->  refined answer (candidate #1)
      4-call quick drafts at T=0.5  ->  candidates #2-#5
      offline ensemble vote across all 5
    Each stage checks the budget and degrades gracefully under pressure.
    """
    candidates: List[str] = []

    # Stage 1: PaS+SR. Always run plan+solve at minimum, since that's the
    # primary candidate. Critique+refine only if budget allows for both.
    plan = _plan(question)                          # call 1
    draft = _solve(question, plan)                  # call 2
    draft_boxed = extract_final_answer(draft)
    if draft_boxed:
        candidates.append(draft_boxed)

    if _budget_left() >= 2:
        critique = _critique(question, draft)       # call 3
        refined = _refine(question, draft, critique)  # call 4
        refined_boxed = extract_final_answer(refined)
        if refined_boxed:
            # Replace the rough draft with the refined one (same lineage).
            if candidates:
                candidates[-1] = refined_boxed
            else:
                candidates.append(refined_boxed)

    # Stage 2: 4 quick independent drafts at higher temperature for diversity.
    # Now PARALLELIZED via parallel.call_llm_concurrent -- the 4 drafts are
    # independent, so they can fire as one round-trip instead of four.
    # Each draft consumes one call from the per-question budget; the
    # thread-safe slot reservation in utils.call_llm prevents overshoot.
    n_drafts = min(NUM_QUICK_DRAFTS, _budget_left())
    if n_drafts > 0:
        replies = call_llm_concurrent(
            [question] * n_drafts,
            system=_DRAFT_SYSTEM,
            temperature=DRAFT_TEMPERATURE,
            max_tokens=DRAFT_MAX_TOKENS,
            max_workers=n_drafts,
        )
        for raw in replies:
            boxed = extract_final_answer(raw)
            if boxed:
                candidates.append(boxed)

    # Stage 3: offline ensemble vote.
    final = _ensemble_pick(candidates) if candidates else ""

    # Last-ditch fallback: if for some reason no boxed candidate survived,
    # return the latest extractable answer to avoid an empty submission.
    if not final and candidates:
        final = candidates[-1]
    return final


# ---------------------------------------------------------------------------
# Router entry point
# ---------------------------------------------------------------------------

def solve(question: str) -> str:
    """Router calls this. Returns the final \\boxed{...} answer string."""
    return plan_solve_refine_with_ensemble(question)
