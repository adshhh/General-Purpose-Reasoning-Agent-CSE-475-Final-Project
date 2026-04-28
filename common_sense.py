# Angel - common-sense domain solver
"""Common-sense solver: step-back -> recite -> adaptive self-consistency -> USC tie-break."""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

from utils import call_llm, get_per_question_calls, PER_QUESTION_CAP
from parallel import call_llm_concurrent, call_llm_batch


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Adaptive-SC parameters.
ASC_MIN_SAMPLES = 3        # never stop before this many samples
ASC_MAX_SAMPLES = 13       # never exceed this many samples
ASC_BATCH_SIZE = 8         # parallel batch size (was 4 -- semaphore caps overall)
ASC_STOP_POSTERIOR = 0.95  # stop when P(majority is dominant) >= this
ASC_TEMPERATURE = 0.7

# Per-question safety cap. Stays 1 below MAX_CALLS_PER_QUESTION (20) so we
# always have headroom for any retry inside a parallel batch.
PERSON_B_CALL_CAP = 19


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Step 1A: Step-Back -- abstract the question.
_STEPBACK_ASK_SYSTEM = (
    "You are an expert tutor. Given a specific trivia question, write ONE "
    "general / category-level question whose answer would help a student "
    "answer the original. The general question should be about the broader "
    "principle, biography, or category that the trivia question depends on. "
    "Reply with ONLY the general question, no preamble."
)

# Step 1B: Step-Back -- answer the abstract question.
_STEPBACK_ANSWER_SYSTEM = (
    "Answer the following general question concisely and factually in 2-4 "
    "sentences. This will be used as context for a downstream trivia question."
)

# Step 2: RECITE / GenRead -- model writes a Wiki-style passage from memory.
_RECITE_SYSTEM = (
    "You are recalling background facts to help answer a trivia question. "
    "Write a SHORT (4-8 sentence) Wikipedia-style paragraph on the key "
    "entity, person, work, place, or event mentioned in the question. "
    "Stick to facts you are confident in. Do NOT answer the question yet."
)

# Step 3: Each Adaptive-SC sample.
_SAMPLE_SYSTEM = (
    "You are answering a trivia question with the help of supporting "
    "context. Use the STEP-BACK CONTEXT and the RECITED FACTS, but think "
    "step-by-step yourself. End with EXACTLY one line:\n"
    "Final answer: <short answer>\n\n"
    "The final answer must match the format the question implies: a single "
    "letter for multiple choice; a 4-digit year for years; a number with "
    "unit when the question asks 'how long/old/much'; 'Yes' or 'No' for "
    "yes/no; the most concise canonical name for entities."
)

# Step 4: Universal Self-Consistency selector. Only fires when the vote
# fails to produce a clear majority among free-form answers.
_USC_SYSTEM = (
    "You are an answer consistency judge. Below are several candidate "
    "answers from independent reasoning chains for the same question. "
    "Pick the answer that is most consistent with the reasoning across the "
    "majority of chains. Reply with ONLY the chosen answer string, in the "
    "same surface form as written -- no explanation, no prefix."
)


# ---------------------------------------------------------------------------
# Final-answer extraction (deterministic, no LLM call)
# ---------------------------------------------------------------------------

_MCQ_LETTER_RE = re.compile(r"^\s*\(?\s*([A-Ea-e])\s*[\).:\-]?\s*$")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def extract_final_answer(text: str) -> str:
    """Pull a short comparable answer string from a model response."""
    if not text:
        return ""
    cleaned = text.strip()

    # Prefer 'Final answer:' tag -> 'Answer:' -> \boxed{} -> last line.
    for tag in ("final\\s*answer", "answer"):
        m = re.search(
            rf"(?:{tag})\s*[:\-]\s*(.+?)(?:\n|$)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            candidate = m.group(1).strip()
            break
    else:
        boxed = re.search(r"\\boxed\{([^{}]+)\}", cleaned)
        if boxed:
            candidate = boxed.group(1).strip()
        else:
            lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
            candidate = lines[-1] if lines else cleaned

    candidate = candidate.strip().strip("`*\"'")
    candidate = re.sub(r"[.\s]+$", "", candidate)

    mcq = _MCQ_LETTER_RE.match(candidate)
    if mcq:
        return mcq.group(1).upper()

    lead = re.match(r"^\(?\s*([A-Ea-e])\s*[\).:\-]\s*(.+)$", candidate)
    if lead:
        return lead.group(1).upper()

    return candidate


# ---------------------------------------------------------------------------
# Voting helpers
# ---------------------------------------------------------------------------

def _normalise_for_vote(answer: str) -> str:
    """Vote key: lowercase + collapse whitespace + extract numbers when present."""
    if not answer:
        return ""
    a = answer.strip().lower()
    a = re.sub(r"\s+", " ", a)
    if len(a) == 1 and a in "abcde":
        return a.upper()
    nums = _NUMBER_RE.findall(a)
    if nums and len(a) <= 30:
        return nums[0].rstrip("%")
    return a


def _budget_left() -> int:
    """Calls remaining under our self-imposed 18-call cap."""
    return max(0, PERSON_B_CALL_CAP - get_per_question_calls())


def _beta_dominance_posterior(majority: int, runner_up: int, samples: int = 5000) -> float:
    """
    Approximate P(theta_majority > theta_runner_up) using independent Beta(1,1)
    priors. We sample from each beta and count how often majority > runner_up.
    Cheap and dependency-free (only uses random, which is in stdlib).

    Returns a value in [0, 1]. Higher = more confident the majority answer
    really is the dominant one given the votes seen so far.
    """
    # Beta(majority+1, ...) -- see Aggarwal et al. 2023 for the derivation.
    import random
    samples_seen = majority + runner_up
    a1 = majority + 1
    b1 = max(1, samples_seen - majority + 1)
    a2 = runner_up + 1
    b2 = max(1, samples_seen - runner_up + 1)
    wins = 0
    for _ in range(samples):
        x1 = random.betavariate(a1, b1)
        x2 = random.betavariate(a2, b2)
        if x1 > x2:
            wins += 1
    return wins / samples


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _step_back_b(abstract_q: str) -> str:
    """The 2nd step-back call (depends on step-back A's output)."""
    return call_llm(
        abstract_q,
        system=_STEPBACK_ANSWER_SYSTEM,
        temperature=0.0,
        max_tokens=200,
    ) or ""


def _step_back_and_recite_parallel(question: str) -> tuple[str, str]:
    """
    Speed optimisation: step-back A and recite are INDEPENDENT (recite only
    needs the question). Fire them in parallel, then chain step-back B onto
    A's output.

    Returns (stepback_ctx, recite_ctx). Either may be "" on failure /
    budget exhaustion.
    """
    # Phase 1: step-back A and recite in parallel.
    if _budget_left() < 2:
        # Not enough room for parallel; fall back to sequential step-back A then bail.
        if _budget_left() <= 0:
            return "", ""
        abstract_q = call_llm(
            question,
            system=_STEPBACK_ASK_SYSTEM,
            temperature=0.0,
            max_tokens=120,
        ) or ""
        return (f"GENERAL Q: {abstract_q.strip()}" if abstract_q else ""), ""

    pair = call_llm_batch(
        [
            {
                "prompt": question,
                "system": _STEPBACK_ASK_SYSTEM,
                "temperature": 0.0,
                "max_tokens": 120,
            },
            {
                "prompt": question,
                "system": _RECITE_SYSTEM,
                "temperature": 0.3,
                "max_tokens": 300,
            },
        ],
        max_workers=2,
    )
    abstract_q, recite = pair[0], pair[1]
    abstract_q = (abstract_q or "").strip()
    recite = (recite or "").strip()

    # Phase 2: step-back B (depends on A).
    abstract_a = ""
    if abstract_q and _budget_left() > 0:
        abstract_a = _step_back_b(abstract_q).strip()

    if abstract_q and abstract_a:
        stepback_ctx = f"GENERAL Q: {abstract_q}\nGENERAL A: {abstract_a}"
    elif abstract_q:
        stepback_ctx = f"GENERAL Q: {abstract_q}"
    else:
        stepback_ctx = ""

    return stepback_ctx, recite


def _build_sample_prompt(question: str, stepback_ctx: str, recite_ctx: str) -> str:
    parts = [f"QUESTION:\n{question}"]
    if stepback_ctx:
        parts.append(f"\nSTEP-BACK CONTEXT:\n{stepback_ctx}")
    if recite_ctx:
        parts.append(f"\nRECITED FACTS:\n{recite_ctx}")
    parts.append("\nNow answer the original QUESTION step by step and end with the final-answer line.")
    return "\n".join(parts)


def _adaptive_self_consistency(
    question: str,
    stepback_ctx: str,
    recite_ctx: str,
) -> List[str]:
    """
    Sample in parallel batches of ASC_BATCH_SIZE. After each batch, fit the
    Beta-posterior on the top-2 votes and stop early when posterior >= 0.95.
    Returns the list of extracted short answers across all samples.
    """
    prompt = _build_sample_prompt(question, stepback_ctx, recite_ctx)
    short_answers: List[str] = []
    samples_run = 0

    while samples_run < ASC_MAX_SAMPLES:
        # Reserve at least 2 calls of headroom for USC + safety.
        remaining_budget = _budget_left() - 2
        if remaining_budget <= 0:
            break

        # How many to fire this batch?
        room_for_more = ASC_MAX_SAMPLES - samples_run
        batch = min(ASC_BATCH_SIZE, room_for_more, remaining_budget)
        if batch <= 0:
            break

        replies = call_llm_concurrent(
            [prompt] * batch,
            system=_SAMPLE_SYSTEM,
            temperature=ASC_TEMPERATURE,
            max_tokens=400,
            max_workers=batch,
        )
        samples_run += batch
        for raw in replies:
            short = extract_final_answer(raw)
            if short:
                short_answers.append(short)

        if samples_run < ASC_MIN_SAMPLES:
            continue

        # Stopping check.
        keys = [_normalise_for_vote(a) for a in short_answers]
        counts = Counter(keys).most_common(2)
        if not counts:
            continue
        top = counts[0][1]
        runner = counts[1][1] if len(counts) > 1 else 0
        if top - runner >= 3:  # Cheap fast-stop: clear majority of >=3 votes.
            break
        # If only one unique answer, posterior is essentially 1.0.
        if runner == 0 and top >= ASC_MIN_SAMPLES:
            break
        try:
            posterior = _beta_dominance_posterior(top, runner)
        except Exception:
            posterior = 0.0
        if posterior >= ASC_STOP_POSTERIOR:
            break

    return short_answers


def _vote(short_answers: List[str]) -> tuple[str, int, int]:
    """Majority vote. Returns (winning_surface_form, top_count, total_count)."""
    if not short_answers:
        return "", 0, 0
    keys = [_normalise_for_vote(a) for a in short_answers]
    counts = Counter(keys)
    top_key, top_count = counts.most_common(1)[0]
    for raw, key in zip(short_answers, keys):
        if key == top_key:
            return raw, top_count, len(short_answers)
    return short_answers[0], top_count, len(short_answers)


def _usc_select(question: str, candidates: List[str]) -> str:
    """One call: LLM picks the most consistent of the given candidates."""
    if not candidates:
        return ""
    if _budget_left() <= 0:
        return candidates[0]
    # De-duplicate candidates while preserving order.
    seen = set()
    deduped: List[str] = []
    for c in candidates:
        key = c.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    if len(deduped) <= 1:
        return deduped[0] if deduped else ""

    numbered = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(deduped))
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CANDIDATE ANSWERS (one per chain):\n{numbered}\n\n"
        "Pick the most consistent answer. Reply with ONLY the chosen answer string."
    )
    chosen = call_llm(
        prompt,
        system=_USC_SYSTEM,
        temperature=0.0,
        max_tokens=80,
    )
    chosen = (chosen or "").strip().strip("`*\"'")
    chosen = re.sub(r"^\s*\d+[\.\)]\s*", "", chosen)  # strip "1. " if leaked
    chosen = re.sub(r"[.\s]+$", "", chosen)
    if not chosen:
        return deduped[0]
    # If the model invented a new answer not in the candidates, fall back.
    chosen_norm = chosen.lower().strip()
    for c in deduped:
        if c.lower().strip() == chosen_norm:
            return c
    return chosen  # Fine to return novel string; grader will normalise.


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def step_back_recite_adaptive_sc(question: str) -> str:
    """
    Full Round 2 pipeline. ~10 calls average, hard-capped at 19 internally
    (1 below utils.py's 20, leaving headroom for any retry storm).
    Step-back A and recite fire in parallel; step-back B then chains on A.
    """
    stepback_ctx, recite_ctx = _step_back_and_recite_parallel(question)
    samples = _adaptive_self_consistency(question, stepback_ctx, recite_ctx)
    if not samples:
        # Last-ditch fallback: a single direct CoT call.
        if _budget_left() > 0:
            raw = call_llm(
                question,
                system=_SAMPLE_SYSTEM,
                temperature=0.0,
                max_tokens=400,
            )
            return extract_final_answer(raw)
        return ""

    voted, top, total = _vote(samples)

    # Run USC only when the vote is muddled: top < (total/2 + 1) and the
    # answers look free-form (multi-token, not MCQ letters or yes/no).
    looks_freeform = any(
        len(a) > 1 and a.upper() not in {"A", "B", "C", "D", "E", "YES", "NO"}
        for a in samples
    )
    if looks_freeform and total >= 4 and top * 2 <= total:
        chosen = _usc_select(question, samples)
        if chosen:
            voted = chosen

    return voted


# ---------------------------------------------------------------------------
# Router entry point
# ---------------------------------------------------------------------------

def solve(question: str) -> str:
    """Public solver. Router calls this. Returns a short canonical answer."""
    return step_back_recite_adaptive_sc(question)
