"""
Shared LLM API wrapper for CSE 476 Final Project.

Wraps ASU's OpenAI-compatible endpoint (qwen3-30b-a3b-instruct-2507) with:
  - call counting (global + per-question)
  - retries on transient errors
  - a hard budget (MAX_CALLS_PER_QUESTION) to protect our <=20 calls/question limit
  - a simple `call_llm(prompt, system, temperature)` interface that every
    domain module (math.py, coding.py, planning.py, etc.) uses.

Connection requirements:
  - Cisco VPN connected to sslvpn.asu.edu
  - Environment variable OPENAI_API_KEY set (from the Voyager portal)
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Project spec: max 20 LLM calls per question (10% grade penalty if exceeded).
# We set the internal limit at 18 to leave safety margin.
MAX_CALLS_PER_QUESTION = 18

# Retry settings for transient network / server errors.
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


# ---------------------------------------------------------------------------
# Call counters
# ---------------------------------------------------------------------------

_total_calls = 0            # across entire program run
_per_question_calls = 0     # reset at the start of each question


class CallBudgetExceeded(Exception):
    """Raised when a single question tries to exceed MAX_CALLS_PER_QUESTION."""


def reset_per_question_counter() -> None:
    """Call this at the start of handling each new question."""
    global _per_question_calls
    _per_question_calls = 0


def reset_call_count() -> None:
    """Reset BOTH counters. Use for a fresh eval run."""
    global _total_calls, _per_question_calls
    _total_calls = 0
    _per_question_calls = 0


def get_call_count() -> int:
    """Total LLM calls made since last reset_call_count()."""
    return _total_calls


def get_per_question_calls() -> int:
    """Calls made on the current question."""
    return _per_question_calls


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def call_llm(
    prompt: str,
    system: str = "You are a helpful assistant. Reply with only the final answer—no explanation.",
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
) -> str:
    """
    Send a single chat completion to the model and return the text reply.

    Raises CallBudgetExceeded if we'd exceed MAX_CALLS_PER_QUESTION for this
    question. Returns "" on hard failure after retries (instead of raising)
    so that agent logic can gracefully fall back.
    """
    global _total_calls, _per_question_calls

    if _per_question_calls >= MAX_CALLS_PER_QUESTION:
        raise CallBudgetExceeded(
            f"Already made {_per_question_calls} calls on this question "
            f"(limit {MAX_CALLS_PER_QUESTION})."
        )

    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run `export OPENAI_API_KEY=...` or "
            "add it to your ~/.zshrc and restart your shell."
        )

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                _total_calls += 1
                _per_question_calls += 1
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            # Retry on transient server errors; give up on 4xx auth errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            # Non-retriable (e.g. 401 Unauthorized, 400 bad request)
            return ""
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = str(e)
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

    print(f"[call_llm] All {MAX_RETRIES} retries failed. Last error: {last_error}")
    return ""