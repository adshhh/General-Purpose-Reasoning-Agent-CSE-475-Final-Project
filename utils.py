# Ismail - shared API wrapper. Concurrency cells added by Person B (Soul).
"""
Shared LLM API wrapper for CSE 476 Final Project.

Wraps ASU's OpenAI-compatible endpoint (qwen3-30b-a3b-instruct-2507) with:
  - call counting (global + per-question)
  - retries on transient errors
  - a hard budget (MAX_CALLS_PER_QUESTION) to protect our <=20 calls/question limit
  - a simple `call_llm(prompt, system, temperature)` interface that every
    domain module (math_solver.py, coding.py, planning.py, etc.) uses.

Concurrency model:
  - The per-question call counter is held in a thread-local mutable cell
    (a one-element list). When dev_eval runs questions in parallel via a
    ThreadPoolExecutor, each worker thread gets its own counter; sub-pools
    created by parallel.call_llm_concurrent inherit the parent question's
    cell via parallel.set_current_cell so all sub-calls increment the SAME
    counter. This lets us safely run multiple questions at once without
    conflating their per-question budgets.
  - The aggregate `_total_calls` counter remains a simple lock-protected
    global.

Connection requirements:
  - Cisco VPN connected to sslvpn.asu.edu
  - Environment variable OPENAI_API_KEY set (from the Voyager portal)
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("OPENAI_API_KEY", "")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

# Project spec: max 20 LLM calls per question (10% grade penalty if exceeded).
# We sit exactly at the cap. Solver code self-limits 1 below this to leave
# room for any in-flight retry inside a parallel batch.
MAX_CALLS_PER_QUESTION = 20

# Retry settings for transient network / server errors.
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


# ---------------------------------------------------------------------------
# Call counters
# ---------------------------------------------------------------------------

# Aggregate total across whole program run. Always a simple int; protected
# by _counter_lock since multiple worker threads can update it concurrently.
_total_calls: int = 0
_counter_lock = threading.Lock()

# Per-question counter. Each thread carries a `cell` attribute (a one-element
# list, mutable so concurrent sub-threads can share writes). When parallel.py
# spawns sub-threads to run independent calls, it explicitly binds the
# parent's cell into each worker thread BEFORE running the task, then unsets
# it afterward. This is simpler than contextvars and avoids the
# "Context already entered" error that occurs when the same Context object
# is run on multiple threads concurrently.
_per_q_local = threading.local()


class CallBudgetExceeded(Exception):
    """Raised when a single question tries to exceed MAX_CALLS_PER_QUESTION."""


def reset_per_question_counter() -> None:
    """
    Call this at the start of handling each new question.

    Installs a fresh mutable cell on the current thread. Sub-threads that
    will share this cell must be started via parallel.submit_in_context or
    parallel.call_llm_batch, which explicitly bind the cell on each worker.
    """
    _per_q_local.cell = [0]


def reset_call_count() -> None:
    """Reset BOTH counters. Use for a fresh eval run."""
    global _total_calls
    with _counter_lock:
        _total_calls = 0
    _per_q_local.cell = [0]


def get_call_count() -> int:
    """Total LLM calls made since last reset_call_count()."""
    return _total_calls


def get_per_question_calls() -> int:
    """Calls made on the current question (in this thread's cell)."""
    cell = getattr(_per_q_local, "cell", None)
    return cell[0] if cell else 0


def get_current_cell() -> Optional[List[int]]:
    """Internal helper for parallel.py: snapshot the current thread's cell."""
    return getattr(_per_q_local, "cell", None)


def set_current_cell(cell: Optional[List[int]]) -> None:
    """
    Internal helper for parallel.py: bind a cell onto the current thread.
    Pass None to detach (so a pool thread doesn't leak a cell from one
    task into the next).
    """
    if cell is None:
        if hasattr(_per_q_local, "cell"):
            del _per_q_local.cell
    else:
        _per_q_local.cell = cell


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
    global _total_calls

    # Resolve the per-question cell from the calling thread's local. If none
    # is active (shouldn't happen under dev_eval / agent.run_agent which both
    # reset before each q), we still track totals and skip the per-question
    # budget check.
    cell = getattr(_per_q_local, "cell", None)

    # Atomically check the budget AND reserve a slot. Otherwise, parallel
    # callers can all observe `< cap`, fire concurrently, and exceed it.
    with _counter_lock:
        if cell is not None and cell[0] >= MAX_CALLS_PER_QUESTION:
            raise CallBudgetExceeded(
                f"Already made {cell[0]} calls on this question "
                f"(limit {MAX_CALLS_PER_QUESTION})."
            )
        # Reserve the slot now; we'll roll it back below if the request fails.
        if cell is not None:
            cell[0] += 1
        _total_calls += 1

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

    def _refund() -> None:
        # Roll back the slot we reserved up top -- the request never produced
        # a useful response, so it shouldn't count against the budget.
        global _total_calls
        with _counter_lock:
            if cell is not None and cell[0] > 0:
                cell[0] -= 1
            if _total_calls > 0:
                _total_calls -= 1

    last_error: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                # Slot was already reserved at the top of this function -- no
                # double-counting here.
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            # Retry on transient server errors; give up on 4xx auth errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue
            # Non-retriable (e.g. 401 Unauthorized, 400 bad request)
            _refund()
            return ""
        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = str(e)
            time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

    print(f"[call_llm] All {MAX_RETRIES} retries failed. Last error: {last_error}")
    _refund()
    return ""
