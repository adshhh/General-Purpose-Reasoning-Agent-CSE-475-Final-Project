# Owner: Person B (Soul) - parallel-LLM helper
"""Parallel LLM helper: bounded semaphore + thread-local per-question counter cell."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from utils import (
    call_llm,
    CallBudgetExceeded,
    get_current_cell,
    set_current_cell,
)


# ---------------------------------------------------------------------------
# Concurrency caps
# ---------------------------------------------------------------------------

DEFAULT_API_CONCURRENCY = int(os.getenv("MAX_CONCURRENT_API_CALLS", "12"))
DEFAULT_BATCH_WORKERS = 6

_API_SEMAPHORE = threading.BoundedSemaphore(DEFAULT_API_CONCURRENCY)


def api_concurrency_limit() -> int:
    return DEFAULT_API_CONCURRENCY


# ---------------------------------------------------------------------------
# Cell-binding wrapper
# ---------------------------------------------------------------------------

def _bind_and_run(cell, fn, args, kwargs):
    """
    Bind `cell` as the current thread's per-question counter, run fn, then
    detach. Pool threads are reused across tasks; without the detach a cell
    from one task could leak into the next.
    """
    set_current_cell(cell)
    try:
        return fn(*args, **kwargs)
    finally:
        set_current_cell(None)


# ---------------------------------------------------------------------------
# Single throttled call
# ---------------------------------------------------------------------------

def _throttled_call(spec: Dict[str, Any]) -> str:
    """
    Acquire the global API semaphore, run call_llm with the kwargs in `spec`,
    then release. Translates CallBudgetExceeded and other exceptions to "".
    """
    _API_SEMAPHORE.acquire()
    try:
        return call_llm(**spec) or ""
    except CallBudgetExceeded:
        return ""
    except Exception:
        return ""
    finally:
        _API_SEMAPHORE.release()


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def call_llm_batch(
    specs: Sequence[Dict[str, Any]],
    max_workers: int = DEFAULT_BATCH_WORKERS,
) -> List[str]:
    """
    Run multiple call_llm invocations concurrently.

    Returns a list of replies in the SAME ORDER as `specs`. A failed spec
    contributes "" to the output list.

    Sub-threads inherit the parent thread's per-question counter cell so
    all increments land on the SAME counter. The cell is captured ONCE
    here at submission time; each pool thread binds it explicitly via
    _bind_and_run.
    """
    if not specs:
        return []

    parent_cell = get_current_cell()
    results: List[Optional[str]] = [None] * len(specs)

    def _run(idx_spec):
        idx, spec = idx_spec
        return idx, _bind_and_run(parent_cell, _throttled_call, (spec,), {})

    workers = min(max_workers, len(specs), DEFAULT_API_CONCURRENCY)
    workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, (i, s)) for i, s in enumerate(specs)]
        for fut in as_completed(futures):
            idx, reply = fut.result()
            results[idx] = reply

    return [r if r is not None else "" for r in results]


def call_llm_concurrent(
    prompts: Sequence[str],
    system: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: int = 60,
    max_workers: int = DEFAULT_BATCH_WORKERS,
) -> List[str]:
    """Send N prompts with the SAME system / temperature / max_tokens in parallel."""
    specs = [
        {
            "prompt": p,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        for p in prompts
    ]
    return call_llm_batch(specs, max_workers=max_workers)


# ---------------------------------------------------------------------------
# Cross-question runner
# ---------------------------------------------------------------------------

def submit_in_context(pool: ThreadPoolExecutor, fn, *args, **kwargs):
    """
    Submit fn to the pool with the CALLER'S per-question cell (which is
    typically None at this layer -- dev_eval starts each question in a
    fresh worker that calls reset_per_question_counter() FIRST and then
    spawns its own sub-pools via call_llm_batch). Detaches the bound cell
    in finally so a pool thread cleanly accepts the next task.
    """
    parent_cell = get_current_cell()
    return pool.submit(_bind_and_run, parent_cell, fn, args, kwargs)
