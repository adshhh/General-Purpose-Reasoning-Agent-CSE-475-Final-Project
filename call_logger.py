# Owner: Person B (Soul) - dev-only call logger (side-effect import wraps utils.call_llm)
"""Side-effect import: wraps utils.call_llm to append each call to call_log.jsonl."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

import utils  # the team's shared API wrapper

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LOG_PATH = Path(__file__).resolve().parent / "call_log.jsonl"
PREVIEW_CHARS = 200          # how much of system/prompt/reply to keep
SYSTEM_PREVIEW_CHARS = 120   # system prompts are short; keep tighter

# Override the log path with env var if needed (e.g. for parallel runs).
if os.getenv("CALL_LOG_PATH"):
    LOG_PATH = Path(os.environ["CALL_LOG_PATH"])

_write_lock = Lock()         # protects file appends from concurrent threads
_PATCHED_FLAG = "_call_logger_patched"


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def _truncate(s: str, n: int) -> str:
    if not s:
        return ""
    s = str(s).replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "..."


def _append(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        # Open per-write so a crash mid-run still leaves valid JSONL.
        with LOG_PATH.open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")


_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Reply with only the final answer"
    "—no explanation."
)


def _make_wrapper(orig_call_llm):
    """Return a wrapped call_llm that logs each call."""

    def wrapped(
        prompt,
        system=_DEFAULT_SYSTEM,
        temperature=0.0,
        max_tokens=512,
        timeout=60,
    ):
        t0 = time.time()
        try:
            reply = orig_call_llm(
                prompt,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            err = None
        except Exception as e:
            reply = ""
            err = f"{type(e).__name__}: {e}"
            elapsed_ms = int((time.time() - t0) * 1000)
            _append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "elapsed_ms": elapsed_ms,
                "ok": False,
                "error": err,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_chars": len(prompt or ""),
                "reply_chars": 0,
                "total_calls": utils.get_call_count(),
                "per_q_calls": utils.get_per_question_calls(),
                "system_preview": _truncate(system or "", SYSTEM_PREVIEW_CHARS),
                "prompt_preview": _truncate(prompt or "", PREVIEW_CHARS),
                "reply_preview": "",
            })
            raise

        elapsed_ms = int((time.time() - t0) * 1000)
        _append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "elapsed_ms": elapsed_ms,
            "ok": bool(reply),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_chars": len(prompt or ""),
            "reply_chars": len(reply or ""),
            "total_calls": utils.get_call_count(),
            "per_q_calls": utils.get_per_question_calls(),
            "system_preview": _truncate(system or "", SYSTEM_PREVIEW_CHARS),
            "prompt_preview": _truncate(prompt or "", PREVIEW_CHARS),
            "reply_preview": _truncate(reply or "", PREVIEW_CHARS),
        })
        return reply

    # Preserve the original docstring/name for debugging.
    wrapped.__name__ = getattr(orig_call_llm, "__name__", "call_llm")
    wrapped.__doc__ = getattr(orig_call_llm, "__doc__", None)
    wrapped.__wrapped__ = orig_call_llm  # introspectable
    return wrapped


def _patch_once() -> None:
    """Idempotent: only wrap utils.call_llm a single time per process."""
    if getattr(utils, _PATCHED_FLAG, False):
        return
    utils.call_llm = _make_wrapper(utils.call_llm)
    setattr(utils, _PATCHED_FLAG, True)

    # Make sure modules that already imported `from utils import call_llm`
    # (router.py, planning.py, common_sense.py, future_prediction.py) also
    # see the wrapped version. We rebind the name in each module that has
    # already imported it.
    import sys
    for mod in list(sys.modules.values()):
        if mod is None or not hasattr(mod, "call_llm"):
            continue
        if getattr(mod, "call_llm", None) is wrapped_target_orig:
            mod.call_llm = utils.call_llm


# Capture the *original* before wrapping, so we can find any module that
# already imported it and rebind to the wrapped version.
wrapped_target_orig = utils.call_llm
_patch_once()
