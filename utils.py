import os
import sys
import time
import threading
import requests


class CallBudgetExceeded(Exception):
    pass


API_KEY = os.getenv("OPENAI_API_KEY", "your_key_here")
API_BASE = os.getenv("API_BASE", "https://openai.rc.asu.edu/v1")
MODEL = os.getenv("MODEL_NAME", "qwen3-30b-a3b-instruct-2507")

#allows 20 calls/question.
PER_QUESTION_CAP = 20
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 512

_global_lock = threading.Lock()
_global_count = 0
_global_failures = 0

# Cell system: each question gets a mutable _Counter shared between the question
# thread and any parallel worker threads spawned by parallel.py (call_llm_batch /
# call_llm_concurrent). Workers bind the parent's cell before running so all
# call_llm increments land on the same per-question budget counter.
_cell_local = threading.local()


class _Counter:
    __slots__ = ("count",)
    def __init__(self):
        self.count = 0


def get_current_cell():
    return getattr(_cell_local, "cell", None)


def set_current_cell(cell):
    _cell_local.cell = cell


def reset_per_question_counter():
    _cell_local.cell = _Counter()


def get_per_question_calls():
    cell = get_current_cell()
    return cell.count if cell is not None else 0


def get_call_count():
    return _global_count


def reset_call_count():
    global _global_count, _global_failures
    with _global_lock:
        _global_count = 0
        _global_failures = 0


def get_failure_count():
    return _global_failures


def _bump_counters():
    global _global_count
    cell = get_current_cell()
    if cell is not None:
        cell.count += 1
    with _global_lock:
        _global_count += 1


def _bump_failures():
    global _global_failures
    with _global_lock:
        _global_failures += 1


def call_llm(prompt, system="You are a helpful assistant.", temperature=0.0,
             max_tokens=DEFAULT_MAX_TOKENS):
    cur = get_per_question_calls()
    if cur >= PER_QUESTION_CAP:
        raise CallBudgetExceeded(f"Per-question call cap reached ({PER_QUESTION_CAP}).")

    _bump_counters()

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

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return (content or "").strip()

            # Retry on rate limit and server errors.
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep(2 ** attempt)
                continue

            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            break

        except requests.exceptions.Timeout:
            last_err = "timeout"
            time.sleep(2 ** attempt)
        except requests.exceptions.ConnectionError as e:
            last_err = f"connection error: {e}"
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)

    _bump_failures()
    print(f"[call_llm] FAILED after {MAX_RETRIES} attempts: {last_err}",
          file=sys.stderr, flush=True)
    return ""
