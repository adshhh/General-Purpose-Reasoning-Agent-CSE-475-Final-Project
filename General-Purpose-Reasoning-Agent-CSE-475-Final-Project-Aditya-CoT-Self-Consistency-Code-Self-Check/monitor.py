#!/usr/bin/env python3
"""Read or live-tail call_log.jsonl. --summary / --tail N / --clear."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


LOG_PATH = Path(__file__).resolve().parent / "call_log.jsonl"
if os.getenv("CALL_LOG_PATH"):
    LOG_PATH = Path(os.environ["CALL_LOG_PATH"])


# ---------------------------------------------------------------------------
# Pretty printing of one call record
# ---------------------------------------------------------------------------

def _fmt_record(rec: Dict) -> str:
    """Render one log entry as a compact two-line block."""
    ok = "OK " if rec.get("ok") else "ERR"
    ts = rec.get("ts", "?")
    ms = rec.get("elapsed_ms", 0)
    t = rec.get("temperature", "?")
    pq = rec.get("per_q_calls", "?")
    tot = rec.get("total_calls", "?")
    pchars = rec.get("prompt_chars", 0)
    rchars = rec.get("reply_chars", 0)

    head = (
        f"[{ts}] {ok} t={t} {ms:>5}ms  "
        f"per_q={pq} total={tot}  "
        f"prompt={pchars}c reply={rchars}c"
    )
    sys_p = rec.get("system_preview", "")
    prm_p = rec.get("prompt_preview", "")
    rep_p = rec.get("reply_preview", "")
    err = rec.get("error")

    body_lines = []
    if sys_p:
        body_lines.append(f"   sys: {sys_p}")
    if prm_p:
        body_lines.append(f"   q  : {prm_p}")
    if rep_p:
        body_lines.append(f"   a  : {rep_p}")
    if err:
        body_lines.append(f"   ERR: {err}")
    return head + "\n" + "\n".join(body_lines)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def _iter_records(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_summary() -> int:
    if not LOG_PATH.exists():
        print(f"No log found at {LOG_PATH}.")
        return 0

    records: List[Dict] = list(_iter_records(LOG_PATH))
    if not records:
        print(f"Log {LOG_PATH} is empty.")
        return 0

    n = len(records)
    n_ok = sum(1 for r in records if r.get("ok"))
    n_err = n - n_ok
    total_ms = sum(int(r.get("elapsed_ms", 0)) for r in records)
    avg_ms = total_ms / n if n else 0
    max_ms = max((int(r.get("elapsed_ms", 0)) for r in records), default=0)

    # Group by per-question runs: a per_q_calls value of 1 marks the start
    # of a new question. Count how many distinct questions we saw and the
    # max calls in any single question.
    per_q_starts = sum(1 for r in records if r.get("per_q_calls") == 1)
    max_per_q = max((int(r.get("per_q_calls", 0)) for r in records), default=0)

    by_temp: Counter = Counter()
    chars_by_temp: Dict[float, List[int]] = defaultdict(list)
    for r in records:
        t = r.get("temperature", "?")
        by_temp[t] += 1
        chars_by_temp[t].append(int(r.get("reply_chars", 0)))

    print(f"=== call_log.jsonl summary ({LOG_PATH}) ===")
    print(f"  total calls       : {n}")
    print(f"  successful        : {n_ok}")
    print(f"  failed/empty      : {n_err}")
    print(f"  questions started : ~{per_q_starts}  (per_q_calls==1 markers)")
    print(f"  max calls in 1 q  : {max_per_q}")
    print(f"  total latency     : {total_ms / 1000:.1f} s")
    print(f"  avg latency/call  : {avg_ms:.0f} ms")
    print(f"  slowest call      : {max_ms} ms")
    print()
    print("  by temperature:")
    for t, count in sorted(by_temp.items(), key=lambda kv: str(kv[0])):
        avg_reply = (sum(chars_by_temp[t]) / len(chars_by_temp[t])) if chars_by_temp[t] else 0
        print(f"    t={t:<5} {count:>4} calls   avg reply {avg_reply:.0f} chars")

    if n_err:
        print()
        print("  recent errors:")
        for r in records[-10:]:
            if not r.get("ok"):
                print(f"    [{r.get('ts')}] {r.get('error') or '(empty reply)'}")
    return 0


def mode_tail_n(n: int) -> int:
    records = list(_iter_records(LOG_PATH))
    if not records:
        print(f"No records yet at {LOG_PATH}.")
        return 0
    for r in records[-n:]:
        print(_fmt_record(r))
        print()
    return 0


def mode_follow() -> int:
    """Live tail. Polls the file every 0.5s for new lines."""
    print(f"Watching {LOG_PATH}  (Ctrl-C to stop)")
    print("(If you haven't run anything yet, this will wait silently.)\n")

    # Make sure the file exists so the loop has something to read.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.touch(exist_ok=True)

    pos = LOG_PATH.stat().st_size  # start at current end -- show only NEW calls
    try:
        while True:
            try:
                size = LOG_PATH.stat().st_size
            except FileNotFoundError:
                time.sleep(0.5)
                continue

            if size < pos:
                # File was truncated (e.g. user ran --clear). Restart.
                pos = 0

            if size > pos:
                with LOG_PATH.open("r", encoding="utf-8") as fp:
                    fp.seek(pos)
                    chunk = fp.read()
                    pos = fp.tell()
                for line in chunk.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    print(_fmt_record(rec))
                    print()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n(stopped)")
        return 0


def mode_clear() -> int:
    if not LOG_PATH.exists():
        print(f"No log at {LOG_PATH} -- nothing to clear.")
        return 0
    print(f"About to wipe {LOG_PATH} ({LOG_PATH.stat().st_size} bytes).")
    confirm = input("Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
        print("Aborted.")
        return 1
    LOG_PATH.write_text("", encoding="utf-8")
    print("Cleared.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Read or live-tail the LLM call log written by call_logger.py. "
            "Default is live tail."
        )
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--summary", action="store_true",
                   help="Print one-shot stats over the whole log and exit.")
    g.add_argument("--tail", type=int, metavar="N",
                   help="Print the last N records and exit.")
    g.add_argument("--clear", action="store_true",
                   help="Wipe the log file (asks for confirmation).")
    args = ap.parse_args()

    if args.summary:
        return mode_summary()
    if args.tail is not None:
        return mode_tail_n(args.tail)
    if args.clear:
        return mode_clear()
    return mode_follow()


if __name__ == "__main__":
    sys.exit(main())
