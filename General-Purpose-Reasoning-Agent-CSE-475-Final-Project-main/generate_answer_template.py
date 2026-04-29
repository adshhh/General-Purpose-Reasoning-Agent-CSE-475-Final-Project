from __future__ import annotations
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

from agent import run_agent
from utils import (
    get_call_count,
    reset_call_count,
    get_failure_count,
    PER_QUESTION_CAP,
)


INPUT_PATH = Path("data/cse_476_final_project_test_data.json")
OUTPUT_PATH = Path("data/cse_476_final_project_answers.json")
INPUT_PATH_FALLBACK = Path("cse_476_final_project_test_data.json")
OUTPUT_PATH_FALLBACK = Path("cse_476_final_project_answers.json")

CHECKPOINT_EVERY = 50
PER_QUESTION_WALLCLOCK = 180  #hard ceiling per question


def _resolve_input() -> Path:
    if INPUT_PATH.exists():
        return INPUT_PATH
    if INPUT_PATH_FALLBACK.exists():
        return INPUT_PATH_FALLBACK
    raise FileNotFoundError(
        f"Could not find input file at {INPUT_PATH} or {INPUT_PATH_FALLBACK}"
    )


def _resolve_output() -> Path:
    if INPUT_PATH.exists():
        return OUTPUT_PATH
    return OUTPUT_PATH_FALLBACK


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError("Input file must contain a list of question objects.")
    return data


def load_existing_answers(path: Path, expected_len: int) -> List[Dict[str, str]]:
    if not path.exists():
        return [{"output": ""} for _ in range(expected_len)]
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, list):
            return [{"output": ""} for _ in range(expected_len)]
        out: List[Dict[str, str]] = []
        for entry in data:
            if isinstance(entry, dict) and isinstance(entry.get("output"), str):
                out.append({"output": entry["output"]})
            else:
                out.append({"output": ""})
        while len(out) < expected_len:
            out.append({"output": ""})
        return out[:expected_len]
    except (json.JSONDecodeError, OSError):
        return [{"output": ""} for _ in range(expected_len)]


_write_lock = threading.Lock()


def _write_atomic(path: Path, answers: List[Dict[str, str]]) -> None:
    # Write to a tmp file and rename so a crash mid-write can't corrupt anything.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(answers, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def checkpoint(path: Path, answers: List[Dict[str, str]]) -> None:
    with _write_lock:
        _write_atomic(path, answers)


def validate_results(questions, answers) -> None:
    if len(questions) != len(answers):
        raise ValueError(
            f"Mismatched lengths: {len(questions)} questions vs {len(answers)} answers."
        )
    for idx, ans in enumerate(answers):
        if "output" not in ans:
            raise ValueError(f"Missing 'output' field for answer index {idx}.")
        if not isinstance(ans["output"], str):
            raise TypeError(
                f"Answer at index {idx} has non-string output: {type(ans['output'])}"
            )
        if len(ans["output"]) >= 5000:
            raise ValueError(
                f"Answer at index {idx} exceeds 5000 characters "
                f"({len(ans['output'])} chars)."
            )


def _solve_one(idx: int, question: str) -> Dict[str, Any]:
    t0 = time.time()
    try:
        answer = run_agent(question)
    except Exception as e:
        return {
            "idx": idx,
            "output": "",
            "elapsed": time.time() - t0,
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "idx": idx,
        "output": answer,
        "elapsed": time.time() - t0,
        "error": None,
    }


def run(questions, answers, output_path, workers, resume):
    total = len(questions)
    if resume:
        todo = [i for i, a in enumerate(answers) if not a["output"].strip()]
    else:
        todo = list(range(total))

    print(f"Total questions: {total}")
    print(f"To process this run: {len(todo)} (skipping {total - len(todo)} already-answered)")
    print(f"Workers: {workers}")
    print(f"Output: {output_path}")
    print(f"Per-question call cap: {PER_QUESTION_CAP}")
    print("-" * 60, flush=True)

    if not todo:
        print("Nothing to do. Output is already complete.")
        return

    reset_call_count()
    t_start = time.time()
    completed = 0
    errors = 0
    last_checkpoint = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_solve_one, i, questions[i]["input"]): i for i in todo}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                result = fut.result(timeout=PER_QUESTION_WALLCLOCK)
            except Exception as e:
                result = {
                    "idx": idx, "output": "",
                    "elapsed": -1, "error": f"future failed: {e}",
                }

            answers[result["idx"]] = {"output": result["output"]}
            completed += 1
            if result["error"]:
                errors += 1

            if completed - last_checkpoint >= CHECKPOINT_EVERY:
                checkpoint(output_path, answers)
                last_checkpoint = completed

            if completed % 10 == 0 or completed == len(todo):
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 0.001)
                remaining = len(todo) - completed
                eta_min = remaining / max(rate, 0.001) / 60
                total_calls = get_call_count()
                avg_calls = total_calls / max(completed, 1)
                fail_calls = get_failure_count()
                print(
                    f"[{completed}/{len(todo)}] "
                    f"rate={rate:.2f} q/s  "
                    f"avg_calls={avg_calls:.1f}  "
                    f"failed_calls={fail_calls}  "
                    f"errors={errors}  "
                    f"elapsed={elapsed/60:.1f}m  "
                    f"ETA={eta_min:.1f}m",
                    flush=True,
                )

    checkpoint(output_path, answers)
    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"Done. {completed} processed, {errors} errors, {elapsed/60:.1f} minutes total.")
    print(f"Total LLM calls: {get_call_count()}, failed: {get_failure_count()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="Limit to first N questions.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip questions that already have non-empty answers.")
    ap.add_argument("--workers", type=int, default=8, help="Concurrent workers.")
    args = ap.parse_args()

    input_path = _resolve_input()
    output_path = _resolve_output()

    questions = load_questions(input_path)
    if args.n:
        questions = questions[: args.n]
    expected_len = len(questions)

    answers = load_existing_answers(output_path, expected_len)

    try:
        run(questions, answers, output_path, workers=args.workers, resume=args.resume)
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial progress...", flush=True)
        checkpoint(output_path, answers)
        sys.exit(130)

    checkpoint(output_path, answers)
    with output_path.open("r", encoding="utf-8") as fp:
        saved = json.load(fp)
    validate_results(questions, saved)
    filled = sum(1 for a in saved if a["output"].strip())
    print(f"\nWrote {len(saved)} answers to {output_path} ({filled} non-empty).")


if __name__ == "__main__":
    main()
