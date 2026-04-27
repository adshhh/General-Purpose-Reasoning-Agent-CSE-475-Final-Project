import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent import run_agent
from utils import get_per_question_calls, reset_call_count, get_call_count


DEV_PATH_CANDIDATES = [
    Path("data/cse476_final_project_dev_data.json"),
    Path("cse476_final_project_dev_data.json"),
]


def _resolve_dev() -> Path:
    for p in DEV_PATH_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dev data not found. Tried: "
        + ", ".join(str(p) for p in DEV_PATH_CANDIDATES)
    )


def normalize(text):
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def grade(expected, prediction, domain):
    e, p = normalize(expected), normalize(prediction)
    if not p:
        return False
    if e == p:
        return True

    if domain == "math":
        # Compare first numeric token.
        en = re.findall(r"-?\d+\.?\d*", e)
        pn = re.findall(r"-?\d+\.?\d*", p)
        if en and pn and en[0] == pn[0]:
            return True

    if domain == "coding":
        fn = re.search(r"def\s+(\w+)", e)
        if fn and fn.group(1) in p:
            return True

    if domain == "common_sense":
        if e and (e in p or p in e):
            return True

    return False


def _solve_for_eval(item):
    pred = run_agent(item["input"])
    calls = get_per_question_calls()
    return pred, calls


def evaluate(dev_path, limit=None, workers=1):
    with dev_path.open("r") as fp:
        data = json.load(fp)
    if limit:
        data = data[:limit]

    by_domain = {}
    for item in data:
        by_domain.setdefault(item.get("domain", "unknown"), []).append(item)

    print("=" * 70)
    print(f"EVALUATING ON {len(data)} DEV QUESTIONS  (workers={workers})")
    print("=" * 70)

    reset_call_count()
    t0 = time.time()
    results = {}
    total_correct = 0
    total_calls = 0
    total_q = 0

    for domain in sorted(by_domain):
        questions = by_domain[domain]
        correct = 0
        domain_calls = 0
        print(f"\n[{domain.upper()}] {len(questions)} questions")

        if workers <= 1:
            for item in questions:
                pred, calls = _solve_for_eval(item)
                domain_calls += calls
                if grade(item["output"], pred, domain):
                    correct += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_solve_for_eval, item): item for item in questions}
                for fut in as_completed(futures):
                    item = futures[fut]
                    pred, calls = fut.result()
                    domain_calls += calls
                    if grade(item["output"], pred, domain):
                        correct += 1

        acc = (correct / len(questions) * 100) if questions else 0.0
        avg = domain_calls / len(questions) if questions else 0.0
        results[domain] = (len(questions), correct, acc, avg)
        total_correct += correct
        total_calls += domain_calls
        total_q += len(questions)
        print(f"  -> {acc:.1f}%  ({correct}/{len(questions)})  avg calls={avg:.1f}")

    print("\n" + "=" * 70)
    print(f"{'Domain':<22}{'Count':>8}{'Correct':>10}{'Accuracy':>12}{'Avg calls':>12}")
    print("-" * 70)
    for d in sorted(results):
        n, c, a, av = results[d]
        print(f"{d:<22}{n:>8}{c:>10}{a:>11.1f}%{av:>12.1f}")
    print("-" * 70)
    overall_acc = (total_correct / total_q * 100) if total_q else 0.0
    overall_avg = total_calls / total_q if total_q else 0.0
    print(f"{'OVERALL':<22}{total_q:>8}{total_correct:>10}{overall_acc:>11.1f}%{overall_avg:>12.1f}")
    print("=" * 70)
    print(f"Wall time: {(time.time()-t0)/60:.1f} min  |  Total calls: {get_call_count()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    evaluate(_resolve_dev(), limit=args.n, workers=args.workers)


if __name__ == "__main__":
    main()
