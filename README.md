# General-Purpose Reasoning Agent — CSE 476 Final Project

A domain-aware reasoning agent built on ASU's Qwen3 endpoint
(`qwen3-30b-a3b-instruct-2507`). A single router classifies each question into
one of five domains, then dispatches to a domain-specific solver that uses
inference-time techniques (Chain-of-Thought, PAL, Self-Consistency, Step-Back,
RECITE, Plan-and-Solve, Self-Refine, Least-to-Most, Tree of Thoughts).
Per-question LLM-call budget is hard-capped to stay under the project's
20-call limit.

## Pipeline

```
question
   |
   v
router.classify_domain        (keyword heuristic; LLM fallback only when needed)
   |
   v
{ math | coding | common_sense | future_prediction | planning }
   |
   v
domain solver  (2-13 LLM calls)
   |
   v
answer string  (<= 4900 chars)
```

## Techniques per domain

| Domain              | Owner      | File                  | Techniques                                                  |
|---------------------|------------|-----------------------|-------------------------------------------------------------|
| math                | Person A   | `math_solver.py`      | CoT planning + PAL (Program-Aided Language) + Self-Consistency fallback |
| coding              | Person A   | `coding.py`           | Plan-then-code (reasoning step + code-only emission)        |
| common_sense        | Person B   | `common_sense.py`     | Step-Back + RECITE + Adaptive Self-Consistency + USC tie-break |
| future_prediction   | Person B   | `future_prediction.py`| Plan-and-Solve + Self-Refine + 4-draft ensemble vote        |
| planning            | Person C   | `planning.py`         | Plan-and-Solve / Least-to-Most / Tree of Thoughts (default: Least-to-Most) |
| router              | Person C   | `router.py`           | Keyword classifier + 1-call LLM fallback                    |

## Team split

- **Person A — adshhh (Aditya):** `math_solver.py`, `coding.py`
- **Person B — Soul:** `common_sense.py`, `future_prediction.py`, plus the
  parallel/logging/eval infra (`parallel.py`, `call_logger.py`, `monitor.py`,
  `dev_eval.py`, `batch_test.py`, `test_person_b.py`)
- **Person C — Ismail Wehelie:** `router.py`, `planning.py`, `utils.py`,
  `agent.py`, `evaluator.py`, `generate_answer_template.py`

Every Python file carries an `# Owner: …` header at the top.

## Repo layout (flat)

```
.
|-- agent.py                       # entry point: run_agent(question)
|-- router.py                      # 5-domain classifier + dispatcher
|-- utils.py                       # shared LLM wrapper + per-question budget
|-- parallel.py                    # bounded-semaphore parallel call helpers
|-- coding.py                      # Person A
|-- math_solver.py                 # Person A
|-- common_sense.py                # Person B
|-- future_prediction.py           # Person B
|-- planning.py                    # Person C
|-- evaluator.py                   # dev-set accuracy + LLM-as-judge
|-- generate_answer_template.py    # produces data/cse_476_final_project_answers.json
|-- call_logger.py                 # dev-only: side-effect-wraps utils.call_llm
|-- monitor.py                     # dev-only: read/tail call_log.jsonl
|-- dev_eval.py                    # dev-only: per-domain scorecard for cs+fp
|-- batch_test.py                  # dev-only: sample N test items by domain
|-- test_person_b.py               # dev-only: smoke test for cs+fp solvers
|-- data/
|   |-- cse_476_final_project_test_data.json
|   `-- cse_476_final_project_answers.json
`-- final_project_tutorial_and_dev_data/
    |-- cse476_final_project_dev_data.json
    `-- final_project_tutorial.ipynb
```

## How to run

### 1. Install deps

```bash
pip install requests sympy
```

### 2. Connect to ASU VPN, set the API key

```bash
# Cisco Secure Client -> sslvpn.asu.edu
export OPENAI_API_KEY="<your_key_from_voyager>"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY = "<your_key_from_voyager>"
```

### 3. Smoke-test that everything imports

```bash
py -c "import agent, router, utils, coding, math_solver, common_sense, future_prediction, planning; print('All imports OK')"
```

### 4. Quick offline test (no LLM calls)

```bash
py test_person_b.py --extract-only
```

### 5. Full dev-set evaluation

```bash
py evaluator.py --n 20         # 20-question smoke test
py evaluator.py                # full 1,000-example run
```

### 6. Generate the auto-grader answer file

```bash
py generate_answer_template.py --n 50          # 50-question dry-run
py generate_answer_template.py --workers 4     # full ~6,200 questions, 4 in parallel
py generate_answer_template.py --resume        # pick up after a crash / VPN drop
```

Output is written to `data/cse_476_final_project_answers.json`.

## Budget accounting

`utils.call_llm` keeps a per-question counter and raises `CallBudgetExceeded`
if a pipeline tries to exceed the 20-call cap. `agent.run_agent` catches this
and returns an empty string for that question rather than crashing the run.
The parallel helpers in `parallel.py` use a thread-local mutable cell so that
concurrent sub-calls inside one question all increment the SAME counter,
keeping the budget honest under parallelism.

## Graceful fallback

If a domain solver isn't implemented yet, the router catches the `ImportError`
and falls back to a single direct LLM call. This means the agent always
produces *some* answer for every question.
