# CSE 476 Final Project — Report

**General-Purpose Reasoning Agent**
Team: adshhh, Soul, Ismail Wehelie

## Overview

Our agent is a domain-aware router that dispatches each question to a
specialized pipeline. All pipelines share a single API wrapper
(`utils.call_llm`) that calls ASU's `qwen3-30b-a3b-instruct-2507` endpoint
and counts calls per question, hard-capping at 18 so we never exceed the
project's 20-call limit. A two-stage classifier (`router.classify_domain`)
picks one of five domains: math, coding, common sense, future prediction,
or planning. Each domain has its own strategy chosen to trade off accuracy
against call cost.

## Pipeline

```
question
   |
   v
router.classify_domain   (stage 1: keyword heuristic, 0 calls)
   |                     (stage 2: LLM classifier, 1 call -- only when needed)
   v
{math | coding | common_sense | future_prediction | planning}
   |
   v
domain-specific solver (2-6 LLM calls)
   |
   v
answer string (<= 4900 chars)
```

The keyword classifier correctly identifies 100% of coding, future-prediction,
and planning questions on the dev set (no LLM call needed), and 65.7% of math
and 55.5% of common-sense. The remaining questions fall through to the 1-call
LLM classifier, keeping the total classification cost low.

## The Inference-Time Techniques

| #  | Technique                     | File : function                              | Calls |
|----|-------------------------------|----------------------------------------------|-------|
| 1  | Keyword + LLM Router          | `router.py : classify_domain`                | 0-1   |
| 2  | Chain-of-Thought Planning     | `math_solver.py : cot_pal_solve`             | 1     |
| 3  | PAL (Program-Aided Language)  | `math_solver.py : pal_from_cot`              | 1     |
| 4  | Self-Consistency (majority)   | `math_solver.py : cot_answer_solve`          | 3     |
| 5  | Plan-then-Code                | `coding.py : reason_for_code`                | 2     |
| 6  | ReAct (reason + act)          | `common_sense.py : react_solve`              | 3     |
| 7  | Self-Refine                   | `common_sense.py : self_refine`              | 2     |
| 8  | Majority-Vote Ensemble        | `future_prediction.py : vote`                | 3     |
| 9  | Plan-and-Solve                | `planning.py : plan_and_solve`               | 2     |
| 10 | Least-to-Most Decomposition   | `planning.py : least_to_most`                | 2     |
| 11 | Tree of Thoughts              | `planning.py : tree_of_thoughts`             | 6     |

Techniques 1-5 implemented by adshhh. Techniques 6-8 by Soul. Techniques 9-11 by Ismail.

## Design Choices

**Why a router?** The five domains have very different answer formats (a single
integer, a code block, a name, a predicate list, a sequence of PDDL actions).
Routing lets each domain use a prompt template matching its output format,
which improves exact-match accuracy far more than any single universal prompt.

**Budget accounting.** `utils.call_llm` increments a per-question counter, and
raises `CallBudgetExceeded` if a pipeline tries to exceed 18 calls. The agent
catches this and returns a safe empty string, so a single runaway question
never blows up the full test-set run.

**Graceful fallback.** If a domain module isn't implemented yet (e.g. during
development before a teammate has pushed their code), the router catches the
`ImportError` and falls back to a single direct LLM call, so the pipeline
always produces an answer.

**LLM-as-judge grading.** For free-form domains, the evaluator uses the model
itself as a strict True/False grader. Math uses numeric extraction and coding
uses substring containment plus a judge fallback.

## Team Split

- **adshhh (Aditya):** `math_solver.py` (CoT planning + PAL + Self-Consistency fallback), `coding.py` (Plan-then-code), project report.
- **Soul (Angel):** `common_sense.py` (ReAct + Self-Refine), `future_prediction.py` (majority-vote ensemble).
- **Ismail (Ismail):** Router/dispatcher shared infrastructure, `planning.py` (Plan-and-Solve, Least-to-Most, Tree of Thoughts), `evaluator.py`, `utils.py` API wrapper, project report.

## Results on the Dev Set (1,000 examples)

Run `python evaluator.py` to produce numbers. Placeholder table:

| Domain            | Count | Accuracy | Avg. Calls/Q |
|-------------------|-------|----------|--------------|
| common_sense      | 400   | TBD      | TBD          |
| math              | 300   | TBD      | TBD          |
| coding            | 100   | TBD      | TBD          |
| future_prediction | 100   | TBD      | TBD          |
| planning          | 100   | TBD      | TBD          |
| **Overall**       | 1000  | TBD      | TBD          |

## Reproducibility

```bash
# 1. Install deps
pip install requests sympy

# 2. Connect to ASU VPN (sslvpn.asu.edu) and set API key
export OPENAI_API_KEY="<your_key_from_voyager>"

# 3. Quick smoke test (20 questions, mixed domains)
python evaluator.py --n 20

# 4. Full dev-set evaluation
python evaluator.py

# 5. Produce the final test-set answers
python generate_answer_template.py
```
