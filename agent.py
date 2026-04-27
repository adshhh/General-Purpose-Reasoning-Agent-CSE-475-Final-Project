import sys
from router import route_and_solve
from utils import (
    reset_per_question_counter,
    get_per_question_calls,
    CallBudgetExceeded,
)


MAX_ANSWER_CHARS = 4900  


def run_agent(question):
    reset_per_question_counter()

    try:
        answer = route_and_solve(question)
    except CallBudgetExceeded:
        answer = ""
    except Exception as e:
        #safety so one bad question never kills the whole run.
        print(f"[agent] uncaught error ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)
        answer = ""

    if answer is None:
        answer = ""
    answer = str(answer).strip()
    if len(answer) > MAX_ANSWER_CHARS:
        answer = answer[:MAX_ANSWER_CHARS]

    return answer


def run_agent_with_calls(question):
    answer = run_agent(question)
    return answer, get_per_question_calls()
