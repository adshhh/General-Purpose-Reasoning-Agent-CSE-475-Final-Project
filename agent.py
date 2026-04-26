from router import route_and_solve
from utils import reset_per_question_counter, get_per_question_calls, CallBudgetExceeded

def run_agent(question):
    # Reset the call counter for the new question
    reset_per_question_counter()
    
    try:
        # Send to the router to find the right solver
        answer = route_and_solve(question)
    except CallBudgetExceeded:
        # Return empty if we hit the 20-call limit
        answer = ""

    # Simple cleanup
    if answer is None:
        answer = ""
    
    answer = str(answer).strip()
    
    # Keep it under the 5000 character limit for the grader
    if len(answer) > 4900:
        answer = answer[:4900]

    print(f"  [calls used: {get_per_question_calls()}]")
    return answer