import os
import requests

# 
class CallBudgetExceeded(Exception): pass

API_KEY = os.getenv("OPENAI_API_KEY", "your_key_here")
API_BASE = "https://openai.rc.asu.edu/v1"
MODEL = "qwen3-30b-a3b-instruct-2507"

_count = 0

def get_per_question_calls():
    return _count

def reset_per_question_counter():
    global _count
    _count = 0

def call_llm(prompt, system="Helpful assistant.", temperature=0.0):
    global _count
    # Hard-cap at 18 so we never exceed the project's 20-call limit
    if _count >= 18:
        raise CallBudgetExceeded("Out of calls!")
    
    _count += 1
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    
    try:
        r = requests.post(f"{API_BASE}/chat/completions", json=data, headers=headers)
        return r.json()['choices'][0]['message']['content'].strip()
    except:
        return ""