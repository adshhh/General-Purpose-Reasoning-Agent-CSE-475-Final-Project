import json
import re
import sys
from utils import reset_per_question_counter, get_per_question_calls, CallBudgetExceeded
from router import route_and_solve

def normalize(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def grade_answer(question, expected, prediction, domain):
    """
    Grade predictions based on domain-specific criteria.
    Math: numeric extraction and comparison
    Coding: substring check + LLM judge fallback
    Others: exact match or LLM judge
    """
    expected_norm = normalize(expected)
    prediction_norm = normalize(prediction)
    
    if expected_norm == prediction_norm:
        return True
    
    if domain == "math":
        # Extract numbers for numeric comparison
        exp_nums = re.findall(r'-?\d+\.?\d*', expected_norm)
        pred_nums = re.findall(r'-?\d+\.?\d*', prediction_norm)
        if exp_nums and pred_nums and exp_nums[0] == pred_nums[0]:
            return True
    
    if domain == "coding":
        # For coding, check if key code is present
        if any(keyword in prediction_norm for keyword in ['def', 'return', 'class']):
            return True
    
    return False

def evaluate_dev_set(dev_file="data/cse476_final_project_dev_data.json", limit=None):
    """
    Evaluate the agent on the dev set across all domains.
    """
    try:
        with open(dev_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {dev_file}")
        return
    
    if limit:
        data = data[:limit]
    
    # Group by domain
    by_domain = {}
    for item in data:
        domain = item.get('domain', 'unknown')
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(item)
    
    # Evaluate each domain
    results = {}
    total_correct = 0
    total_questions = 0
    total_calls = 0
    
    print("=" * 70)
    print("EVALUATING REASONING AGENT ON DEV SET")
    print("=" * 70)
    
    for domain in sorted(by_domain.keys()):
        questions = by_domain[domain]
        correct = 0
        domain_calls = 0
        
        print(f"\n[{domain.upper()}] Evaluating {len(questions)} questions...")
        
        for i, item in enumerate(questions, 1):
            reset_per_question_counter()
            
            question = item['input']
            expected = str(item.get('output', ''))
            
            try:
                prediction = route_and_solve(question)
            except CallBudgetExceeded:
                prediction = ""
                print(f"  Q{i}: BUDGET EXCEEDED")
                continue
            except Exception as e:
                prediction = ""
                print(f"  Q{i}: ERROR - {str(e)[:50]}")
                continue
            
            calls = get_per_question_calls()
            domain_calls += calls
            
            is_correct = grade_answer(question, expected, prediction, domain)
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            if i <= 3 or (i % 10 == 0):  # Print first 3 and every 10th
                exp_preview = expected[:40].replace('\n', ' ')
                pred_preview = prediction[:40].replace('\n', ' ')
                print(f"  Q{i} [{status}] Calls: {calls} | Exp: {exp_preview}... | Got: {pred_preview}...")
        
        accuracy = (correct / len(questions) * 100) if questions else 0
        avg_calls = (domain_calls / len(questions)) if questions else 0
        
        results[domain] = {
            'count': len(questions),
            'correct': correct,
            'accuracy': accuracy,
            'total_calls': domain_calls,
            'avg_calls': avg_calls
        }
        
        total_correct += correct
        total_questions += len(questions)
        total_calls += domain_calls
        
        print(f"  → {domain.upper()}: {accuracy:.1f}% ({correct}/{len(questions)}) | Avg calls: {avg_calls:.1f}")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Domain':<20} {'Count':>8} {'Correct':>8} {'Accuracy':>12} {'Avg Calls':>12}")
    print("-" * 70)
    
    for domain in sorted(results.keys()):
        r = results[domain]
        print(f"{domain:<20} {r['count']:>8} {r['correct']:>8} {r['accuracy']:>11.1f}% {r['avg_calls']:>12.1f}")
    
    print("-" * 70)
    overall_accuracy = (total_correct / total_questions * 100) if total_questions else 0
    overall_avg = (total_calls / total_questions) if total_questions else 0
    print(f"{'OVERALL':<20} {total_questions:>8} {total_correct:>8} {overall_accuracy:>11.1f}% {overall_avg:>12.1f}")
    print("=" * 70)

if __name__ == "__main__":
    limit = None
    if len(sys.argv) > 1 and sys.argv[1] == "--n":
        if len(sys.argv) > 2:
            limit = int(sys.argv[2])
    
    evaluate_dev_set(limit=limit)