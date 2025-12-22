import json
import time
from collections import defaultdict
from app import generate_response 

# Config
TEST_JSON = "test_queries.json" 
MODES = ["trivia", "summary", "fanfiction"]  
USE_LITE = True
TEMPERATURE = 0.7
MAX_TOKENS = 512

# Evaluation Functions
def evaluate_overlap(model_output, expected_text, min_words=3):
# Check if at least min_words from expected_text appear in output.
    expected_words = set(expected_text.lower().split())
    output_words = set(model_output.lower().split())
    overlap = expected_words & output_words
    return len(overlap) >= min_words

def evaluate_fraction(model_output, expected_text, fraction=0.3):
# Check if at least a fraction of words from expected_text appear in output.
    expected_words = set(expected_text.lower().split())
    output_words = set(model_output.lower().split())
    overlap = expected_words & output_words
    return len(overlap) / len(expected_words) >= fraction

def evaluate_hybrid(model_output, expected_text):
    words = expected_text.split()
    if len(words) <= 5:
        return evaluate_overlap(model_output, expected_text, min_words=2)
    else:
        return evaluate_fraction(model_output, expected_text, fraction=0.3)

# Load test set
with open(TEST_JSON, "r", encoding="utf-8") as f:
    test_set = json.load(f)

# Metrics storage
metrics = defaultdict(lambda: {"total": 0, "correct": 0, "response_times": [], "wrong_examples": []})

# main
for item in test_set:
    query = item["query"]
    expected = item["expected"]
    mode = item.get("mode", "summary")  # fallback to summary

    metrics[mode]["total"] += 1
    start_time = time.time()
    output = generate_response(query, use_lite=USE_LITE, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, mode=mode)
    end_time = time.time()
    metrics[mode]["response_times"].append(end_time - start_time)

    correct = evaluate_hybrid(output, expected)
    if correct:
        metrics[mode]["correct"] += 1
    else:
        metrics[mode]["wrong_examples"].append({
            "query": query,
            "expected": expected,
            "output": output
        })

# Report
print("Evaluation Report \n")
for mode in MODES:
    total = metrics[mode]["total"]
    correct = metrics[mode]["correct"]
    avg_time = sum(metrics[mode]["response_times"]) / max(total, 1)
    accuracy = (correct / total * 100) if total > 0 else 0

    print("Mode:", mode)
    print("Total queries:", total)
    print("Total correct:", correct)
    print("Accuracy:", f"{accuracy:.2f}%")
    print("Average response time:", f"{avg_time:.2f} seconds")
    print(f"  Wrong examples ({len(metrics[mode]['wrong_examples'])}):")
    for ex in metrics[mode]["wrong_examples"][:3]:  # show up to 3
        print("Query:", ex["query"])
        print("Expected:", ex["expected"])
        if len(ex["output"]) > 100:
            print("Output:", ex["output"][:100] + "...")
        else:
            print("Output:", ex["output"])
    print("\n")
