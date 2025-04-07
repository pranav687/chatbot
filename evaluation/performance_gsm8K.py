import json
import requests
import re
import matplotlib.pyplot as plt
from datasets import load_dataset


API_URL = "http://localhost:8001/chat"  

dataset = load_dataset("gsm8k", "main", split="test[:50]")  # Use a small subset for speed

def normalize_answer(ans):
    """Clean and extract numeric part of answer."""
    return re.sub(r"[^\d\.\-]", "", str(ans).strip())

def extract_final_number(response_json):
    """
    Extract the final numeric value from a chatbot JSON response.
    
    The function handles responses as either str or bytes.
    It first checks if an "=" sign is present and extracts the first number
    from the text after the last "=". Otherwise, it returns the last number found.
    
    Parameters:
        response_json (str or bytes): The JSON string/bytes containing the answer.
        
    Returns:
        float or None: The extracted number as a float, or None if no number is found.
    """
     # Find all numbers (integers or decimals), allowing commas and $
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", response_json)

    if matches:
        last = matches[-1].replace(",", "").replace("$", "")
        try:
            return float(last)
        except ValueError:
            return None

    return None

def is_correct(predicted, actual):
    """Compare numerical answers."""
    try:
        return float(normalize_answer(predicted)) == float(normalize_answer(actual))
    except ValueError:
        return False

def get_chatbot_response(question):
    """Send question to chatbot FastAPI."""
    try:
        response = requests.post(API_URL, json={"query": question,"conversation_id":"benchmark_gsm8K"})
        response.raise_for_status()
        return response.json().get("answer", "")
    except Exception as e:
        print(f"API error: {e}")
        return ""

# Step 3: Evaluate chatbot
correct = 0
results = []

for i, item in enumerate(dataset):
    question = item['question']
    # Extract actual answer from the "answer" string (GSM8K format: "...#### 42")
    actual_answer_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", item['answer'])
    actual_answer = actual_answer_match.group(1) if actual_answer_match else ""

    chatbot_answer = get_chatbot_response(question)
    predicted_answer = extract_final_number(chatbot_answer)
    result = is_correct(predicted_answer, actual_answer)
    
    print(f"\nQ{i+1}: {question.strip()}")
    print(f"Chatbot Answer: {predicted_answer}")
    print(f"Correct Answer: {actual_answer} --> {'✅' if result else '❌'}")

    results.append(1 if result else 0)
    correct += result

# Step 4: Print Solve Rate
total = len(dataset)
solve_rate = (correct / total) * 100
print(f"\nSolve rate: {solve_rate:.2f}%")

# Step 5: Plot Results
plt.figure(figsize=(10, 5))
plt.plot(range(1, total + 1), results, marker='o', linestyle='', color='green')
plt.axhline(y=0.5, color='orange', linestyle='--', label="50% line")
plt.title("Chatbot Solve Accuracy on GSM8K Math Questions")
plt.xlabel("Question Number")
plt.ylabel("Correct (1) / Incorrect (0)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
