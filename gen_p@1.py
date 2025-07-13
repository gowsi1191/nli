import os
import json

def compute_p_at_1(file_path):
    with open(file_path) as f:
        data = json.load(f)

    correct = 0
    total = 0

    for example_id, docs in data.items():
        if not docs:
            continue
        top_relevance = docs[0]['relevance']
        if top_relevance == 0:
            correct += 1
        total += 1

    p_at_1 = correct / total if total > 0 else 0
    return round(p_at_1, 4), correct, total

def main():
    folder = "/Users/L020774/Movies/heu/NLP"

    print("📊 P@1 Evaluation Results Per Model:")
    for filename in os.listdir(folder):
        if filename.startswith("roberta_scores_by_example_") and filename.endswith(".json"):
            full_path = os.path.join(folder, filename)
            p1, correct, total = compute_p_at_1(full_path)
            model_name = filename.replace("roberta_scores_by_example_", "").replace(".json", "")
            print(f"  {model_name}: P@1 = {p1} ({correct}/{total})")

if __name__ == "__main__":
    main()
