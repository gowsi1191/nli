import re
from collections import defaultdict

def compute_p_at_k(predicted, target, k=3):
    top_k = predicted[:k]
    relevant = sum(1 for i in range(k) if target[i] == 0 and top_k[i] == 0)
    return relevant / k

file_path = "strategy_results.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_model = None
current_strategy = None

model_scores = defaultdict(list)
strategy_scores = defaultdict(list)

reversed_model_scores = defaultdict(list)
reversed_strategy_scores = defaultdict(list)

for i in range(len(lines)):
    line = lines[i].strip()

    if line.startswith("🧠 Evaluating Model:"):
        current_model = line.split(":")[1].strip()

    elif line.startswith("Strategy:"):
        current_strategy = line.split(":")[1].strip()

    elif line.startswith("📊 Predicted:"):
        pred = eval(line.split("Predicted:")[1].strip())
        tgt = eval(lines[i + 1].strip().split("Target:")[1].strip())

        # Original P@3
        p3 = compute_p_at_k(pred, tgt, k=3)
        model_scores[current_model].append(p3)
        strategy_scores[current_strategy].append(p3)

        # Reversed P@3
        reversed_pred = list(reversed(pred))
        p3_reversed = compute_p_at_k(reversed_pred, tgt, k=3)
        reversed_model_scores[current_model].append(p3_reversed)
        reversed_strategy_scores[current_strategy].append(p3_reversed)

# === RESULTS ===
print("\n📊 Model-wise Avg P@3 (Original):")
for model, scores in model_scores.items():
    print(f"{model:50} -> Avg P@3: {sum(scores)/len(scores):.3f}")

print("\n🔁 Model-wise Avg P@3 (Reversed):")
for model, scores in reversed_model_scores.items():
    print(f"{model:50} -> Avg P@3: {sum(scores)/len(scores):.3f}")

print("\n📊 Strategy-wise Avg P@3 (Original):")
for strat, scores in strategy_scores.items():
    print(f"{strat:20} -> Avg P@3: {sum(scores)/len(scores):.3f}")

print("\n🔁 Strategy-wise Avg P@3 (Reversed):")
for strat, scores in reversed_strategy_scores.items():
    print(f"{strat:20} -> Avg P@3: {sum(scores)/len(scores):.3f}")
