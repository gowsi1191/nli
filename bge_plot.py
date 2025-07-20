import json
import math
import matplotlib.pyplot as plt
from statistics import mean, stdev

# === Load your local JSON file ===
with open("evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    data = json.load(f)

# === Initialize lists for relevance scores ===
scores_0 = []
scores_1 = []
margins = []

# === Loop through queries and collect scores ===
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue

    docs = query_data["BGE"]["ranking"]
    if not docs:
        continue

    # Sort by descending score
    sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)

    # Get highest score for rel=0 and rel=1
    top_score_0 = next((doc["score"] for doc in sorted_docs if doc["relevance"] == 0), None)
    top_score_1 = next((doc["score"] for doc in sorted_docs if doc["relevance"] == 1), None)

    if top_score_0 is not None:
        scores_0.append(top_score_0)
    if top_score_1 is not None:
        scores_1.append(top_score_1)
    if top_score_0 is not None and top_score_1 is not None:
        margins.append(top_score_0 - top_score_1)

# === Compute Margin Metrics ===
margin_mean = mean(margins) if margins else 0
margin_std = stdev(margins) if len(margins) > 1 else 0
cv = margin_std / margin_mean if margin_mean != 0 else 0
generalization_score = margin_mean - margin_std

# === Plot Scatter Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(range(len(scores_0)), scores_0, color="green", label="Relevance 0", s=50)
plt.scatter(range(len(scores_1)), scores_1, color="red", label="Relevance 1", s=50)
plt.axhline(margin_mean, color='gray', linestyle='--', label=f"Margin Mean = {margin_mean:.4f}")
plt.title("Score Distribution by Relevance")
plt.xlabel("Query Index")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Print Margin Stats ===
print("\n📐 Margin Analysis:")
print(f"✅ Margin Mean: {margin_mean:.4f}")
print(f"✅ Margin STD:  {margin_std:.4f}")
print(f"✅ CV (Std/Mean): {cv:.4f}")
print(f"🎯 Generalization Score: {generalization_score:.4f}")
