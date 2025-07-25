import json
import matplotlib.pyplot as plt
from statistics import mean, stdev

# === Load the file ===
with open("evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    data = json.load(f)

# === Accumulators ===
all_scores_0 = []
all_scores_1 = []
margins = []
doc_idx = 0  # to assign unique index for scatter plot

# === Collect all document scores ===
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue

    docs = query_data["BGE"]["ranking"]
    scores_0 = [doc["score"] for doc in docs if doc["relevance"] == 0]
    scores_1 = [doc["score"] for doc in docs if doc["relevance"] == 1]

    all_scores_0.extend(scores_0)
    all_scores_1.extend(scores_1)

    # For margin (only if at least one of each)
    if scores_0 and scores_1:
        margins.append(mean(scores_0) - mean(scores_1))

# === Compute margin stats ===
margin_mean = mean(margins) if margins else 0
margin_std = stdev(margins) if len(margins) > 1 else 0
cv = margin_std / margin_mean if margin_mean else 0
generalization_score = margin_mean - margin_std

# === Scatter Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(range(len(all_scores_0)), all_scores_0, color="blue", label="Relevance 0", alpha=0.7)
plt.scatter(range(len(all_scores_1)), all_scores_1, color="orange", label="Relevance 1", alpha=0.7)
plt.axhline(margin_mean, color='gray', linestyle='--', label=f"Margin Mean = {margin_mean:.4f}")
plt.title("BGE Document Score Distribution by Relevance (All Documents)")
plt.xlabel("Document Index")
plt.ylabel("Cosine Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Stats Summary ===
print(f"🔹 Total Queries in Dataset: {len(data.get('Explicit_NOT', {}))}")
print(f"🔹 Total Relevance 0 Docs:   {len(all_scores_0)}")
print(f"🔹 Total Relevance 1 Docs:   {len(all_scores_1)}")
print(f"🔹 Total Docs Plotted:       {len(all_scores_0) + len(all_scores_1)}")
print(f"🔹 Margin Pairs Count:       {len(margins)}")

print("\n📐 Margin Analysis:")
print(f"✅ Margin Mean:           {margin_mean:.4f}")
print(f"✅ Margin STD:            {margin_std:.4f}")
print(f"✅ CV (Std/Mean):         {cv:.4f}")
print(f"🎯 Generalization Score:  {generalization_score:.4f}")
