import json
import matplotlib.pyplot as plt

# === Load JSON File ===
file_path = "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json"

with open(file_path, "r") as f:
    data = json.load(f)

# === Extract BGE scores with relevance ===
rows = []
for example in data.get("Explicit_NOT", {}).values():
    if "BGE" in example:
        for doc in example["BGE"]["ranking"]:
            rows.append({
                "score": doc["score"],
                "true_label": doc["relevance"]
            })

# === Split scores by relevance ===
scores_0 = [row["score"] for row in rows if row["true_label"] == 0]
scores_1 = [row["score"] for row in rows if row["true_label"] == 1]

# === Calculate margin mean (optional horizontal line) ===
margin_mean = sum(scores_0) / len(scores_0) - sum(scores_1) / len(scores_1)

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.scatter(range(len(scores_0)), scores_0, color="Blue", label="Relevance 0", alpha=0.7)
plt.scatter(range(len(scores_1)), scores_1, color="Orange", label="Relevance 1", alpha=0.7)
plt.axhline(margin_mean, color='gray', linestyle='--', label=f"Margin Mean = {margin_mean:.4f}")

plt.title("BGE Document Score Distribution by Relevance")
plt.xlabel("Document Index")
plt.ylabel("Score (Cosine Similarity)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
