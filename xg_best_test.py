import os
import json
import numpy as np
import xgboost as xgb
import pandas as pd
from statistics import mean, stdev
import matplotlib.pyplot as plt

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "xgb_nli_enc_model.json")
test_path = os.path.join(script_dir, "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json")

# === Load XGBoost model ===
clf = xgb.XGBClassifier()
clf.load_model(model_path)

# === Custom sigmoid on e ===
def custom_sigmoid_linear_e(e):
    return 1 / (1 + np.exp(-12 * (e - 0.3)))

# === Load and parse test data ===
with open(test_path, "r") as f:
    test_data = json.load(f)

X_test, y_test, query_ids, rows = [], [], [], []

for ex_id, ex in test_data["Explicit_NOT"].items():
    for doc in ex["Roberta"]["ranking"]:
        e, n, c = doc["e"], doc["n"], doc["c"]
        sig_e = custom_sigmoid_linear_e(e)

        X_test.append([e, n, c])
        y_test.append(doc["relevance"])
        query_ids.append(ex_id)

        rows.append({
            "example_id": ex_id,
            "doc_id": doc["doc_id"],
            "query": doc["query"],
            "text": doc["text"],
            "true_label": doc["relevance"]
        })

X_test = np.array(X_test)
y_test = np.array(y_test)

# === Predict probabilistic scores (prob of class 0) ===
scores = clf.predict_proba(X_test)[:, 0]

# === Add scores to rows ===
for i in range(len(rows)):
    rows[i]["score"] = float(scores[i])

# === Print dataset with scores ===
print("\n📄 Prediction Results (Sample Output):")
for row in rows:
    print(f"[{row['example_id']}] DocID: {row['doc_id']}, Label: {row['true_label']}, Score: {row['score']:.4f}")

# === Save CSV ===
df = pd.DataFrame(rows)
csv_path = os.path.join(script_dir, "xgb_nli_enc_scores.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved scores to {csv_path}")

# === Create structured JSON grouped by example_id ===
scored_data = {}
for row in rows:
    ex_id = row["example_id"]
    doc_entry = {
        "doc_id": row["doc_id"],
        "score": row["score"],
        "true_label": row["true_label"],
        "query": row["query"],
        "text": row["text"]
    }
    if ex_id not in scored_data:
        scored_data[ex_id] = []
    scored_data[ex_id].append(doc_entry)

# === Save JSON output ===
json_out_path = os.path.join(script_dir, "xgb_nli_enc_scored.json")
with open(json_out_path, "w") as f:
    json.dump(scored_data, f, indent=2)

print(f"✅ Saved scored JSON to {json_out_path}")


# # === Save scored output ===
# df = pd.DataFrame(rows)
# csv_path = os.path.join(script_dir, "xgb_nli_enc_scores.csv")
# df.to_csv(csv_path, index=False)
# print(f"✅ Saved scores to {csv_path}")

# === Compute P@1 ===
p_at_1_hits = 0
query_groups = df.groupby("example_id")

for qid, group in query_groups:
    ranked = group.sort_values(by="score", ascending=False)
    top_row = ranked.iloc[0]
    if top_row["true_label"] == 0:
        p_at_1_hits += 1

total_queries = len(query_groups)
p_at_1 = p_at_1_hits / total_queries

print(f"\n🎯 P@1: {p_at_1:.4f} ({p_at_1_hits}/{total_queries})")

# === Print full JSON of specific failed cases ===
failed_ids_to_inspect = ["example_616", "example_1001"]

print("\n📦 Full JSON for Failed P@1 Examples:")
for ex_id in failed_ids_to_inspect:
    if ex_id in scored_data:
        print(f"\n--- {ex_id} ---")
        print(json.dumps({ex_id: scored_data[ex_id]}, indent=2))
    else:
        print(f"\n⚠️ {ex_id} not found in scored data.")


# === Margin stats (top 0 - top 1 score) ===
margins = []

for qid, group in query_groups:
    ranked = group.sort_values(by="score", ascending=False)
    top_0 = ranked[ranked["true_label"] == 0]["score"].tolist()
    top_1 = ranked[ranked["true_label"] == 1]["score"].tolist()

    if top_0 and top_1:
        margin = top_0[0] - top_1[0]
        margins.append(margin)

if margins:
    margin_mean = mean(margins)
    margin_std = stdev(margins)
    cv = margin_std / margin_mean if margin_mean != 0 else 0
    generalization_score = margin_mean - margin_std

    print("\n📐 Margin Analysis:")
    print(f"✅ Margin Mean: {margin_mean:.4f}")
    print(f"✅ Margin STD:  {margin_std:.4f}")
    print(f"✅ CV (Std/Mean): {cv:.4f}")
    print(f"🎯 Generalization Score: {generalization_score:.4f}")
else:
    print("⚠️ Not enough margin pairs to compute generalization.")

from math import log2

p_at_2_hits = 0
reciprocal_ranks = []
ndcg_scores = []

for qid, group in query_groups:
    ranked = group.sort_values(by="score", ascending=False).reset_index(drop=True)
    
    # Top-2 documents
    top2 = ranked.iloc[:2]

    # ---------- P@2 ----------
    p2_rels = (top2["true_label"] == 0).sum()
    p_at_2_hits += p2_rels

    # ---------- MRR@2 ----------
    rr = 0
    for rank, (_, row) in enumerate(top2.iterrows(), start=1):
        if row["true_label"] == 0:
            rr = 1 / rank
            break
    reciprocal_ranks.append(rr)

    # ---------- nDCG@2 ----------
    def dcg(labels):
        return sum([
            (1 if rel == 0 else 0) / log2(i + 2)  # relevance 0 is ideal
            for i, rel in enumerate(labels)
        ])

    actual_labels = top2["true_label"].tolist()
    ideal_labels = sorted(actual_labels, key=lambda x: 0 if x == 0 else 1)  # ideal = relevant first

    dcg_val = dcg(actual_labels)
    idcg_val = dcg(ideal_labels)
    ndcg = dcg_val / idcg_val if idcg_val != 0 else 0
    ndcg_scores.append(ndcg)

# === Final Metrics ===
p_at_2 = p_at_2_hits / (total_queries * 2)
mrr_at_2 = sum(reciprocal_ranks) / total_queries
avg_ndcg_2 = sum(ndcg_scores) / total_queries

print("\n📊 Ranking Metrics (Top 2 Docs Only):")
print(f"🎯 P@2:     {p_at_2:.4f}")
print(f"🎯 MRR@2:   {mrr_at_2:.4f}")
print(f"🎯 nDCG@2:  {avg_ndcg_2:.4f}")
print("Total XGBoost docs:", len(rows))  # in XGBoost script
# === Scatter Plot Based on Relevance ===
scores_0 = [row["score"] for row in rows if row["true_label"] == 0]
scores_1 = [row["score"] for row in rows if row["true_label"] == 1]

plt.figure(figsize=(10, 6))
plt.scatter(range(len(scores_0)), scores_0, color="Blue",  alpha=0.7)
plt.scatter(range(len(scores_1)), scores_1, color="Orange",  alpha=0.7)
plt.axhline(margin_mean, color='gray', linestyle='--', label=f"Margin Mean = {margin_mean:.4f}")
plt.title("XGBoost Document Score Distribution by Relevance")
plt.xlabel("Document Index")
plt.ylabel("Score (Class 0 Probability)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()