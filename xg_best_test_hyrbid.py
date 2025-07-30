import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from statistics import mean
from math import log2

# === Configs ===
nli_dir = os.getcwd()
output_dir = os.path.join(nli_dir, "reranked_outputs")
os.makedirs(output_dir, exist_ok=True)

# === Custom model path for combined features ===
model_filename = "xgb_model_DeBERTa+BGE+CrossEncoder.json"
model_path = os.path.join(nli_dir, model_filename)

# === Evaluation function ===
def evaluate(df):
    query_groups = df.groupby("example_id")
    total_queries = len(query_groups)

    p_at_k_list = []
    rr_list = []
    ndcg_list = []

    for _, group in query_groups:
        ranked = group.sort_values(by="score", ascending=False).reset_index(drop=True)
        k = sum(group["true_label"] == 0)
        if k == 0:
            continue

        top_k = ranked.iloc[:k]
        correct_hits = sum(top_k["true_label"] == 0)
        p_at_k = correct_hits / k
        p_at_k_list.append(p_at_k)

        rr = 0
        for rank, (_, row) in enumerate(ranked.iloc[:k].iterrows(), start=1):
            if row["true_label"] == 0:
                rr = 1 / rank
                break
        rr_list.append(rr)

        def dcg(labels):
            return sum([(1 if rel == 0 else 0) / log2(i + 2) for i, rel in enumerate(labels)])

        actual = ranked.iloc[:k]["true_label"].tolist()
        ideal = sorted(actual, key=lambda x: 0 if x == 0 else 1)
        dcg_val = dcg(actual)
        idcg_val = dcg(ideal)
        ndcg = dcg_val / idcg_val if idcg_val != 0 else 0
        ndcg_list.append(ndcg)

    return {
        "P@K (Dynamic)": round(mean(p_at_k_list), 4) if p_at_k_list else 0.0,
        "MRR@K (Dynamic)": round(mean(rr_list), 4) if rr_list else 0.0,
        "nDCG@K (Dynamic)": round(mean(ndcg_list), 4) if ndcg_list else 0.0
    }

# === Load evaluation input ===
combined_input_file = "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json"
cross_input_file = "evaluation_results_testcross-encoder_nli-deberta-base.json"

with open(os.path.join(nli_dir, combined_input_file)) as f1, open(os.path.join(nli_dir, cross_input_file)) as f2:
    deberta_data = json.load(f1)["Explicit_NOT"]
    cross_data = json.load(f2)["Explicit_NOT"]

# === Test IDs ===
test_ids = {6, 12, 13, 14, 15, 32, 35, 36, 38, 39, 40, 41}

X_test, rows = [], []

# === Feature extraction ===
for ex_id, ex in deberta_data.items():
    example_num = int(ex_id.split("_")[1])
    if example_num not in test_ids or "Roberta" not in ex or "BGE" not in ex:
        continue

    cross_docs = cross_data.get(ex_id, {}).get("Roberta", {}).get("ranking", [])
    cross_map = {d["doc_id"]: d for d in cross_docs}
    bge_map = {d["doc_id"]: d for d in ex["BGE"]["ranking"]}

    for doc in ex["Roberta"]["ranking"]:
        doc_id = doc["doc_id"]
        if doc_id not in cross_map or doc_id not in bge_map:
            continue

        features = [
            doc["e"], doc["n"], doc["c"],                  # DeBERTa
            bge_map[doc_id]["score"],                      # BGE
            cross_map[doc_id]["e"], cross_map[doc_id]["n"], cross_map[doc_id]["c"]  # Cross-Encoder
        ]

        X_test.append(features)
        rows.append({
            "example_id": ex_id,
            "doc_id": doc_id,
            "query": doc["query"],
            "text": doc["text"],
            "true_label": doc["relevance"]
        })

# === Load model and predict ===
clf = xgb.XGBClassifier()
clf.load_model(model_path)

X_test = np.array(X_test)
scores = clf.predict_proba(X_test)[:, 0]

for i in range(len(rows)):
    rows[i]["score"] = float(scores[i])

df = pd.DataFrame(rows)
metrics = evaluate(df)

print(f"\n📦 Using model: {model_filename}")
for metric, value in metrics.items():
    print(f"✅ {metric}: {value:.4f}")

# === Save reranked output ===
reranked = {}
for row in rows:
    ex_id = row["example_id"]
    if ex_id not in reranked:
        reranked[ex_id] = []
    reranked[ex_id].append({
        "doc_id": row["doc_id"],
        "score": row["score"],
        "true_label": row["true_label"],
        "query": row["query"],
        "text": row["text"]
    })

outname = "reranked_test_Combined_XGBoost.json"
outpath = os.path.join(output_dir, outname)
with open(outpath, "w") as f:
    json.dump(reranked, f, indent=2)

print(f"💾 Saved reranked output: {outpath}")
