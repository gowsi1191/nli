import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from statistics import mean
from math import log2

# === Configs ===
nli_dir = os.getcwd()  # assume current directory is /nli
output_dir = os.path.join(nli_dir, "reranked_outputs")
os.makedirs(output_dir, exist_ok=True)

# === Model mapping based on test file name ===
model_map = {
    "testDeBERTa-v3-base_(MNLI_FEVER_ANLI)": "xgb_model_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json",
    "testcross-encoder_nli-deberta-base": "xgb_model_cross-encoder_nli-deberta-base.json"
}


def evaluate(df):
    query_groups = df.groupby("example_id")
    total_queries = len(query_groups)

    p_at_k_list = []
    rr_list = []
    ndcg_list = []

    for _, group in query_groups:
        ranked = group.sort_values(by="score", ascending=False).reset_index(drop=True)
        
        # Dynamically determine K = # of relevant docs
        k = sum(group["true_label"] == 0)

        if k == 0:
            continue  # skip queries with no relevant docs

        # Compute P@K
        top_k = ranked.iloc[:k]
        correct_hits = sum(top_k["true_label"] == 0)
        p_at_k = correct_hits / k
        p_at_k_list.append(p_at_k)

        # MRR@K (first correct hit in top K)
        rr = 0
        for rank, (_, row) in enumerate(ranked.iloc[:k].iterrows(), start=1):
            if row["true_label"] == 0:
                rr = 1 / rank
                break
        rr_list.append(rr)

        # nDCG@K
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


# === Process each NLI test file ===
for filename in os.listdir(nli_dir):
    if not filename.endswith(".json") or not filename.startswith("evaluation_results_test"):
        continue

    filepath = os.path.join(nli_dir, filename)
    model_key = filename.replace("evaluation_results_", "").replace(".json", "")
    model_filename = model_map.get(model_key)

    if model_filename is None:
        print(f"⚠️ No matching model found for: {filename}")
        continue

    model_path = os.path.join(nli_dir, model_filename)
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    print(f"\n📂 Processing: {filename}")
    print(f"📦 Using model: {model_filename}")

    with open(filepath, "r") as f:
        test_data = json.load(f)

    X_test, rows = [], []

    for ex_id, ex in test_data.get("Explicit_NOT", {}).items():
        if "Roberta" not in ex:
            continue
        example_num = int(ex_id.split("_")[1])
        if example_num not in test_ids:
            continue
        for doc in ex["Roberta"]["ranking"]:
            e, n, c = doc["e"], doc["n"], doc["c"]

            X_test.append([e, n, c])
            rows.append({
                "example_id": ex_id,
                "doc_id": doc["doc_id"],
                "query": doc["query"],
                "text": doc["text"],
                "true_label": doc["relevance"]
            })

    if not rows:
        print("⚠️ No valid entries, skipping...")
        continue

    X_test = np.array(X_test)
    scores = clf.predict_proba(X_test)[:, 0]

    for i in range(len(rows)):
        rows[i]["score"] = float(scores[i])

    df = pd.DataFrame(rows)
    metrics = evaluate(df)
    for metric, value in metrics.items():
        print(f"✅ {metric}: {value:.4f}")

    # === Save reranked JSON ===
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

    outname = filename.replace("evaluation_results_", "reranked_")
    outpath = os.path.join(output_dir, outname)

    with open(outpath, "w") as f:
        json.dump(reranked, f, indent=2)

    print(f"💾 Saved reranked output: {outpath}")
