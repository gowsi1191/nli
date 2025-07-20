import json
import math

# Load the JSON file
with open("evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    data = json.load(f)

# Initialize counters
total_queries = 0
true_p1_hits = 0
adjusted_p1_sum = 0.0
p_at_2_sum = 0.0
mrr_sum = 0.0
ndcg_sum = 0.0

# Function for DCG calculation
def dcg_at_k(rels):
    return sum((1 if rel == 0 else 0) / math.log2(i + 2) for i, rel in enumerate(rels))

# Loop through each query
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue

    bge_docs = query_data["BGE"]["ranking"]
    if len(bge_docs) < 2:
        continue

    # Sort documents by score descending
    sorted_docs = sorted(bge_docs, key=lambda x: x["score"], reverse=True)
    top2 = sorted_docs[:2]
    top2_rels = [doc["relevance"] for doc in top2]

    # === True P@1 ===
    if sorted_docs[0]["relevance"] == 0:
        true_p1_hits += 1

    # === Adjusted P@1 ===
    if top2_rels == [0, 0]:
        adjusted_p1_sum += 1.0
    elif 0 in top2_rels and 1 in top2_rels:
        adjusted_p1_sum += 0.5
    else:
        adjusted_p1_sum += 0.0

    # === P@2 ===
    p_at_2_sum += top2_rels.count(0) / 2.0

    # === MRR@2 ===
    rr = 0.0
    for rank, rel in enumerate(top2_rels, start=1):
        if rel == 0:
            rr = 1 / rank
            break
    mrr_sum += rr

    # === nDCG@2 ===
    dcg = dcg_at_k(top2_rels)
    ideal_rels = sorted(top2_rels, key=lambda x: 0 if x == 0 else 1)
    idcg = dcg_at_k(ideal_rels)
    ndcg = dcg / idcg if idcg != 0 else 0.0
    ndcg_sum += ndcg

    total_queries += 1

# === Final metrics ===
true_p1 = true_p1_hits / total_queries if total_queries else 0
adjusted_p1 = adjusted_p1_sum / total_queries if total_queries else 0
p_at_2 = p_at_2_sum / total_queries if total_queries else 0
mrr_at_2 = mrr_sum / total_queries if total_queries else 0
ndcg_at_2 = ndcg_sum / total_queries if total_queries else 0

# === Print results ===
print(f"\n📊 Evaluation Metrics across {total_queries} queries:")
print(f"🎯 True P@1 (Top-1 only):      {true_p1:.4f}")
print(f"🎯 Adjusted P@1 (2-doc logic): {adjusted_p1:.4f}")
print(f"🎯 P@2:                        {p_at_2:.4f}")
print(f"🎯 MRR@2:                      {mrr_at_2:.4f}")
print(f"🎯 nDCG@2:                     {ndcg_at_2:.4f}")
