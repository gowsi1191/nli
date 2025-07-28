import json
import math

# Load the JSON file
with open("evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    data = json.load(f)

# Initialize counters
total_queries = 0
p_at_3_sum = 0.0
p_at_5_sum = 0.0
mrr_5_sum = 0.0
ndcg_3_sum = 0.0
ndcg_5_sum = 0.0

# Function for DCG calculation
def dcg_at_k(rels, k):
    return sum((1 if rel == 0 else 0) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

# Loop through each query
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue

    bge_docs = query_data["BGE"]["ranking"]
    if len(bge_docs) < 5:
        continue

    # Sort by score descending
    sorted_docs = sorted(bge_docs, key=lambda x: x["score"], reverse=True)

    # ✅ PRINT DOC ID, SCORE, RELEVANCE, RANK
    print(f"\n📌 Query ID: {query_id}")
    print("Rank\tDoc ID\t\tScore\t\tRelevance")
    for rank, doc in enumerate(sorted_docs, start=1):
        print(f"{rank}\t{doc['doc_id']}\t{doc['score']:.6f}\t{doc['relevance']}")

    rels = [doc["relevance"] for doc in sorted_docs]

    # === Precision@3 and @5 ===
    p_at_3_sum += rels[:3].count(0) / 3.0
    p_at_5_sum += rels[:5].count(0) / 5.0

    # === MRR@5 ===
    rr = 0.0
    for rank, rel in enumerate(rels[:4], start=1):
        if rel == 0:
            rr = 1 / rank
            break
    mrr_5_sum += rr

    # === nDCG@3 and @5 ===
    ideal_rels = sorted(rels[:5], key=lambda x: 0 if x == 0 else 1)
    dcg_3 = dcg_at_k(rels, 3)
    dcg_5 = dcg_at_k(rels, 5)
    idcg_3 = dcg_at_k(ideal_rels, 3)
    idcg_5 = dcg_at_k(ideal_rels, 5)

    ndcg_3 = dcg_3 / idcg_3 if idcg_3 != 0 else 0.0
    ndcg_5 = dcg_5 / idcg_5 if idcg_5 != 0 else 0.0

    ndcg_3_sum += ndcg_3
    ndcg_5_sum += ndcg_5

    total_queries += 1

# === Final Metrics ===
p_at_3 = p_at_3_sum / total_queries if total_queries else 0
p_at_5 = p_at_5_sum / total_queries if total_queries else 0
mrr_at_5 = mrr_5_sum / total_queries if total_queries else 0
ndcg_at_3 = ndcg_3_sum / total_queries if total_queries else 0
ndcg_at_5 = ndcg_5_sum / total_queries if total_queries else 0

# === Print Results ===
print(f"\n📊 Evaluation Metrics across {total_queries} queries:")
print(f"🎯 P@3:      {p_at_3:.4f}")
# print(f"🎯 P@5:      {p_at_5:.4f}")
print(f"🎯 MRR@4:    {mrr_at_5:.4f}")
print(f"🎯 nDCG@3:   {ndcg_at_3:.4f}")
# print(f"🎯 nDCG@5:   {ndcg_at_5:.4f}")
