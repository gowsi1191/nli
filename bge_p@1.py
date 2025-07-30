import json
import math

# === Load JSON ===
with open("evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    data = json.load(f)

# === Set of test IDs to include ===
test_ids = {6, 12, 13, 14, 15, 32, 35, 36, 38, 39, 40, 41}

# === Metric accumulators ===
total_queries = 0
p_at_k_sum = 0.0
mrr_sum = 0.0
ndcg_sum = 0.0

# === DCG Helper ===
def dcg_at_k(rels, k):
    return sum((1 if rel == 0 else 0) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))

# === Loop through JSON ===
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue

    # Extract example number from ID string (e.g., "example_3" -> 3)
    query_num = int(query_id.split("_")[1])
    if query_num not in test_ids:
        continue

    bge_docs = query_data["BGE"]["ranking"]
    if len(bge_docs) == 0:
        continue

    # Sort by BGE score
    sorted_docs = sorted(bge_docs, key=lambda x: x["score"], reverse=True)
    rels = [doc["relevance"] for doc in sorted_docs]

    # Compute K = number of relevant documents
    k = sum(1 for rel in rels if rel == 0)
    if k == 0:
        continue

    # P@K
    top_k = rels[:k]
    p_at_k = top_k.count(0) / k
    p_at_k_sum += p_at_k

    # MRR@K
    rr = 0.0
    for rank, rel in enumerate(rels[:k], start=1):
        if rel == 0:
            rr = 1.0 / rank
            break
    mrr_sum += rr

    # nDCG@K
    dcg = dcg_at_k(rels, k)
    ideal_rels = sorted(rels[:k], key=lambda r: 0 if r == 0 else 1)  # Relevant first
    idcg = dcg_at_k(ideal_rels, k)
    ndcg = dcg / idcg if idcg > 0 else 0.0
    ndcg_sum += ndcg

    total_queries += 1

    # === Output this query's summary ===
    print(f"\n📌 Query ID: {query_id} (K={k})")
    print("Rank\tDoc ID\t\tScore\t\tRelevance")
    for rank, doc in enumerate(sorted_docs, start=1):
        print(f"{rank}\t{doc['doc_id']}\t{doc['score']:.6f}\t{doc['relevance']}")
    print(f"🎯 BGE P@{k}: {p_at_k:.4f}")
    print(f"🔁 MRR@{k}: {rr:.4f}")
    print(f"📈 nDCG@{k}: {ndcg:.4f}")

# === Final Aggregated Results ===
if total_queries > 0:
    final_p_at_k = p_at_k_sum / total_queries
    final_mrr = mrr_sum / total_queries
    final_ndcg = ndcg_sum / total_queries
else:
    final_p_at_k = final_mrr = final_ndcg = 0.0

print(f"\n📊 Final Metrics across {total_queries} queries:")
print(f"✅ Mean P@K:   {final_p_at_k:.4f}")
print(f"✅ Mean MRR@K: {final_mrr:.4f}")
print(f"✅ Mean nDCG@K:{final_ndcg:.4f}")
