import json

# Load the JSON file
with open("evaluation_results_BERT-base_(MNLI).json", "r") as f:
    data = json.load(f)

# Initialize count
total_queries = 0
correct_p1 = 0

# Loop through each query
for query_id, query_data in data.get("Explicit_NOT", {}).items():
    if "BGE" not in query_data:
        continue  # skip if BGE not available

    bge_docs = query_data["BGE"]["ranking"]
    if not bge_docs:
        continue  # skip if no documents

    # Sort documents by score descending
    sorted_docs = sorted(bge_docs, key=lambda x: x["score"], reverse=True)

    # Check if top document is relevance 0
    p1 = 1 if sorted_docs[0]["relevance"] == 0 else 0
    correct_p1 += p1
    total_queries += 1

# Compute final P@1
p_at_1 = correct_p1 / total_queries if total_queries > 0 else 0

print(f"P@1 for BGE across {total_queries} queries: {p_at_1:.4f}")
