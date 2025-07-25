import os
import json
import time
import torch
import numpy as np
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === CONFIGURATION ===
nli_model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
xgb_model_path = "xgb_nli_enc_model.json"
input_json_path = os.path.join(os.path.dirname(__file__), "json", "data", "performance.json")
label_order = ("contradiction", "neutral", "entailment")

# === DEVICE SETUP ===
if torch.backends.mps.is_available():
    device = torch.device("mps")
    backend = "Apple MPS (Metal Performance Shaders)"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    backend = "CUDA (NVIDIA GPU)"
else:
    device = torch.device("cpu")
    backend = "CPU (No GPU/MPS available)"

print(f"\n🧠 Loading DeBERTa model on device: {device} — {backend}")
tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
model.eval()

# === MAIN PIPELINE ===
def main():
    with open(input_json_path) as f:
        data = json.load(f)

    all_rows = []
    all_queries = []
    all_docs = []

    print("\n📦 Preparing document pairs...")
    for ex_id, item in data.items():
        query = item["query"]
        for doc in item["documents"]:
            all_queries.append(query)
            all_docs.append(doc["text"])
            all_rows.append({
                "example_id": ex_id,
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "true_label": doc["relevance"],
                "query": query
            })

    total_docs = len(all_docs)
    print(f"🚀 Running batched inference on {total_docs} document pairs...")
    # === Token Stats ===
    print("🧮 Calculating token statistics...")

    # Tokenize each pair individually for token count stats
    token_counts = []
    for q, d in zip(all_queries, all_docs):
        tokens = tokenizer.encode(q, d, truncation=False)
        token_counts.append(len(tokens))

    total_tokens = sum(token_counts)
    avg_tokens = round(total_tokens / len(token_counts), 2)
    max_tokens = max(token_counts)

    print(f"🧾 Total Tokens:   {total_tokens}")
    print(f"📊 Avg Tokens/Pair: {avg_tokens}")
    print(f"📏 Max Tokens/Pair: {max_tokens}")

    start = time.time()
    

    # === Batch Inference ===
    inputs = tokenizer(all_queries, all_docs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # === Extract (entailment, neutral, contradiction) scores ===
    label_idx = {label: i for i, label in enumerate(label_order)}
    nli_scores = [(row[label_idx["entailment"]], row[label_idx["neutral"]], row[label_idx["contradiction"]]) for row in probs]

    # === Load XGBoost and Predict ===
    print("⚖️ Running XGBoost reranking...")
    clf = xgb.XGBClassifier()
    clf.load_model(xgb_model_path)
    scores = clf.predict_proba(np.array(nli_scores))[:, 0]  # Class 0 = relevant

    for i in range(total_docs):
        all_rows[i]["score"] = float(scores[i])

    # === Group & Rerank ===
    reranked = {}
    for row in all_rows:
        reranked.setdefault(row["example_id"], []).append(row)

    for ex_id in reranked:
        reranked[ex_id].sort(key=lambda x: x["score"], reverse=True)

    elapsed = time.time() - start
    avg_time = round(1000 * elapsed / total_docs, 2)

    # === Final Metrics Summary ===
    print("\n📊 Performance Summary:")
    print(f"🧠 NLI Model Used: {nli_model_name}")
    print(f"📦 XGBoost Model:  {xgb_model_path}")
    print(f"🖥️ Device Used:    {backend}")
    print(f"📄 Total Docs:     {total_docs}")
    print(f"⏱️ Total Time:     {elapsed:.2f} seconds")
    print(f"⚡ Avg Time/Doc:   {avg_time} ms")

    # === Print Top Results for Each Query ===
    for ex_id, docs in reranked.items():
        print(f"\n🔍 Query: {data[ex_id]['query']}\n")
        for rank, doc in enumerate(docs[:10], start=1):
            print(f"{rank:2d}. DocID: {doc['doc_id']} | Score: {doc['score']:.4f} | Label: {doc['true_label']}")
            print(f"    ➤ {doc['text']}\n")



if __name__ == "__main__":
    main()
