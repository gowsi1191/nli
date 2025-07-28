import os
import json
import time
import torch
import math
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

models = {
    "DeBERTa-v3-base (MNLI/FEVER/ANLI)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-deberta-base": "cross-encoder/nli-deberta-base"
}

def chunk_document(doc, tokenizer, strategy="sliding", separator=None):
    tokens = tokenizer.tokenize(doc)
    total_tokens = len(tokens)

    if strategy.startswith("sentence_window_"):
        window_size = int(strategy.split("_")[-1])
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        chunks = []
        for i in range(len(sentences)):
            chunk = " ".join(sentences[i:i + window_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    # sliding window fallback
    overlap =15
    chunk_size = 100
    step = chunk_size - overlap

    if total_tokens <= chunk_size:
        return [doc]

    chunks = []
    for i in range(0, total_tokens, step):
        chunk_tokens = tokens[i:i + chunk_size]
        if not chunk_tokens:
            continue
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        if chunk_text:
            chunks.append(chunk_text)

    return chunks

class ModelOperations:
    def __init__(self, nli_model_name):
        self.model_name = nli_model_name
        self.bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
        self.nli_model.eval()

        if "roberta" in nli_model_name:
            self.label_order = ("entailment", "neutral", "contradiction")
        elif "deberta" in nli_model_name:
            self.label_order = ("contradiction", "neutral", "entailment")
        else:
            self.label_order = ("entailment", "neutral", "contradiction")

    def compute_nli_scores(self, query, doc, separator=None):
        strategies = ["sliding", "sentence_window_2"]
        best_combined_score = -1.0
        best_scores = None

        for strategy in strategies:
            chunks = chunk_document(doc, self.nli_tokenizer, strategy=strategy, separator=separator)
            for chunk in chunks:
                inputs = self.nli_tokenizer(query, chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                scores = dict(zip(self.label_order, probs))
                combined_score = scores["entailment"] + scores["neutral"]
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_scores = scores

        if best_scores is None:
            best_scores = {label: 0.0 for label in self.label_order}
        return best_scores


def evaluate_p_at_k(documents, query, model_ops, strategy, k=3, separator=None):
    rankings = []
    for doc in documents:
        scores = model_ops.compute_nli_scores(query, doc["text"], strategy=strategy, separator=separator)
        rankings.append({"doc_id": doc["doc_id"], "e": scores["entailment"], "relevance": doc["relevance"]})

    sorted_docs = sorted(rankings, key=lambda x: x["e"], reverse=True)
    predicted = [doc["relevance"] for doc in sorted_docs]
    target = [doc["relevance"] for doc in documents]
    top_k = sorted_docs[:k]
    correct = sum(1 for doc in top_k if doc["relevance"] == 0)
    match = predicted == target
    return correct / k, match, predicted, target

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json", "data")
    query_files = [f for f in os.listdir(base_dir) if f.startswith("query") and f.endswith(".json") and int(re.search(r'\d+', f).group()) <= 5]
    
    strategies = ["sliding"] + [f"sentence_window_{w}" for w in [2]]

    for model_name, model_id in models.items():
        model_ops = ModelOperations(model_id)
        output_lines = [f"\n🧠 Evaluating Model: {model_name}\n"]

        for strat in strategies:
            output_lines.append(f"\nStrategy: {strat}")
            p2_scores, p3_scores = [], []

            for qfile in sorted(query_files):
                qpath = os.path.join(base_dir, qfile)
                with open(qpath) as f:
                    data = json.load(f)[0]

                query = data["query"]
                documents = data["documents"]

                p2, _, _, _ = evaluate_p_at_k(documents, query, model_ops, strategy=strat, k=2)
                p3, match, predicted, target = evaluate_p_at_k(documents, query, model_ops, strategy=strat, k=3)

                p2_scores.append(p2)
                p3_scores.append(p3)

                output_lines.append(f"📁 {qfile} | P@2: {p2:.2f} | P@3: {p3:.2f} | ✅ Match: {match}")
                output_lines.append(f"📊 Predicted: {predicted}")
                output_lines.append(f"🎯 Target:    {target}\n")

            mean_p2 = sum(p2_scores) / len(p2_scores)
            mean_p3 = sum(p3_scores) / len(p3_scores)
            output_lines.append(f"📈 Mean P@2 for {strat}: {mean_p2:.3f}")
            output_lines.append(f"📈 Mean P@3 for {strat}: {mean_p3:.3f}\n")

        # Append results for this model to the file
        with open("strategy_results.txt", "a") as f:
            f.write("\n".join(output_lines))
            f.write("\n" + "="*80 + "\n")
