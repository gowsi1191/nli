import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

models = {
    "DeBERTa-v3-base (MNLI/FEVER/ANLI)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-deberta-base": "cross-encoder/nli-deberta-base"
}

def chunk_document(doc, tokenizer, strategy):
    tokens = tokenizer.tokenize(doc)
    if strategy == "sentence_window_2":
        sentences = re.split(r'(?<=[.!?])\s+', doc)
        return [" ".join(sentences[i:i+2]) for i in range(len(sentences)) if sentences[i:i+2]]
    chunk_size, overlap = 100, 20
    step = chunk_size - overlap
    return [tokenizer.convert_tokens_to_string(tokens[i:i+chunk_size]) for i in range(0, len(tokens), step)]

class ModelOperations:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        if "roberta" in model_name:
            self.labels = ("entailment", "neutral", "contradiction")
        elif "deberta" in model_name:
            self.labels = ("contradiction", "neutral", "entailment")
        else:
            self.labels = ("entailment", "neutral", "contradiction")

    def compute_best_score(self, query, doc):
        best = {"score": -1.0, "result": {l: 0.0 for l in self.labels}}
        for strategy in ["sliding", "sentence_window_2"]:
            chunks = chunk_document(doc, self.tokenizer, strategy)
            for chunk in chunks:
                inputs = self.tokenizer(query, chunk, return_tensors="pt", truncation=True, max_length=512)
                input_len = inputs["input_ids"].shape[-1]
                if input_len > 512:
                    continue
                with torch.no_grad():
                    probs = torch.softmax(self.model(**inputs).logits, dim=-1).squeeze().tolist()
                scores = dict(zip(self.labels, probs))
                combined = scores["entailment"] + scores["neutral"]
                if combined > best["score"]:
                    best = {"score": combined, "result": scores}
        return best["result"]

def evaluate_p_at_k(docs, query, model_ops, k=3):
    scored = [
        {
            "doc_id": d["doc_id"],
            "e": model_ops.compute_best_score(query, d["text"])["entailment"],
            "relevance": d["relevance"]
        }
        for d in docs
    ]
    ranked = sorted(scored, key=lambda x: x["e"], reverse=True)
    return sum(1 for d in ranked[:k] if d["relevance"] == 0) / k

if __name__ == "__main__":
    base = "json/data"
    files = [
        f for f in os.listdir(base)
        if f.startswith("query") and f.endswith(".json") and re.search(r'\d+', f) and int(re.search(r'\d+', f).group()) <= 10
    ]
    with open("strategy_results.txt", "w") as out:
        for name, model_id in models.items():
            ops = ModelOperations(model_id)
            out.write(f"\nModel: {name}\n")
            for f in sorted(files):
                data = json.load(open(os.path.join(base, f)))[0]
                score = evaluate_p_at_k(data["documents"], data["query"], ops, k=3)
                out.write(f"{f}: P@3 = {score:.2f}\n")
