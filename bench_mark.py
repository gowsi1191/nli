import os
import json
import time
import torch
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util


def chunk_document_sliding(doc, tokenizer):
    tokens = tokenizer.tokenize(doc)
    total_tokens = len(tokens)
    chunk_size = 100
    overlap = 20
    step = chunk_size - overlap
    chunks = []
    for i in range(0, total_tokens, step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def chunk_document_sentence_window(doc, window=2):
    sentences = re.split(r'(?<=[.!?])\s+', doc)
    return [" ".join(sentences[i:i + window]) for i in range(len(sentences)) if sentences[i:i + window]]


class ModelOperations:
    def __init__(self, nli_model_name):
        self.model_name = nli_model_name
        self.bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device='cpu')
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to("cpu")
        self.nli_model.eval()

        if "roberta" in nli_model_name:
            self.label_order = ("entailment", "neutral", "contradiction")
        elif "deberta" in nli_model_name:
            self.label_order = ("contradiction", "neutral", "entailment")
        else:
            self.label_order = ("entailment", "neutral", "contradiction")

    def compute_best_nli_scores(self, query, doc):
        strategies = {
            "sliding": chunk_document_sliding(doc, self.nli_tokenizer),
            "sentence": chunk_document_sentence_window(doc)
        }
        best_score = -1.0
        best_scores = {label: 0.0 for label in self.label_order}
        start = time.time()

        for chunks in strategies.values():
            for chunk in chunks:
                inputs = self.nli_tokenizer(query, chunk, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    logits = self.nli_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                scores = dict(zip(self.label_order, probs))
                combined_score = scores["entailment"] + scores["neutral"]
                if combined_score > best_score:
                    best_score = combined_score
                    best_scores = scores

        elapsed = time.time() - start
        return best_scores, elapsed

    def compute_bge_score(self, query, doc):
        start = time.time()
        query_emb = self.bge_model.encode(query, convert_to_tensor=True)
        doc_emb = self.bge_model.encode(doc, convert_to_tensor=True)
        score = util.cos_sim(query_emb, doc_emb).item()
        elapsed = time.time() - start
        return score, elapsed


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "json", "data")
    other_ids = {1, 10, 11, 12, 14,2, 3, 4, 5, 6, 8, 13,16, 18, 20, 24,  15, 17, 19, 21, 22, 23}
    {
#   "3_4": [1, 10, 11, 12, 14,],
#   "others": [2, 3, 4, 5, 6, 8, 13,16, 18, 20, 24,  15, 17, 19, 21, 22, 23]
}

    examples = []
    for query_id in other_ids:
        input_path = os.path.join(base_dir, f"query{query_id}.json")
        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path}")
            continue
        with open(input_path) as f:
            data = json.load(f)
            examples.extend(data)

    models = {
        "DeBERTa-v3-base (MNLI/FEVER/ANLI)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "cross-encoder/nli-deberta-base": "cross-encoder/nli-deberta-base"
    }

    for model_name, model_id in models.items():
        output_file = f"enhanced_query_type_analysis_{model_id.replace('/', '-').replace(' ', '_')}.json"
        model_ops = ModelOperations(model_id)

        output = {"Explicit_NOT": {}}
        total_nli_time = 0
        total_bge_time = 0
        total_calls = 0

        for i, example in enumerate(examples):
            example_id = f"example_{example.get('id', i)}"
            query = example["query"]
            roberta_ranking = []
            bge_ranking = []

            for doc in example["documents"]:
                nli_scores, nli_elapsed = model_ops.compute_best_nli_scores(query, doc["text"])
                bge_score, bge_elapsed = model_ops.compute_bge_score(query, doc["text"])

                total_nli_time += nli_elapsed
                total_bge_time += bge_elapsed
                total_calls += 1

                roberta_ranking.append({
                    "doc_id": doc["doc_id"],
                    "e": nli_scores["entailment"],
                    "n": nli_scores["neutral"],
                    "c": nli_scores["contradiction"],
                    "relevance": doc["relevance"],
                    "nli_time": round(nli_elapsed, 4),
                    "bge_time": round(bge_elapsed, 4),
                    "text": doc["text"],
                    "query": query
                })

                bge_ranking.append({
                    "doc_id": doc["doc_id"],
                    "score": bge_score,
                    "relevance": doc["relevance"]
                })

            output["Explicit_NOT"][example_id] = {
                "Roberta": {"ranking": roberta_ranking},
                "BGE": {"ranking": bge_ranking}
            }

        if total_calls:
            print(f"\n✅ {model_name} Inference Summary:")
            print(f"📊 Avg NLI Inference Time per Document: {total_nli_time / total_calls:.4f} seconds")
            print(f"⚡ Avg BGE Similarity Time per Document: {total_bge_time / total_calls:.4f} seconds\n")

        out_file = f"evaluation_results_test{model_name.replace(' ', '_').replace('/', '_')}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"📁 Output saved to: {out_file}")
