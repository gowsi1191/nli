import os
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util


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

    def compute_nli_scores(self, query, doc):
        start = time.time()
        inputs = self.nli_tokenizer(query, doc, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        elapsed = time.time() - start
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        return dict(zip(self.label_order, probs)), elapsed

    def compute_bge_score(self, query, doc):
        start = time.time()
        query_emb = self.bge_model.encode(query, convert_to_tensor=True)
        doc_emb = self.bge_model.encode(doc, convert_to_tensor=True)
        score = util.cos_sim(query_emb, doc_emb).item()
        elapsed = time.time() - start
        return score, elapsed


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "json", "data", "1.json")

    with open(input_path) as f:
        examples = json.load(f)

    models = {
        "DeBERTa-v3-base (MNLI/FEVER/ANLI)": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        #      "BERT-base (MNLI)": "textattack/bert-base-uncased-MNLI",
        # "roberta-large-mnli": "roberta-large-mnli",
        # "facebook/bart-large-mnli": "facebook/bart-large-mnli",
        # "microsoft/deberta-large-mnli": "microsoft/deberta-large-mnli",
        # "prajjwal1/albert-base-v2-mnli": "prajjwal1/albert-base-v2-mnli",
        # "pritamdeka/PubMedBERT-MNLI-MedNLI": "pritamdeka/PubMedBERT-MNLI-MedNLI",
        # "typeform/distilbert-base-uncased-mnli": "typeform/distilbert-base-uncased-mnli",
        # "cross-encoder/nli-deberta-base": "cross-encoder/nli-deberta-base"
   
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
                nli_scores, nli_elapsed = model_ops.compute_nli_scores(query, doc["text"])
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

        # Print average times
        if total_calls:
            print(f"\n✅ {model_name} Inference Summary:")
            print(f"📊 Avg NLI Inference Time per Document: {total_nli_time / total_calls:.4f} seconds")
            print(f"⚡ Avg BGE Similarity Time per Document: {total_bge_time / total_calls:.4f} seconds\n")

        # Save JSON result
        out_file = f"evaluation_results_test{model_name.replace(' ', '_').replace('/', '_')}.json"
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"📁 Output saved to: {out_file}")
