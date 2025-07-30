import os
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


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

    def compute_nli_scores(self, query, doc_text):
        inputs = self.nli_tokenizer(query, doc_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        start = time.time()
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        elapsed = time.time() - start
        probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        scores = dict(zip(self.label_order, probs))
        return scores, elapsed


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "json", "data")
    combination_path = os.path.join(script_dir, "combination.json")

    with open(combination_path) as f:
        combo = json.load(f)
        hypothesis_list = [h["hypothesis"] if isinstance(h, dict) else h for h in combo["rephrased_hypotheses"]][:5]

    model_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    model_ops = ModelOperations(model_id)

    query_ids = {1}  # Modify as needed
    examples = []
    for qid in query_ids:
        input_path = os.path.join(base_dir, f"query{qid}.json")
        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path}")
            continue
        with open(input_path) as f:
            examples.extend(json.load(f))

    all_results = {}

    for i, example in enumerate(examples):
        example_id = f"example_{example.get('id', i)}"
        all_results[example_id] = []

        for doc in example["documents"]:
            best_combined_score = -1
            best_hypothesis = None
            best_scores = None
            best_time = 0

            for hypothesis in hypothesis_list:
                scores, elapsed = model_ops.compute_nli_scores(hypothesis, doc["text"])
                combined = scores["entailment"] + scores["neutral"]
                if combined > best_combined_score:
                    best_combined_score = combined
                    best_hypothesis = hypothesis
                    best_scores = scores
                    best_time = elapsed

            all_results[example_id].append({
                "doc_id": doc["doc_id"],
                "relevance": doc.get("relevance", None),
                "best_hypothesis": best_hypothesis,
                "entailment": best_scores["entailment"],
                "neutral": best_scores["neutral"],
                "contradiction": best_scores["contradiction"],
                "combined_en": best_scores["entailment"] + best_scores["neutral"],
                "nli_time": round(best_time, 4)
            })

    output_path = f"nli_combination_results_{model_id.replace('/', '_')}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Completed. Output saved to: {output_path}")
