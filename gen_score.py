import json
import os
import re
from curve import sigmoid_e, inverted_sigmoid

MODEL_CONFIGS = {
    "typeform_distilbert-base-uncased-mnli": {
        "activation": {
            "e": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.5},
            "n": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.5},
            "c": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.5}
        },
        "weights": (0.5, 1.0, 1.3)
    },
    "cross-encoder_nli-deberta-base": {
        "activation": {
            "e": {"type": "sigmoid", "k": 6, "midpoint": 0.6},
            "n": {"type": "sigmoid", "k": 6, "midpoint": 0.2},
            "c": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.2}
        },
        "weights": (0.5, 1.0, 1.3)
    },
    "DeBERTa-v3-base_(MNLI_FEVER_ANLI)": {
        "activation": {
            "e": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.5},
            "n": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.2},
            "c": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.6}
        },
        "weights": (0.5, 1.2, 1.3)
    },
    "BERT-base_(MNLI)": {
        "activation": {
            "e": {"type": "sigmoid", "k": 6, "midpoint": 0.5},
            "n": {"type": "sigmoid", "k": 6, "midpoint": 0.2},
            "c": {"type": "sigmoid", "k": 4, "midpoint": 0.2}
        },
        "weights": (0.5, 1.2, 1.3)
    },
    "pritamdeka_PubMedBERT-MNLI-MedNLI": {
        "activation": {
            "e": {"type": "sigmoid", "k": 6, "midpoint": 0.6},
            "n": {"type": "sigmoid", "k": 6, "midpoint": 0.3},
            "c": {"type": "sigmoid", "k": 4, "midpoint": 0.6}
        },
        "weights": (0.5, 1.0, 1.3)
    },
    "roberta-large-mnli": {
        "activation": {
            "e": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.2},
            "n": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.3},
            "c": {"type": "sigmoid", "k": 6, "midpoint": 0.2}
        },
        "weights": (0.5, 1.2, 1.3)
    },
    "microsoft_deberta-large-mnli": {
        "activation": {
            "e": {"type": "sigmoid", "k": 6, "midpoint": 0.6},
            "n": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.6},
            "c": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.5}
        },
        "weights": (0.5, 1.0, 1.3)
    },
    "prajjwal1_albert-base-v2-mnli": {
        "activation": {
            "e": {"type": "sigmoid", "k": 4, "midpoint": 0.5},
            "n": {"type": "sigmoid", "k": 6, "midpoint": 0.2},
            "c": {"type": "sigmoid", "k": 6, "midpoint": 0.5}
        },
        "weights": (0.5, 1.2, 1.3)
    },
    "facebook_bart_large_mnli": {
        "activation": {
            "e": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.3},
            "n": {"type": "sigmoid", "k": 6, "midpoint": 0.3},
            "c": {"type": "sigmoid", "k": 6, "midpoint": 0.2}
        },
        "weights": (0.5, 1.2, 1.3)
    }
}




ACTIVATION_FUNCTIONS = {
    "sigmoid": sigmoid_e,
    "inverted_sigmoid": inverted_sigmoid
}

def load_data_by_example(file_path):
    with open(file_path) as f:
        return json.load(f)

def compute_roberta_scores_by_example(data, weights, activation_config):
    roberta_scores = {}
    reg_penalty = 0.1 * sum(w**2 for w in weights)

    for query_type, examples in data.items():
        for example_id, entry in examples.items():
            if "Roberta" in entry:
                r_scores = []

                for r in entry["Roberta"]["ranking"]:
                    e_conf = activation_config["e"]
                    n_conf = activation_config["n"]
                    c_conf = activation_config["c"]

                    e = ACTIVATION_FUNCTIONS[e_conf["type"]](r['e'], k=e_conf["k"], midpoint=e_conf["midpoint"])
                    n = ACTIVATION_FUNCTIONS[n_conf["type"]](r['n'], k=n_conf["k"], midpoint=n_conf["midpoint"])
                    c = ACTIVATION_FUNCTIONS[c_conf["type"]](r['c'], k=c_conf["k"], midpoint=c_conf["midpoint"])

                    score = (
                        weights[0] * e +
                        weights[1] * n +
                        weights[2] * c
                    ) - reg_penalty

                    r_scores.append({
                        "doc_id": r.get("doc_id", "unknown"),
                        "score": score,
                        "relevance": r["relevance"]
                    })

                roberta_scores[example_id] = sorted(r_scores, key=lambda x: x["score"], reverse=True)

    return roberta_scores

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved {filename}")

def extract_model_key(filename):
    return filename.replace("evaluation_results_", "").replace(".json", "").split("-")[0]

def main():
    folder = "/Users/L020774/Movies/heu/NLP"

    for filename in os.listdir(folder):
        if not filename.startswith("evaluation_results_") or not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder, filename)
        model_key = filename.replace("evaluation_results_", "").replace(".json", "")

        matched_key = next((key for key in MODEL_CONFIGS if model_key.startswith(key)), None)

        if matched_key is None:
            print(f"⏭️ Skipping {filename} — no config match for '{model_key}'")
            continue

        print(f"📂 Processing: {filepath} using config for '{matched_key}'")

        config = MODEL_CONFIGS[matched_key]
        data = load_data_by_example(filepath)
        roberta_data = compute_roberta_scores_by_example(data, config["weights"], config["activation"])

        output_file = f"roberta_scores_by_example_{matched_key}.json"
        save_json(roberta_data, output_file)

if __name__ == "__main__":
    main()
