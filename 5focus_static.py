import numpy as np
import json
import os
from statistics import stdev
from curve import sigmoid_e, inverted_sigmoid
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

MODEL_CONFIGS = {
    "DeBERTa_v3_base_MNLI_FEVER_ANLI": {
        "activation": {
            "e": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.5},
            "n": {"type": "inverted_sigmoid", "k": 4, "midpoint": 0.2},
            "c": {"type": "inverted_sigmoid", "k": 6, "midpoint": 0.6}
        },
        "weights": (0.5, 1.2, 1.3)
    }
}

def normalize_model_name(filename):
    return filename.replace("evaluation_results_", "").replace(".json", "").replace("(", "").replace(")", "").replace("-", "_").replace("__", "_")

def load_data(file_path):
    with open(file_path) as f:
        raw_data = json.load(f)

    datasets = []
    for outer in raw_data.values():
        for example in outer.values():
            if "Roberta" not in example or "BGE" not in example:
                continue

            roberta_rank = example["Roberta"]["ranking"]
            bge_rank = example["BGE"]["ranking"]

            if len(roberta_rank) != len(bge_rank):
                continue

            datasets.append({"Roberta": roberta_rank, "BGE": bge_rank})
    return datasets

def evaluate_model(datasets, e_fn, n_fn, c_fn, weights, lambda_reg=0.1):
    p1_flags = []
    score_margins = []
    raw_scores_collector = []

    for data in datasets:
        rankings = data['Roberta']
        if len(rankings) < 2:
            continue

        reg_penalty = 0.1 * sum(w**2 for w in weights)

        raw_scores = []
        for r in rankings:
            enc_e = e_fn(r['e'])
            enc_n = n_fn(r['n'])
            enc_c = c_fn(r['c'])

            score = (
                weights[0] * enc_e +
                weights[1] * enc_n +
                weights[2] * enc_c
            ) - reg_penalty

            raw_scores.append((score, r['relevance']))

        if len(raw_scores) >= 2:
            raw_scores.sort(reverse=True, key=lambda x: x[0])
            raw_scores_collector.extend([s for s, _ in raw_scores])
            score_margins.append(raw_scores[0][0] - raw_scores[1][0])

        top_idx = 0  # now that raw_scores is sorted
        p1_flags.append(1 if raw_scores[top_idx][1] == 0 else 0)

    p_at_1 = np.mean(p1_flags) if p1_flags else 0
    p1_std = stdev(p1_flags) if len(p1_flags) > 1 else 0
    avg_margin = trim_mean(score_margins, 0.2) if score_margins else 0
    margin_std = stdev(score_margins) if len(score_margins) > 1 else 0
    generalization_score = avg_margin - 0.5 * p1_std - 0.3 * margin_std if p1_flags and len(p1_flags) > 1 else 0
    margin_cv = margin_std / avg_margin if avg_margin != 0 else float('inf')

    return {
        "P@1": p_at_1,
        "P@1_STD": p1_std,
        "Avg Margin": avg_margin,
        "Margin_STD": margin_std,
        "Generalization_Score": generalization_score,
        "n_samples": len(p1_flags),
        "Margin_CV": margin_cv,
        "Raw_Scores": raw_scores_collector
    }

def main():
    file_path = "evaluation_results_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json"
    model_key = normalize_model_name(file_path)
    print(f"\n--- Processing File: {file_path} as {model_key} ---")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    datasets = load_data(file_path)
    config = MODEL_CONFIGS[model_key]

    activation = config['activation']
    weights = config['weights']

    e_fn = lambda x: inverted_sigmoid(x, k=activation['e']['k'], midpoint=activation['e']['midpoint'])
    n_fn = lambda x: inverted_sigmoid(x, k=activation['n']['k'], midpoint=activation['n']['midpoint'])
    c_fn = lambda x: inverted_sigmoid(x, k=activation['c']['k'], midpoint=activation['c']['midpoint'])

    result = evaluate_model(datasets, e_fn, n_fn, c_fn, weights)

    print(f"\nP@1: {result['P@1']:.3f}")
    print(f"Generalization Score: {result['Generalization_Score']:.4f}")
    print(f"Avg Margin: {result['Avg Margin']:.4f}, Margin_STD: {result['Margin_STD']:.4f}, CV: {result['Margin_CV']:.4f}")

if __name__ == "__main__":
    main()