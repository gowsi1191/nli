import numpy as np
import json
import os
from statistics import stdev
from curve import sigmoid_e, inverted_sigmoid
from scipy.stats import trim_mean
import itertools
import matplotlib.pyplot as plt

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

        reg_penalty = lambda_reg * sum(w**2 for w in weights)

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
            raw_scores_collector.extend([s for s, _ in raw_scores])
            sorted_scores = sorted([s for s, _ in raw_scores], reverse=True)
            score_margins.append(sorted_scores[0] - sorted_scores[1])

        top_idx = np.argmax([s for s, _ in raw_scores])
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
    files = [f for f in os.listdir('.') if f.startswith("evaluation_results_DeB") and f.endswith(".json")]

    weight_options = [
        (0.5, 1.0, 1.3),
        (0.5, 1.2, 1.3),
    ]

    k_values = [6, 8, 10]
    midpoints = [0.2, 0.3, 0.5, 0.6]
    sigmoid_types = [sigmoid_e, inverted_sigmoid]

    for file_path in files:
        model_key = normalize_model_name(file_path)
        print(f"\n--- Processing File: {file_path} as {model_key} ---")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        datasets = load_data(file_path)
        total_combinations = 0
        evaluated = []

        for e_type in sigmoid_types:
            for n_type in sigmoid_types:
                for c_type in sigmoid_types:
                    for ke in k_values:
                        for km in midpoints:
                            for kn in k_values:
                                for knm in midpoints:
                                    for kc in k_values:
                                        for kcm in midpoints:
                                            for weights in weight_options:
                                                total_combinations += 1
                                                e_fn = lambda x, k=ke, m=km, fn=e_type: fn(x, k=k, midpoint=m)
                                                n_fn = lambda x, k=kn, m=knm, fn=n_type: fn(x, k=k, midpoint=m)
                                                c_fn = lambda x, k=kc, m=kcm, fn=c_type: fn(x, k=k, midpoint=m)

                                                result = evaluate_model(
                                                    datasets,
                                                    lambda x: e_fn(x),
                                                    lambda x: n_fn(x),
                                                    lambda x: c_fn(x),
                                                    weights
                                                )
                                                if result["P@1"]>=0.88:
                                                    print(result["P@1"])

                                                evaluated.append((result["P@1"], result["Generalization_Score"], e_type.__name__, ke, km, n_type.__name__, kn, knm, c_type.__name__, kc, kcm, weights, result))

        unique_top = []
        seen = set()

        for combo in sorted(evaluated, key=lambda x: (x[0], x[1]), reverse=True):
            key = (combo[2], combo[3], combo[4], combo[5], combo[6], combo[7], combo[8], combo[9], combo[10])
            if key not in seen:
                unique_top.append(combo)
                seen.add(key)
            if len(unique_top) == 10:
                break

        print("\n=== Top 10 Unique Combinations by P@1 and Generalization Score ===\n")
        for i, combo in enumerate(unique_top, 1):
            p1, gen, e_name, ke, km, n_name, kn, knm, c_name, kc, kcm, weights, result = combo
            print(f"{i:2d}. E:{e_name}_mid{km}_k{ke} | N:{n_name}_mid{knm}_k{kn} | C:{c_name}_mid{kcm}_k{kc} | W:{weights}")
            print(f"    => P@1: {p1:.3f}, Generalization: {gen:.4f}, Avg Margin: {result['Avg Margin']:.4f}, Margin_STD: {result['Margin_STD']:.4f}, CV: {result['Margin_CV']:.4f}\n")

if __name__ == "__main__":
    main()
