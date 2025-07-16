import numpy as np
import json
import os
import sys
from statistics import stdev
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

# Add parent directory to import `curve`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curve import inverted_sigmoid

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
                weights[2] * enc_c +
                weights[3] * (enc_e * enc_n) +
                weights[4] * (enc_n * enc_c)
            ) - reg_penalty


            raw_scores.append((score, r['relevance']))

        if len(raw_scores) >= 2:
            raw_scores.sort(reverse=True, key=lambda x: x[0])
            score_margins.append(raw_scores[0][0] - raw_scores[1][0])

        top_idx = 0
        p1_flags.append(1 if raw_scores[top_idx][1] == 0 else 0)

    p_at_1 = np.mean(p1_flags) if p1_flags else 0
    p1_std = stdev(p1_flags) if len(p1_flags) > 1 else 0
    avg_margin = trim_mean(score_margins, 0.1) if score_margins else 0
    margin_std = stdev(score_margins) if len(score_margins) > 1 else 0
    generalization_score = avg_margin - 0.5 * p1_std - 0.3 * margin_std if p1_flags and len(p1_flags) > 1 else 0
    margin_cv = margin_std / avg_margin if avg_margin != 0 else float('inf')

    return {
        "P@1": p_at_1,
        "P@1_STD": p1_std,
        "Avg Margin": avg_margin,
        "Margin_STD": margin_std,
        "Generalization_Score": generalization_score,
        "Margin_CV": margin_cv
    }

def main():
    file_path = "/Users/L020774/Documents/nlp_nli/nli/evaluation_results_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    datasets = load_data(file_path)
    weights = (0.5, 1.2, 1.3, 0.8, 0.9)

    # Define ranges for k and midpoint for e, n, c
    k_values = [4, 6, 10]
    mid_values = [0.2, 0.3, 0.5,0.6]

    # Iterate through all combinations
    for k_e in k_values:
        for m_e in mid_values:
            for k_n in k_values:
                for m_n in mid_values:
                    for k_c in k_values:
                        for m_c in mid_values:
                            e_fn = lambda x: inverted_sigmoid(x, k=k_e, midpoint=m_e)
                            n_fn = lambda x: inverted_sigmoid(x, k=k_n, midpoint=m_n)
                            c_fn = lambda x: inverted_sigmoid(x, k=k_c, midpoint=m_c)

                            result = evaluate_model(datasets, e_fn, n_fn, c_fn, weights)
                            if result['P@1']>.85:
                                print(f"\nE:(k={k_e}, m={m_e}) | N:(k={k_n}, m={m_n}) | C:(k={k_c}, m={m_c})")
                                print(f"  ➤ P@1: {result['P@1']:.3f}, Generalization: {result['Generalization_Score']:.4f}")
                                print(f"  ➤ Margin: {result['Avg Margin']:.4f}, STD: {result['Margin_STD']:.4f}, CV: {result['Margin_CV']:.4f}")

if __name__ == "__main__":
    main()
