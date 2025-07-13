import numpy as np
import json
import os
from statistics import stdev
from curve import sigmoid_e
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

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

def evaluate_model(datasets, e_fn, n_fn, c_fn, bge_fn, weights, lambda_reg=0.1):
    p1_flags = []
    score_margins = []
    raw_scores_collector = []

    for data in datasets:
        rankings = data['Roberta']
        bge_scores = data['BGE']
        if len(rankings) < 2:
            continue

        reg_penalty = lambda_reg * sum(w**2 for w in weights)

        raw_scores = []
        for r, bge in zip(rankings, bge_scores):
            enc_e = e_fn(r['e'])
            enc_n = n_fn(r['n'])
            enc_c = c_fn(r['c'])
            enc_b = bge_fn(bge['score'])
            score = (
                weights[0] * (enc_e + enc_b) +
                weights[1] * enc_n +
                weights[2] * enc_c +
                weights[3] * (enc_e**2 + enc_b**2) +
                weights[4] * enc_n**2 +
                weights[5] * enc_c**2
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

def plot_scores_distribution(scores, output_path):
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=50, range=(0, 1), edgecolor='black', alpha=0.7)
    plt.title("Distribution of Raw Scores (0 to 1)")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    file_path = "evaluation_results_pritamdeka_PubMedBERT-MNLI-MedNLI.json"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    datasets = load_data(file_path)

    weights = (0.5, 1.0, 1.3, 0.1, 0.1, 0.1)
    e_fn = lambda x: sigmoid_e(x, k=6, midpoint=0.2)
    n_fn = lambda x: sigmoid_e(x, k=6, midpoint=0.5)
    c_fn = lambda x: sigmoid_e(x, k=10, midpoint=0.3)
    bge_fn = lambda x: sigmoid_e(x, k=10, midpoint=0.2)

    result = evaluate_model(datasets, e_fn, n_fn, c_fn, bge_fn, weights)

    print(f"P@1 >= 0.80 at Combination 1213 in {file_path}:")
    print(f"  P@1={result['P@1']:.3f}, W={weights},")
    print("  E:sigmoid_e_mid0.2_k6 | N:sigmoid_e_mid0.5_k6 | C:sigmoid_e_mid0.3_k10 | BGE:sigmoid_e_mid0.2_k10")

    plot_scores_distribution(result['Raw_Scores'], "score_distribution_pubmedbert.png")

if __name__ == "__main__":
    main()
