import numpy as np
import json
import os
import sys
from statistics import stdev
from scipy.stats import trim_mean
import matplotlib.pyplot as plt

# Add parent directory to import `curve`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from curve import sigmoid_e, inverted_sigmoid

# # Model configuration
MODEL_CONFIGS = {
    "DeBERTa_v3_base_MNLI_FEVER_ANLI": {
        "activation": {
            "e": {"type": "sigmoid_e", "k": 5, "midpoint": 0.3},
            "n": {"type": "sigmoid_e", "k": 7, "midpoint": 0.6},
            "c": {"type": "sigmoid_e", "k": 6, "midpoint": 0.6}
        },
        "weights": (0.5, 1.2, 1.3, 0.8, 0.9)
    }
}






# Normalize filename into config key
def normalize_model_name(filename):
    base = os.path.basename(filename)
    return base.replace("evaluation_results_", "").replace(".json", "").replace("(", "").replace(")", "").replace("-", "_").replace("__", "_")

# Load dataset
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
    rel_0_scores = []
    rel_1_scores = []
    default_due_to_tie = []

    for ex_idx, data in enumerate(datasets):
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

            if r['relevance'] == 0:
                rel_0_scores.append(score)
            elif r['relevance'] == 1:
                rel_1_scores.append(score)

        sorted_scores = sorted(raw_scores, key=lambda x: x[0], reverse=True)
        top_score = sorted_scores[0][0]
        top_items = [item for item in raw_scores if abs(item[0] - top_score) < 1e-6]

        if len(top_items) > 1:
            default_due_to_tie.append({"index": ex_idx, "top_scores": top_items})

        score_margins.append(sorted_scores[0][0] - sorted_scores[1][0])
        p1_flags.append(1 if sorted_scores[0][1] == 0 else 0)

    p_at_1 = np.mean(p1_flags) if p1_flags else 0
    p1_std = stdev(p1_flags) if len(p1_flags) > 1 else 0
    avg_margin = trim_mean(score_margins, 0.10) if score_margins else 0
    margin_std = stdev(score_margins) if len(score_margins) > 1 else 0
    margin_cv = margin_std / avg_margin if avg_margin else float('inf')
    generalization_score = avg_margin - 0.5 * p1_std - 0.3 * margin_std

    return {
        "P@1": p_at_1,
        "P@1_STD": p1_std,
        "Avg Margin": avg_margin,
        "Margin_STD": margin_std,
        "Generalization_Score": generalization_score,
        "n_samples": len(p1_flags),
        "Margin_CV": margin_cv,
        "Rel0": rel_0_scores,
        "Rel1": rel_1_scores,
        "DefaultDueToTie": default_due_to_tie
    }


# Main execution
def main():
    file_path = "/Users/L020774/Documents/nlp_nli/nli/evaluation_results_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    model_key = normalize_model_name(file_path)
    print(f"\n--- Processing File: {file_path} as {model_key} ---")

    if model_key not in MODEL_CONFIGS:
        print(f"❌ MODEL_CONFIGS does not contain key: {model_key}")
        return

    datasets = load_data(file_path)
    config = MODEL_CONFIGS[model_key]

    activation = config['activation']
    weights = config['weights']

    e_fn = lambda x: sigmoid_e(x, k=activation['e']['k'], midpoint=activation['e']['midpoint']) if activation['e']['type'] == "sigmoid_e" else inverted_sigmoid(x, k=activation['e']['k'], midpoint=activation['e']['midpoint'])
    n_fn = lambda x: sigmoid_e(x, k=activation['n']['k'], midpoint=activation['n']['midpoint']) if activation['n']['type'] == "sigmoid_e" else inverted_sigmoid(x, k=activation['n']['k'], midpoint=activation['n']['midpoint'])
    c_fn = lambda x: sigmoid_e(x, k=activation['c']['k'], midpoint=activation['c']['midpoint']) if activation['c']['type'] == "sigmoid_e" else inverted_sigmoid(x, k=activation['c']['k'], midpoint=activation['c']['midpoint'])


    result = evaluate_model(datasets, e_fn, n_fn, c_fn, weights)

    print(f"\nP@1: {result['P@1']:.3f} ± {result['P@1_STD']:.3f}")
    print(f"Generalization Score: {result['Generalization_Score']:.4f}")
    print(f"Avg Margin: {result['Avg Margin']:.4f}, Margin_STD: {result['Margin_STD']:.4f}, CV: {result['Margin_CV']:.4f}")

    # Extra: Print mean and std of raw scores by relevance
    if result["Rel0"]:
        print(f"Relevance 0 → Mean: {np.mean(result['Rel0']):.4f}, Std: {np.std(result['Rel0']):.4f}")
    if result["Rel1"]:
        print(f"Relevance 1 → Mean: {np.mean(result['Rel1']):.4f}, Std: {np.std(result['Rel1']):.4f}")

    if result["DefaultDueToTie"]:
        print(f"\n⚠️ {len(result['DefaultDueToTie'])} examples defaulted to first due to tie:")
        for item in result["DefaultDueToTie"][:5]:  # limit print to first 5 for brevity
            print(f"  Example #{item['index']} — Top tied scores: {[round(s[0], 4) for s in item['top_scores']]}")

    # 🎯 Scatter plot of normalized scores
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(result["Rel0"])), result["Rel0"], color='red', label='Relevance 0', alpha=0.6)
    plt.scatter(range(len(result["Rel1"])), result["Rel1"], color='green', label='Relevance 1', alpha=0.6)
    plt.title("Raw Score Distribution by Relevance")
    plt.xlabel("Document Index")
    plt.ylabel("Raw Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
