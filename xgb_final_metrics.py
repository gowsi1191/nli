import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, f1_score

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "xgb_nli_enc_model.json")
test_path = os.path.join(script_dir, "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json")

# === Load model ===
clf = xgb.XGBClassifier()
clf.load_model(model_path)


# === Load and process test data ===
with open(test_path, "r") as f:
    test_data = json.load(f)

X_test, y_test, query_ids, rows = [], [], [], []

for ex_id, ex in test_data["Explicit_NOT"].items():
    for doc in ex["Roberta"]["ranking"]:
        e, n, c = doc["e"], doc["n"], doc["c"]

        X_test.append([e, n, c])
        y_test.append(doc["relevance"])
        query_ids.append(ex_id)

        rows.append({
            "example_id": ex_id,
            "doc_id": doc["doc_id"],
            "query": doc["query"],
            "text": doc["text"],
            "true_label": doc["relevance"]
        })

X_test = np.array(X_test)
y_test = np.array(y_test)

# === Predictions ===
probs = clf.predict_proba(X_test)[:, 0]  # score for relevance 0
y_pred = clf.predict(X_test)

# === Add predictions to dataframe ===
for i in range(len(rows)):
    rows[i]["score"] = float(probs[i])
    rows[i]["pred_label"] = int(y_pred[i])

df = pd.DataFrame(rows)
csv_path = os.path.join(script_dir, "xgb_nli_enc_scores.csv")
df.to_csv(csv_path, index=False)
print(f"✅ Saved scores to {csv_path}")

# === Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"📉 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"🎯 F1 Score: {f1_score(y_test, y_pred):.4f}")

# === Feature Importance Plot ===
xgb.plot_importance(clf)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# === Tree Visualization (1st tree) ===
xgb.plot_tree(clf, num_trees=0, rankdir='LR')
plt.title("First Tree Visualization")
plt.tight_layout()
plt.show()

# === Score Distribution ===
plt.hist(probs[np.array(y_test) == 0], bins=30, alpha=0.6, label="Relevance 0")
plt.hist(probs[np.array(y_test) == 1], bins=30, alpha=0.6, label="Relevance 1")
plt.title("XGBoost Score Distribution")
plt.xlabel("Predicted Probability (Relevance 0)")
plt.ylabel("Count")
plt.legend()
plt.show()

# === Margin Analysis ===
margins = []
for qid, group in df.groupby("example_id"):
    sorted_group = group.sort_values(by="score", ascending=False)
    if len(sorted_group) > 1:
        margin = sorted_group.iloc[0]["score"] - sorted_group.iloc[1]["score"]
        margins.append(margin)

margin_mean = np.mean(margins)
margin_std = np.std(margins)
cv = margin_std / margin_mean if margin_mean != 0 else 0
generalization_score = margin_mean * (1 - cv)

print("\n📐 Margin Analysis:")
print(f"✅ Margin Mean: {margin_mean:.4f}")
print(f"✅ Margin STD:  {margin_std:.4f}")
print(f"✅ CV (Std/Mean): {cv:.4f}")
print(f"🎯 Generalization Score: {generalization_score:.4f}")

# === Margin Histogram ===
plt.hist(margins, bins=30, color='purple', alpha=0.7)
plt.title("Score Margin Distribution (Top 1 - Top 2)")
plt.xlabel("Score Margin")
plt.ylabel("Number of Queries")
plt.axvline(margin_mean, color='red', linestyle='--', label=f"Mean = {margin_mean:.4f}")
plt.legend()
plt.tight_layout()
plt.show()
