import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# === Directory Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir  # since files are directly in /nli
files = [f for f in os.listdir(input_dir) if f.endswith(".json")]


# === Iterate over all files ===
for file in files:
    input_path = os.path.join(input_dir, file)
    model_name = file.replace(".json", "").replace("evaluation_results_", "")
    model_path = os.path.join(script_dir, f"xgb_model_{model_name}.json")

    print(f"\n📁 Processing file: {file}")

    # Load data
    with open(input_path, "r") as f:
        raw = json.load(f)

    data = []
    try:
        for example in raw["Explicit_NOT"].values():
            for doc in example["Roberta"]["ranking"]:
                data.append({
                    "e": doc["e"],
                    "n": doc["n"],
                    "c": doc["c"],
                    "relevance": doc["relevance"]
                })
    except KeyError:
        print(f"⚠️ Skipped {file} — No 'Explicit_NOT' or 'Roberta' format.")
        continue

    if not data:
        print(f"⚠️ Skipped {file} — No valid data points found.")
        continue

    # Prepare features
    X, y = [], []
    for d in data:
        X.append([d["e"], d["n"], d["c"]])
        y.append(d["relevance"])

    X = np.array(X)
    y = np.array(y)

    # Train XGBoost model
    clf = xgb.XGBClassifier(max_depth=3, n_estimators=30, eval_metric="logloss")
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    print("✅ F1 CV Scores:", np.round(f1_scores, 4))
    print("🎯 Mean F1:", round(f1_scores.mean(), 4))

    # Save trained model
    clf.fit(X, y)
    clf.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
