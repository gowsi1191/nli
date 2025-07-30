import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# === Configuration ===
train_ids = {    1, 2, 3, 4, 5, 8, 10,9, 11, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 28, 29,
    43, 44, 45, 47, 49, 50, 51, 52, 53, 22, 26, 27, 30, 31, 34, 37, 42, 46, 48}

# === Directory Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir
files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# === Iterate over all files ===
for file in files:
    input_path = os.path.join(input_dir, file)
    model_name = file.replace(".json", "").replace("evaluation_results_test", "")
    model_path = os.path.join(script_dir, f"xgb_model_{model_name}.json")

    print(f"\n📁 Processing file: {file}")

    # Load data
    with open(input_path, "r") as f:
        raw = json.load(f)

    data = []
    try:
        for example_id, example in raw["Explicit_NOT"].items():
            ex_id = int(example_id.split("_")[1])
            if ex_id not in train_ids:
                continue
            for doc in example["Roberta"]["ranking"]:
                data.append({
                    "e": doc["e"],
                    "n": doc["n"],
                    "c": doc["c"],
                    "relevance": doc["relevance"]
                })
    except KeyError:
        print(f"⚠️ Skipped {file} — Missing required fields.")
        continue

    if not data:
        print(f"⚠️ Skipped {file} — No training data found for Train IDs.")
        continue

    # Prepare features
    X = np.array([[d["e"], d["n"], d["c"]] for d in data])
    y = np.array([d["relevance"] for d in data])

    # Train XGBoost model
    clf = xgb.XGBClassifier(max_depth=3, n_estimators=30, eval_metric="logloss")
    f1_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
    print("✅ F1 CV Scores:", np.round(f1_scores, 4))
    print("🎯 Mean F1:", round(f1_scores.mean(), 4))

    # Save trained model
    clf.fit(X, y)
    clf.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
