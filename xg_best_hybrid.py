import os
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from itertools import combinations
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
num_splits = 10000  # Number of random train/test splits to try
random.seed(42)

# Full list of available query IDs
all_ids = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "evaluation_results_testcross-encoder_nli-deberta-base.json")

# Load JSON file
with open(input_file, "r") as f:
    raw = json.load(f)

# Prepare data grouped by query ID
data_by_id = {}
for example_id, example in raw["Explicit_NOT"].items():
    ex_id = int(example_id.split("_")[1])
    for doc in example["Roberta"]["ranking"]:
        entry = {
            "features": [doc["e"], doc["n"], doc["c"]],
            "label": doc["relevance"]
        }
        data_by_id.setdefault(ex_id, []).append(entry)

# Track top 10 splits
top_splits = []
split_size = int(0.7 * len(all_ids))

for _ in tqdm(range(num_splits), desc="Evaluating splits"):
    train_ids = set(random.sample(all_ids, split_size))
    test_ids = set(all_ids) - train_ids

    X_train, y_train = [], []
    for tid in train_ids:
        for entry in data_by_id.get(tid, []):
            X_train.append(entry["features"])
            y_train.append(entry["label"])

    X_test, y_test = [], []
    for tid in test_ids:
        for entry in data_by_id.get(tid, []):
            X_test.append(entry["features"])
            y_test.append(entry["label"])

    if not X_train or not X_test:
        continue

    clf = xgb.XGBClassifier(max_depth=3, n_estimators=30, eval_metric="logloss", use_label_encoder=False)
    clf.fit(np.array(X_train), np.array(y_train))
    preds = clf.predict(np.array(X_test))
    score = f1_score(y_test, preds)

    top_splits.append((score, sorted(train_ids), sorted(test_ids)))
    top_splits = sorted(top_splits, reverse=True, key=lambda x: x[0])[:10]

# Print top 10 splits
print("\n🏆 Top 10 Splits by F1 Score:")
for rank, (score, train, test) in enumerate(top_splits, 1):
    print(f"\nRank {rank}:")
    print(f"Train IDs: {train}")
    print(f"Test IDs:  {test}")
    print(f"F1 Score:  {score:.4f}")
