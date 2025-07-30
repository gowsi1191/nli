import os
import json
import random
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import f1_score

# === Configuration ===
TRAIN_SIZE = 35
NUM_SPLITS = 10
RANDOM_SEED = 42

# === All Query IDs ===
all_ids = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
           34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
           50, 51, 52, 53]

# === Load Data (e, n, c from Roberta or Cross-Encoder) ===
def load_and_prepare_data(file_path):
    with open(file_path, "r") as f:
        raw = json.load(f)

    data_by_id = {}
    for example_id, example in raw.get("Explicit_NOT", {}).items():
        try:
            ex_id = int(example_id.split("_")[1])
        except:
            continue

        # Check for both keys: Roberta and CrossEncoder
        for source_key in ["Roberta", "CrossEncoder"]:
            doc_list = example.get(source_key, {}).get("ranking", [])
            if doc_list:
                break  # Use the first available one

        for doc in doc_list:
            if all(k in doc for k in ["e", "n", "c", "relevance"]):
                entry = {
                    "features": [doc["e"], doc["n"], doc["c"]],
                    "label": doc["relevance"]
                }
                data_by_id.setdefault(ex_id, []).append(entry)
    return data_by_id

# === Train and Evaluate ===
def train_and_score(data_by_id, train_ids, test_ids):
    X_train, y_train, X_test, y_test = [], [], [], []

    for tid in train_ids:
        for entry in data_by_id.get(tid, []):
            X_train.append(entry["features"])
            y_train.append(entry["label"])

    for tid in test_ids:
        for entry in data_by_id.get(tid, []):
            X_test.append(entry["features"])
            y_test.append(entry["label"])

    if not X_train or not X_test:
        return None

    clf = xgb.XGBClassifier(
        max_depth=3,
        n_estimators=30,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_SEED
    )
    clf.fit(np.array(X_train), np.array(y_train))
    preds = clf.predict(np.array(X_test))
    return f1_score(y_test, preds)

# === Main ===
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    model_files = {
        "DeBERTa-XGB": "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json",
        "CrossEncoder-XGB": "evaluation_results_testcross-encoder_nli-deberta-base.json"
    }

    for model_name, file_name in model_files.items():
        print(f"\n🚀 Model: {model_name}")
        input_path = os.path.join(script_dir, file_name)
        data_by_id = load_and_prepare_data(input_path)

        for i in range(NUM_SPLITS):
            train_ids = sorted(random.sample(all_ids, TRAIN_SIZE))
            test_ids = sorted(list(set(all_ids) - set(train_ids)))
            f1 = train_and_score(data_by_id, train_ids, test_ids)

            print(f"\n🔹 Split {i+1}")
            print(f"Train IDs: {train_ids}")
            print(f"Test IDs:  {test_ids}")
            if f1 is not None:
                print(f"✅ F1 Score: {f1:.4f}")
            else:
                print("⚠️ Skipped (No data for this split)")
