import os
import json
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# === Configuration ===
train_ids = {
    1, 2, 3, 4, 5, 8, 10, 11, 16, 17, 18, 19, 20, 21,
    23, 24, 25, 28, 29,
    43, 44, 45, 47, 49, 50, 51, 52, 53, 22, 26, 27, 30, 31, 34, 37, 42, 46, 48
}

# === Directory Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = script_dir
files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

# === Identify input sources ===
deberta_file = next((f for f in files if "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI)" in f), None)
crossenc_file = next((f for f in files if "evaluation_results_testcross-encoder_nli-deberta-base" in f), None)

if not deberta_file or not crossenc_file:
    print("❌ Required files not found. Ensure both DeBERTa and cross-encoder files are present.")
    exit()

print(f"🔹 Using DeBERTa file: {deberta_file}")
print(f"🔹 Using CrossEncoder file: {crossenc_file}")

# === Load both files with safety ===
def load_file(file):
    with open(os.path.join(input_dir, file), "r") as f:
        content = json.load(f)
        return content.get("Explicit_NOT", {})

deberta_data = load_file(deberta_file)
crossenc_data = load_file(crossenc_file)

# === Merge features ===
X, y = [], []

for example_id, deb_entry in deberta_data.items():
    try:
        ex_id = int(example_id.split("_")[1])
    except:
        continue

    if ex_id not in train_ids:
        continue

    # Defensive access
    deb_docs = deb_entry.get("Roberta", {}).get("ranking", [])
    bge_docs = deb_entry.get("BGE", {}).get("ranking", [])
    cross_docs = crossenc_data.get(example_id, {}).get("Roberta", {}).get("ranking", [])

    # Index by doc_id for merging
    bge_map = {doc["doc_id"]: doc for doc in bge_docs if "doc_id" in doc}
    cross_map = {doc["doc_id"]: doc for doc in cross_docs if "doc_id" in doc}

    for d in deb_docs:
        doc_id = d.get("doc_id")
        if not doc_id or doc_id not in bge_map or doc_id not in cross_map:
            continue

        try:
            deb_feat = [d["e"], d["n"], d["c"]]
            bge_feat = [bge_map[doc_id]["score"]]
            cross_feat = [cross_map[doc_id]["e"], cross_map[doc_id]["n"], cross_map[doc_id]["c"]]
            features = deb_feat + bge_feat + cross_feat
            label = d["relevance"]
        except KeyError:
            continue  # Skip incomplete entries

        X.append(features)
        y.append(label)

if not X:
    print("❌ No training data found — check matching doc_ids or field presence.")
    exit()

# === Train and Evaluate ===
X = np.array(X)
y = np.array(y)

clf = xgb.XGBClassifier(max_depth=3, n_estimators=30, eval_metric="logloss")
f1_scores = cross_val_score(clf, X, y, cv=5, scoring="f1")

print("\n✅ F1 CV Scores:", np.round(f1_scores, 4))
print("🎯 Mean F1:", round(f1_scores.mean(), 4))

# === Save model ===
model_name = "xgb_model_DeBERTa+BGE+CrossEncoder.json"
clf.fit(X, y)
clf.save_model(os.path.join(script_dir, model_name))
print(f"✅ Model saved to: {model_name}")
