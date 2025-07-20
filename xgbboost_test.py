import os
import json
import numpy as np
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, confusion_matrix
import re
import pandas as pd

# === File paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "xgb_model.json")
test_data_path = os.path.join(script_dir, "json", "data", "2explicit_test.json")


# === Load trained model ===
clf = xgb.XGBClassifier()
clf.load_model(model_path)

# === Load test dataset ===
with open(test_data_path, "r") as f:
    test_data = json.load(f)

# === Embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Negation keywords ===
NEGATION_TERMS = ["not", "excluding", "without", "other than", "doesn't include", "does not include"]

def has_negation(query, doc_text):
    query = query.lower()
    doc_text = doc_text.lower()
    for term in NEGATION_TERMS:
        if term in query and any(word in doc_text for word in term.split()):
            return 1
    return 0

def jaccard_similarity(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    return len(a_set & b_set) / len(a_set | b_set)

X_test, y_test = [], []
prediction_rows = []

for entry in test_data:
    query = entry["query"]
    for doc in entry["documents"]:
        doc_text = doc["text"]
        label = doc["relevance"]

        # Embeddings
        q_emb = model.encode(query)
        d_emb = model.encode(doc_text)

        # Features
        abs_diff = np.abs(q_emb - d_emb)
        cos_sim = cosine_similarity([q_emb], [d_emb])[0][0]
        jaccard = jaccard_similarity(query, doc_text)
        neg_flag = has_negation(query, doc_text)

        feature_vector = np.concatenate([abs_diff, [cos_sim, jaccard, neg_flag]])
        X_test.append(feature_vector)
        y_test.append(label)

        prediction_rows.append({
            "query": query,
            "doc_id": doc["doc_id"],
            "text": doc_text,
            "true_label": label
        })

X_test = np.array(X_test)
y_test = np.array(y_test)

# === Make predictions ===
y_pred = clf.predict(X_test)

# === Add predictions to table ===
for i in range(len(prediction_rows)):
    prediction_rows[i]["predicted_label"] = int(y_pred[i])

# === Print metrics ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n📉 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save predictions to CSV ===
df = pd.DataFrame(prediction_rows)
output_csv = os.path.join(script_dir, "explicit_test_predictions.csv")
df.to_csv(output_csv, index=False)
print(f"\n✅ Predictions saved to {output_csv}")
