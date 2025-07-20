import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb

# === File path setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(script_dir, "json", "data", "2explicit.json")

with open(input_path, "r") as f:
    dataset = json.load(f)

# === Sentence transformer for embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Negation keywords ===
NEGATION_TERMS = ["not", "excluding", "without", "other than", "doesn't include", "does not include"]

def has_negation(query, doc_text):
    """Flag if excluded term appears in doc that was negated in query"""
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

X, y = [], []

for example in dataset:
    query = example["query"]
    for doc in example["documents"]:
        doc_text = doc["text"]
        label = doc["relevance"]

        # Sentence embeddings
        q_emb = model.encode(query)
        d_emb = model.encode(doc_text)

        # Feature 1: Embedding absolute difference
        abs_diff = np.abs(q_emb - d_emb)

        # Feature 2: Cosine similarity
        cos_sim = cosine_similarity([q_emb], [d_emb])[0][0]

        # Feature 3: Jaccard similarity
        jaccard = jaccard_similarity(query, doc_text)

        # Feature 4: Negation flag
        neg_flag = has_negation(query, doc_text)

        # Final feature vector
        feature_vector = np.concatenate([abs_diff, [cos_sim, jaccard, neg_flag]])
        X.append(feature_vector)
        y.append(label)

X = np.array(X)
y = np.array(y)

# === XGBoost model ===
clf = xgb.XGBClassifier(max_depth=3, n_estimators=30, use_label_encoder=False, eval_metric="logloss")

# === Cross-validation F1 ===
scores = cross_val_score(clf, X, y, cv=5, scoring="f1")
print("✅ F1 Scores per fold:", scores)
print("🎯 Mean F1 Score:", scores.mean())

# === Train on full dataset ===
clf.fit(X, y)

# === Save model ===
model_path = os.path.join(script_dir, "xgb_model.json")
clf.save_model(model_path)
print(f"✅ Trained model saved to {model_path}")
