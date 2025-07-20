import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# === Load test data ===
with open("evaluation_results_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json", "r") as f:
    test_data = json.load(f)

# === Extract features ===
data = []
for ex_id, ex in test_data["Explicit_NOT"].items():
    for doc in ex["Roberta"]["ranking"]:
        data.append({
            "e": doc["e"],
            "n": doc["n"],
            "c": doc["c"],
            "relevance": doc["relevance"]
        })

df = pd.DataFrame(data)

# ---------------------------------------
# ✅ 1. 3D Scatter Plot: e vs n vs c
# ---------------------------------------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

relevance_0 = df[df["relevance"] == 0]
relevance_1 = df[df["relevance"] == 1]

ax.scatter(relevance_0["e"], relevance_0["n"], relevance_0["c"], c="red", label="Relevance 0", alpha=0.6)
ax.scatter(relevance_1["e"], relevance_1["n"], relevance_1["c"], c="green", label="Relevance 1", alpha=0.6)

ax.set_xlabel("Entailment (e)")
ax.set_ylabel("Neutral (n)")
ax.set_zlabel("Contradiction (c)")
ax.set_title("3D Distribution of [e, n, c] by Relevance")
ax.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------
# ✅ 2. PCA 2D Projection
# ---------------------------------------
features = df[["e", "n", "c"]].values
labels = df["relevance"]

pca = PCA(n_components=2)
components = pca.fit_transform(features)

df["pca1"] = components[:, 0]
df["pca2"] = components[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="pca1", y="pca2", hue="relevance", palette="Set1", alpha=0.7)
plt.title("PCA Projection of [e, n, c] Features by Relevance")
plt.tight_layout()
plt.show()
