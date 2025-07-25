import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# === List of NLI evaluation files ===
nli_files = [
    "evaluation_results_testBERT-base_(MNLI).json",
    "evaluation_results_testcross-encoder_nli-deberta-base.json",
    "evaluation_results_testDeBERTa-v3-base_(MNLI_FEVER_ANLI).json",
    "evaluation_results_testfacebook_bart-large-mnli.json",
    "evaluation_results_testmicrosoft_deberta-large-mnli.json",
    "evaluation_results_testprajjwal1_albert-base-v2-mnli.json",
    "evaluation_results_testpritamdeka_PubMedBERT-MNLI-MedNLI.json",
    "evaluation_results_testroberta-large-mnli.json",
    "evaluation_results_testtypeform_distilbert-base-uncased-mnli.json"
]

# === Helper function to extract model name ===
def get_model_key(filename):
    # Remove prefixes/suffixes to get clean model key
    fname = os.path.basename(filename).replace("evaluation_results_test", "").replace(".json", "")
    return fname

# === Process each model file ===
results = []

for file in nli_files:
    model_key = get_model_key(file)

    with open(file, "r") as f:
        raw = json.load(f)

    if "Explicit_NOT" not in raw:
        continue

    # Find the key automatically (may not always be 'Roberta')
    sample_query = next(iter(raw["Explicit_NOT"].values()))
    ranker_key = next(iter(sample_query))

    data = []
    for ex in raw["Explicit_NOT"].values():
        for doc in ex[ranker_key]["ranking"]:
            data.append({
                "e": doc["e"],
                "n": doc["n"],
                "c": doc["c"],
                "relevance": doc["relevance"]
            })

    df = pd.DataFrame(data)
    if df.empty:
        continue

    # PCA projection
    features = df[["e", "n", "c"]].values
    pca = PCA(n_components=2)
    components = pca.fit_transform(features)
    df["pca1"] = components[:, 0]
    df["pca2"] = components[:, 1]

    # Explained variance
    explained_var = pca.explained_variance_ratio_

    # Class separation
    centroid_0 = df[df["relevance"] == 0][["pca1", "pca2"]].mean().values
    centroid_1 = df[df["relevance"] == 1][["pca1", "pca2"]].mean().values
    distance = euclidean(centroid_0, centroid_1)

    std_0 = df[df["relevance"] == 0][["pca1", "pca2"]].std()
    std_1 = df[df["relevance"] == 1][["pca1", "pca2"]].std()

    results.append({
        "Model": model_key,
        "Explained_PCA1 (%)": round(explained_var[0] * 100, 2),
        "Explained_PCA2 (%)": round(explained_var[1] * 100, 2),
        "Centroid_Distance": round(distance, 4),
        "Std0_PCA1": round(std_0.iloc[0], 4),
        "Std0_PCA2": round(std_0.iloc[1], 4),
        "Std1_PCA1": round(std_1.iloc[0], 4),
        "Std1_PCA2": round(std_1.iloc[1], 4),
    })

# Display results
df_result = pd.DataFrame(results)
df_result = df_result.sort_values(by="Centroid_Distance", ascending=False)
print(df_result.to_string(index=False))
