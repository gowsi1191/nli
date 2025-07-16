import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt

# --- File path ---
file_path = "/Users/L020774/Documents/nlp_nli/nli/evaluation_results_DeBERTa-v3-base_(MNLI_FEVER_ANLI).json"

# --- Load Data ---
with open(file_path) as f:
    raw_data = json.load(f)

e_0, e_1 = [], []
n_0, n_1 = [], []
c_0, c_1 = [], []

for outer in raw_data.values():
    for example in outer.values():
        if "Roberta" not in example:
            continue
        for doc in example["Roberta"]["ranking"]:
            rel = doc["relevance"]
            e_val = doc["e"]  # use raw value
            n_val = doc["n"]  # use raw value
            c_val = doc["c"]  # use raw value
            
            if rel == 0:
                e_0.append(e_val)
                n_0.append(n_val)
                c_0.append(c_val)
            elif rel == 1:
                e_1.append(e_val)
                n_1.append(n_val)
                c_1.append(c_val)

# --- Plot ---
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(range(len(e_0)), e_0, color='green', label='Rel 0', alpha=0.6)
plt.scatter(range(len(e_1)), e_1, color='red', label='Rel 1', alpha=0.6)
plt.title("Raw e")
plt.xlabel("Document Index")
plt.ylabel("e value")
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(range(len(n_0)), n_0, color='green', label='Rel 0', alpha=0.6)
plt.scatter(range(len(n_1)), n_1, color='red', label='Rel 1', alpha=0.6)
plt.title("Raw n")
plt.xlabel("Document Index")
plt.ylabel("n value")
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(range(len(c_0)), c_0, color='green', label='Rel 0', alpha=0.6)
plt.scatter(range(len(c_1)), c_1, color='red', label='Rel 1', alpha=0.6)
plt.title("Raw c")
plt.xlabel("Document Index")
plt.ylabel("c value")
plt.legend()

plt.tight_layout()
plt.show()
