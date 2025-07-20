import numpy as np
import matplotlib.pyplot as plt

# Number of bands (e.g., k = 10)
k = 10

# Cosine decay formula for top-right quarter circle
x = np.linspace(0, np.pi/2, k)
weights = np.cos(x)  # cos(0) = 1, cos(pi/2) = 0

# Normalize to sum to 100
doc_alloc = (weights / weights.sum()) * 100

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(range(1, k+1), doc_alloc, marker='o')
plt.title("Cosine-Based Sampling Curve (Top-Right Quarter Circle)")
plt.xlabel("Band Number (k)")
plt.ylabel("Documents per Band")
plt.grid(True)
plt.show()

# Optional: print allocations
print("Documents allocated per band:", np.round(doc_alloc, 2))
