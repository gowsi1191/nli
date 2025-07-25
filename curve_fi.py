import numpy as np
import matplotlib.pyplot as plt

# Number of bands (e.g., B = 8)
B = 8

# Manually define non-linear xᵢ values between 0 and 1
x_vals = np.array([0.00, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 0.95])
assert len(x_vals) == B, "x_vals must have B elements"

# Cosine decay curve: wᵢ = cos(π * xᵢ / 2)
weights = np.cos(np.pi * x_vals / 2)

# Normalize and allocate 100 documents
normalized_weights = weights / weights.sum()
allocations = np.round(normalized_weights * 100).astype(int)

# Plotting the curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, B + 1), allocations, marker='o')
plt.title("Cosine-Based Adaptive Band Sampling (B=8)")
plt.xlabel("Band Number (k)")
plt.ylabel("Documents per Band")
plt.grid(True)
plt.xticks(range(1, B + 1))
plt.show()

# Print the values
print("xᵢ values:", x_vals)
print("Cosine Weights:", np.round(weights, 4))
print("Normalized Weights:", np.round(normalized_weights, 4))
print("Documents allocated per band:", allocations)
