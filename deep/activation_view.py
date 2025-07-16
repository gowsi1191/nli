import numpy as np
import matplotlib.pyplot as plt

# Sigmoid-e definition
def sigmoid_e(x, k=1, midpoint=0.5):
    return 1 / (1 + np.exp(-k * (x - midpoint)))

# Configs from MODEL_CONFIGS
activations = {
    "e": {"type": "sigmoid_e", "k": 5, "midpoint": 0.3},
    "n": {"type": "sigmoid_e", "k": 6, "midpoint": 0.6},
    "c": {"type": "sigmoid_e", "k": 5, "midpoint": 0.6}
}

# Generate x values
x = np.linspace(0, 1, 500)

# Plot
plt.figure(figsize=(8, 5))
for label, cfg in activations.items():
    y = sigmoid_e(x, k=cfg["k"], midpoint=cfg["midpoint"])
    plt.plot(x, y, label=f"{label.upper()} → k={cfg['k']}, mid={cfg['midpoint']}")

plt.title("Sigmoid_e Activation Functions (k, midpoint)")
plt.xlabel("x")
plt.ylabel("sigmoid_e(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
