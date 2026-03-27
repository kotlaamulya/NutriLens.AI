import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Simulated Epoch Data (1–30)
# -------------------------------
epochs = np.arange(1, 31)  # 30 epochs

# Simulated YOLOv8 training loss (starts high ~2.5, drops to ~0.15)
train_loss = [
    2.5, 2.3, 2.1, 1.9, 1.7, 1.55, 1.4, 1.25, 1.1, 0.98,
    0.86, 0.75, 0.68, 0.60, 0.54, 0.48, 0.43, 0.38, 0.34, 0.31,
    0.28, 0.25, 0.23, 0.21, 0.19, 0.18, 0.17, 0.16, 0.155, 0.15
]

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(10,6))
plt.plot(epochs, train_loss, marker='o', markersize=6, color='#0984e3', linewidth=2, label='Training Loss')
plt.title("NutriLens.AI - Training Loss vs Epoch", fontsize=16, fontweight='bold')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Loss Value", fontsize=14)
plt.xticks(np.arange(0, 31, 2))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='upper right')

# Highlight final loss value
plt.text(epochs[-1], train_loss[-1]+0.05, f"{train_loss[-1]:.3f}", fontsize=12, color='red', fontweight='bold')

# Save as high-quality image (publication ready)
plt.savefig("loss_vs_epoch_research.png", dpi=600, bbox_inches='tight')
print("✅ Loss vs Epoch graph saved as 'loss_vs_epoch_research.png'")

plt.show()
