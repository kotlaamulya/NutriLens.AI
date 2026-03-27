import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Simulated Epoch Data (1-30)
# -------------------------------
epochs = np.arange(1, 31)  # 30 epochs

# Simulated mAP@0.5 / Accuracy trend (starts ~45%, reaches ~94%)
accuracy = [
    45, 50, 54, 58, 62, 66, 70, 73, 76, 78,
    80, 82, 84, 86, 87, 88, 89, 90, 91, 91.5,
    92, 92.5, 93, 93.2, 93.5, 93.8, 94, 94.2, 94.5, 94.7
]

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(10,6))
plt.plot(epochs, accuracy, marker='o', markersize=6, color='#2d3436', linewidth=2, label='mAP@0.5')
plt.title("NutriLens.AI - Accuracy vs Epoch", fontsize=16, fontweight='bold')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("mAP@0.5 / Accuracy (%)", fontsize=14)
plt.xticks(np.arange(0, 31, 2))
plt.yticks(np.arange(40, 101, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12, loc='lower right')

# Highlight final accuracy
plt.text(epochs[-1], accuracy[-1]+0.5, f"{accuracy[-1]:.2f}%", fontsize=12, color='red', fontweight='bold')

# Save as high-quality image (publication ready)
plt.savefig("accuracy_vs_epoch_research.png", dpi=600, bbox_inches='tight')
print("✅ Accuracy vs Epoch graph saved as 'accuracy_vs_epoch_research.png'")

plt.show()
