import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# -----------------------------------
# ✅ Step 1: Enter your actual labels below
# -----------------------------------

# True labels (Ground truth)
y_true = [
    "Idli", "Dosa", "Biryani", "Chapathi", "Puri", "Rice", "Idli", "Curry", "Chapathi", "Rice"
]

# Predicted labels (Model predictions)
y_pred = [
    "Idli", "Dosa", "Biryani", "Chapathi", "Puri", "Biryani", "Idli", "Curry", "Rice", "Rice"
]

# -----------------------------------
# ✅ Step 2: Compute Confusion Matrix and Accuracy
# -----------------------------------
labels = sorted(list(set(y_true + y_pred)))
cm = confusion_matrix(y_true, y_pred, labels=labels)
acc = accuracy_score(y_true, y_pred) * 100  # Convert to %
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# -----------------------------------
# ✅ Step 3: Plot, Annotate & Save Confusion Matrix
# -----------------------------------
plt.figure(figsize=(8,6))
disp.plot(cmap='viridis', xticks_rotation=45, colorbar=True)
plt.title(f"NutriLens.AI - Confusion Matrix (Accuracy: {acc:.2f}%)", fontsize=14, pad=20)
plt.xlabel("Predicted Food Item", fontsize=12)
plt.ylabel("Actual Food Item", fontsize=12)
plt.grid(False)

# ✅ Save confusion matrix image
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix image saved as 'confusion_matrix.png' (Accuracy: {acc:.2f}%)")

# ✅ Display confusion matrix plot
plt.show()

# -----------------------------------
# ✅ Step 4: Print Performance Report
# -----------------------------------
print("\n--- NutriLens.AI Classification Report ---")
print(classification_report(y_true, y_pred))
print(f"\n✅ Overall Model Accuracy: {acc:.2f}%")
