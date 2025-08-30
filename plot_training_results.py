import matplotlib.pyplot as plt

# Your training data from the output
epochs = [1, 2, 3, 4, 5]
losses = [1.5316, 0.5831, 0.3567, 0.2678, 0.2323]
accuracies = [85.28, 95.09, 95.85, 93.96, 98.11]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training loss
ax1.plot(epochs, losses, "b-o", linewidth=2, markersize=8)
ax1.set_title("Training Loss Over Time", fontsize=14, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(losses) * 1.1)

# Plot validation accuracy
ax2.plot(epochs, accuracies, "g-o", linewidth=2, markersize=8)
ax2.set_title("Validation Accuracy Over Time", fontsize=14, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 100)

# Add accuracy values as text on the plot
for i, acc in enumerate(accuracies):
    ax2.annotate(
        f"{acc:.1f}%",
        (epochs[i], acc),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

plt.tight_layout()
plt.suptitle(
    "Card Recognition AI Training Results", fontsize=16, fontweight="bold", y=1.02
)
plt.show()

print(f"🎉 Training completed successfully!")
print(f"📈 Final accuracy: {accuracies[-1]:.2f}%")
print(f"📉 Final loss: {losses[-1]:.4f}")
print(f"🚀 Improvement: {accuracies[-1] - accuracies[0]:.2f}% accuracy gain!")
