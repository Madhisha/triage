import matplotlib.pyplot as plt
import numpy as np

# Best test-set accuracies from the pre-relabeling chart (blue-green chart)
without_relabelling = {
    "Random Forest": 68.54,
    "MLP": 70.36,
    "LightGBM": 67.97,
}

# Best validation accuracies from post-relabeling chart:
# ml_model/training_results/all_accuracies_default_random_grid_bayesian_1.png
with_relabelling = {
    "Random Forest": 84.41,
    "MLP": 83.86,
    "LightGBM": 84.73,
}

models = ["Random Forest", "MLP", "LightGBM"]
without_vals = [without_relabelling[m] for m in models]
with_vals = [with_relabelling[m] for m in models]

x = np.arange(len(models))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_without = ax.bar(
    x - width / 2,
    without_vals,
    width,
    label="Without Relabelling",
    color="#f4a261",
    edgecolor="white",
    linewidth=1.0,
)
bars_with = ax.bar(
    x + width / 2,
    with_vals,
    width,
    label="With Relabelling",
    color="#264653",
    edgecolor="white",
    linewidth=1.0,
)

for bar in bars_without:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=10)

for bar in bars_with:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_title("Best Accuracy Comparison: With vs Without Relabelling", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Models", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 100)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper right", frameon=True)

plt.tight_layout()
out_path = "../ml_model/training_results/comparison_with_vs_without_relabelling_rf_mlp_lightgbm.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")
