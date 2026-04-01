import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main() -> None:
	models = [
		"Random Forest",
		"MLP",
		"XGBoost",
		"LightGBM",
		"CatBoost",
		"Logistic Regression",
		"AdaBoost",
	]

	tfidf_only = [84.41, 83.86, 81.57, 80.48, 84.28, 80.06, 75.14]
	groups_only = [82.47, 82.02, 79.80, 78.87, 79.61, 68.88, 68.59]
	tfidf_groups = [83.76, 80.62, 81.52, 80.42, 80.45, 65.76, 73.29]

	x = np.arange(len(models))
	width = 0.26
	colors = ["#1F7A8C", "#BF4342", "#E1B12C"]

	fig, ax = plt.subplots(figsize=(13, 6))

	bars_1 = ax.bar(x - width, tfidf_only, width, label="TF-IDF only", color=colors[0])
	bars_2 = ax.bar(x, groups_only, width, label="Groups only", color=colors[1])
	bars_3 = ax.bar(x + width, tfidf_groups, width, label="TF-IDF + Groups", color=colors[2])

	ax.set_title("Model Accuracy Comparison by Feature Engineering Approach", pad=12)
	ax.set_ylabel("Accuracy (%)")
	ax.set_xlabel("Model")
	ax.set_xticks(x)
	ax.set_xticklabels(models, rotation=20, ha="right")
	ax.set_ylim(60, 90)
	ax.grid(axis="y", linestyle="--", alpha=0.35)
	ax.legend()

	for bars in (bars_1, bars_2, bars_3):
		ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)

	fig.tight_layout()

	output_dir = Path(__file__).resolve().parent / "visuals" / "images"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_path = output_dir / "comparison_diff_tech.png"
	fig.savefig(output_path, dpi=300, bbox_inches="tight")
	print(f"Saved chart to: {output_path}")

	plt.show()


if __name__ == "__main__":
	main()
