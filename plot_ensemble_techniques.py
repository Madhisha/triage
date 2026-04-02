import matplotlib.pyplot as plt
from pathlib import Path


def plot_ensemble_methods_comparison() -> None:
	models = ["Hard\nVoting", "Soft\nVoting", "Weighted\nVoting", "Stacking\n(LR)"]

	validation_accuracy = [0.8527, 0.8510, 0.8524, 0.8514]
	test_accuracy = [0.8554, 0.8551, 0.8567, 0.8612]
	test_macro_f1 = [0.7121, 0.7158, 0.7248, 0.7249]

	# Convert to percentage scale to match the chart style.
	validation_pct = [value * 100 for value in validation_accuracy]
	test_pct = [value * 100 for value in test_accuracy]
	macro_f1_pct = [value * 100 for value in test_macro_f1]

	x_positions = list(range(len(models)))
	bar_width = 0.25

	fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
	ax.set_facecolor("white")

	bars_val = ax.bar(
		[x - bar_width for x in x_positions],
		validation_pct,
		bar_width,
		label="Validation Accuracy",
		color="#9AD9D3",
	)
	bars_test = ax.bar(
		x_positions,
		test_pct,
		bar_width,
		label="Test Accuracy",
		color="#1F6F78",
	)
	bars_f1 = ax.bar(
		[x + bar_width for x in x_positions],
		macro_f1_pct,
		bar_width,
		label="Test Macro F1",
		color="#2A9D8F",
	)

	ax.set_title("Ensemble Methods Comparison", fontsize=17, weight="bold", pad=16)
	ax.set_xlabel("Ensemble Models", fontsize=15, weight="bold")
	ax.set_ylabel("Score (%)", fontsize=14, weight="bold")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(models, fontsize=12)
	ax.set_ylim(0, 110)
	ax.grid(axis="y", linestyle="--", alpha=0.35)
	ax.legend(
		loc="upper right",
		bbox_to_anchor=(0.98, 0.99),
		ncol=3,
		fontsize=12,
	)

	for bars in (bars_val, bars_test, bars_f1):
		for bar in bars:
			height = bar.get_height()
			ax.text(
				bar.get_x() + bar.get_width() / 2,
				height + 0.35,
				f"{height:.2f}%",
				ha="center",
				va="bottom",
				fontsize=9,
				color="#4d4d4d",
				weight="bold",
			)

	plt.tight_layout(rect=[0, 0, 1, 0.95])
	output_dir = Path(__file__).resolve().parent / "visuals"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_file = output_dir / "ensemble_methods_comparison.png"
	fig.savefig(output_file, dpi=300, bbox_inches="tight")
	plt.show()


if __name__ == "__main__":
	plot_ensemble_methods_comparison()
