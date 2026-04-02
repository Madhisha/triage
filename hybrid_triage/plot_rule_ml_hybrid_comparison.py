import re
from pathlib import Path

import matplotlib.pyplot as plt


def extract_accuracy(text: str, pattern: str, label: str) -> float:
    normalized_text = text.replace("\r\n", "\n")
    match = re.search(pattern, normalized_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find {label} using pattern: {pattern}")
    return float(match.group(1))


def extract_overall_hybrid_accuracy(text: str) -> float:
    """Extract accuracy from the OVERALL RESULTS block only."""
    normalized_text = text.replace("\r\n", "\n")

    section_match = re.search(
        r"OVERALL RESULTS:\s*(.*?)\n-{5,}",
        normalized_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not section_match:
        raise ValueError("Could not find OVERALL RESULTS section in relabelled report.")

    section_text = section_match.group(1)
    acc_match = re.search(r"^\s*Accuracy\s*:\s*([0-9]*\.?[0-9]+)", section_text, flags=re.IGNORECASE | re.MULTILINE)
    if not acc_match:
        raise ValueError("Could not find Accuracy line inside OVERALL RESULTS section.")

    return float(acc_match.group(1))


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_rule_path = base_dir / "hybrid_triage_results_raw_rule.txt"
    relabeled_path = base_dir / "hybrid_triage_results_relabelled.txt"

    raw_rule_text = raw_rule_path.read_text(encoding="utf-8")
    relabeled_text = relabeled_path.read_text(encoding="utf-8")

    # From raw-rule report: use rule-based-only subset accuracy.
    rule_only_acc = extract_accuracy(
        raw_rule_text,
        r"Rule-Based subset\s*\(.*?\):\s*\n\s*Accuracy:\s*([0-9]*\.?[0-9]+)",
        "Rule-Based subset accuracy",
    )

    # From relabeled report: use hybrid overall + ML subset accuracy.
    hybrid_acc = extract_overall_hybrid_accuracy(relabeled_text)
    ml_only_acc = extract_accuracy(
        relabeled_text,
        r"ML\s*\(Stacking LR\)\s*subset\s*\(.*?\):\s*\n\s*Accuracy:\s*([0-9]*\.?[0-9]+)",
        "ML subset accuracy",
    )

    labels = ["Rule Only", "ML Only", "Rule + ML"]
    values = [rule_only_acc, ml_only_acc, hybrid_acc]
    # Use a brighter palette (no dark navy).
    colors = ["#E76F51", "#2A9D8F", "#FF6FAE"]

    pct_values = [v * 100 for v in values]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    bars = ax.bar(labels, pct_values, color=colors)

    ax.set_title("Accuracy Comparison: Rule vs ML vs Hybrid", fontsize=14, weight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, pct_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    visuals_dir = base_dir.parent / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    output_path = visuals_dir / "rule_ml_hybrid_accuracy_comparison.png"
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved graph to: {output_path}")
    print(f"Rule Only: {rule_only_acc * 100:.2f}%")
    print(f"ML Only: {ml_only_acc * 100:.2f}%")
    print(f"Rule + ML: {hybrid_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
