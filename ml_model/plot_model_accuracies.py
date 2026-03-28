from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


EXCLUDED_MODELS = {"svm"}
PALETTE = {
    "purple": "#7E57C2",
    "teal": "#00B8D4",
    "pink": "#EC407A",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _format_model_name(stem: str) -> str:
    name = stem.replace("_default", "").replace("_tuned", "")
    pretty = name.replace("_", " ").title()
    acronym_fixes = {
        "Mlp": "MLP",
        "Svm": "SVM",
    }
    return acronym_fixes.get(pretty, pretty)


def _extract_validation_accuracy(text: str) -> Optional[float]:
    match = re.search(
        r"Evaluation on Validation Set.*?Accuracy:\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        return float(match.group(1))
    # Fallback: first occurrence of Accuracy in file
    fallback = re.search(r"Accuracy:\s*([0-9]*\.?[0-9]+)", text, flags=re.IGNORECASE)
    if fallback:
        return float(fallback.group(1))
    return None


def _extract_best_validation_accuracy(text: str) -> Optional[float]:
    match = re.search(
        r"Best\s+validation\s+accuracy:\s*([0-9]*\.?[0-9]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group(1))


def _extract_selected_tuning_method(text: str) -> Optional[str]:
    match = re.search(
        r"Selected\s+strategy:\s*all\s+tuning\s+techniques\s*->\s*selected\s*([A-Za-z0-9_\- ]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip().lower()


def _extract_tuning_techniques(text: str) -> Dict[str, float]:
    methods: Dict[str, float] = {}
    block_match = re.search(
        r"Tuning\s+method\s+validation\s+accuracies:\s*(.*?)\n\s*Final\s+model\s+parameters:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not block_match:
        return methods

    block = block_match.group(1)
    for method, score in re.findall(
        r"^\s*-\s*([A-Za-z0-9_\- ]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$",
        block,
        flags=re.MULTILINE,
    ):
        methods[method.strip().lower()] = float(score)
    return methods


def _collect_results(
    results_dir: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]], Dict[str, str]]:
    default_acc: Dict[str, float] = {}
    tuned_best_acc: Dict[str, float] = {}
    tuned_methods: Dict[str, Dict[str, float]] = {}
    tuned_selected_method: Dict[str, str] = {}

    for default_path in sorted(results_dir.glob("*_default.txt")):
        model = _format_model_name(default_path.stem)
        if model.strip().lower() in EXCLUDED_MODELS:
            continue
        text = _read_text(default_path)
        val_acc = _extract_validation_accuracy(text)
        if val_acc is not None:
            default_acc[model] = val_acc

    for tuned_path in sorted(results_dir.glob("*_tuned.txt")):
        model = _format_model_name(tuned_path.stem)
        if model.strip().lower() in EXCLUDED_MODELS:
            continue
        text = _read_text(tuned_path)

        best_acc = _extract_best_validation_accuracy(text)
        if best_acc is None:
            best_acc = _extract_validation_accuracy(text)
        if best_acc is not None:
            tuned_best_acc[model] = best_acc

        methods = _extract_tuning_techniques(text)
        if methods:
            tuned_methods[model] = methods

        selected = _extract_selected_tuning_method(text)
        if selected:
            tuned_selected_method[model] = selected

    return default_acc, tuned_best_acc, tuned_methods, tuned_selected_method


def _add_labels_percent(
    ax: plt.Axes,
    bars,
    values: List[Optional[float]],
    x_padding: float = 0.6,
    bold_mask: Optional[List[bool]] = None,
) -> None:
    for idx, (bar, val) in enumerate(zip(bars, values)):
        if val is None or np.isnan(val):
            continue
        is_bold = bold_mask[idx] if bold_mask is not None else False
        ax.text(
            val + x_padding,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold" if is_bold else "normal",
        )


def _plot_default_vs_best_tuned(
    default_acc: Dict[str, float],
    tuned_best_acc: Dict[str, float],
    tuned_selected_method: Dict[str, str],
    output_path: Path,
) -> None:
    models = sorted(default_acc.keys())
    default_values = [default_acc.get(m, np.nan) * 100 if m in default_acc else np.nan for m in models]
    tuned_values = [tuned_best_acc.get(m, np.nan) * 100 if m in tuned_best_acc else np.nan for m in models]

    y = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 1.0)))
    bars_default = ax.barh(
        y - width / 2,
        default_values,
        height=width,
        label="Default",
        color=PALETTE["purple"],
    )
    bars_tuned = ax.barh(
        y + width / 2,
        tuned_values,
        height=width,
        label="Best Tuned",
        color=PALETTE["teal"],
    )

    _add_labels_percent(ax, bars_default, default_values)
    # Add bold accuracy and plain tuning-method text separately.
    for i, (bar, v) in enumerate(zip(bars_tuned, tuned_values)):
        if np.isnan(v):
            continue
        selected_method = tuned_selected_method.get(models[i])
        ax.text(
            v + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
            fontweight="bold",
        )
        if selected_method:
            ax.text(
                v + 6.0,
                bar.get_y() + bar.get_height() / 2,
                f"({selected_method.title()})",
                ha="left",
                va="center",
                fontsize=8,
                color="black",
            )

    # Mark models with no tuned file.
    for i, v in enumerate(tuned_values):
        if np.isnan(v):
            ax.text(1.0, y[i] + width / 2, "N/A", ha="left", va="center", fontsize=8, color=PALETTE["pink"])
            continue

    ax.set_title("Validation Accuracy: Default vs Best Tuned")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_tuning_techniques_by_model(
    tuned_methods: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    models = sorted(tuned_methods.keys())
    if not models:
        return

    preferred_order = ["random", "grid", "bayesian"]
    all_methods = {m for model in models for m in tuned_methods[model].keys()}
    methods = [m for m in preferred_order if m in all_methods]
    methods.extend(sorted(m for m in all_methods if m not in methods))

    y = np.arange(len(models))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 1.0)))

    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2) * width
        vals = [
            tuned_methods[model].get(method, np.nan) * 100 if method in tuned_methods[model] else np.nan
            for model in models
        ]
        color_cycle = [PALETTE["purple"], PALETTE["teal"], PALETTE["pink"]]
        bars = ax.barh(y + offset, vals, height=width, label=method.title(), color=color_cycle[idx % len(color_cycle)])
        _add_labels_percent(ax, bars, vals, x_padding=0.5)

    ax.set_title("Validation Accuracy by Tuning Technique")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_all_accuracies_single_graph(
    default_acc: Dict[str, float],
    tuned_methods: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    models = sorted(set(default_acc.keys()) | set(tuned_methods.keys()))
    if not models:
        return

    series_names = ["Default", "Random", "Grid", "Bayesian"]
    series_values = {
        "Default": [default_acc.get(m, np.nan) * 100 if m in default_acc else np.nan for m in models],
        "Random": [tuned_methods.get(m, {}).get("random", np.nan) * 100 if "random" in tuned_methods.get(m, {}) else np.nan for m in models],
        "Grid": [tuned_methods.get(m, {}).get("grid", np.nan) * 100 if "grid" in tuned_methods.get(m, {}) else np.nan for m in models],
        "Bayesian": [tuned_methods.get(m, {}).get("bayesian", np.nan) * 100 if "bayesian" in tuned_methods.get(m, {}) else np.nan for m in models],
    }

    # Per model, bold only the highest available accuracy label.
    per_model_max: List[float] = []
    for model_idx in range(len(models)):
        vals = [series_values[s][model_idx] for s in series_names]
        valid_vals = [v for v in vals if not np.isnan(v)]
        per_model_max.append(max(valid_vals) if valid_vals else np.nan)

    color_map = {
        "Default": PALETTE["purple"],
        "Random": PALETTE["teal"],
        "Grid": PALETTE["pink"],
        "Bayesian": "#43A047",
    }

    y = np.arange(len(models))
    width = 0.78 / len(series_names)

    fig, ax = plt.subplots(figsize=(12, max(6, len(models) * 1.05)))

    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2) * width
        vals = series_values[series_name]
        bold_mask = [
            (not np.isnan(v)) and (not np.isnan(per_model_max[i])) and np.isclose(v, per_model_max[i])
            for i, v in enumerate(vals)
        ]
        bars = ax.barh(
            y + offset,
            vals,
            height=width,
            label=series_name,
            color=color_map[series_name],
        )
        _add_labels_percent(ax, bars, vals, x_padding=0.4, bold_mask=bold_mask)

    ax.set_title("Validation Accuracy: Default + All Tuning Methods")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Model")
    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate model accuracy plots from ml_model/training_results text files."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_results",
        help="Directory containing *_default.txt and *_tuned.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "training_results",
        help="Directory to save generated plot images.",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    default_acc, tuned_best_acc, tuned_methods, tuned_selected_method = _collect_results(results_dir)

    if not default_acc:
        raise FileNotFoundError(f"No default result files found in: {results_dir}")

    combined_path = output_dir / "all_accuracies_default_random_grid_bayesian.png"
    _plot_all_accuracies_single_graph(default_acc, tuned_methods, combined_path)
    print(f"Saved: {combined_path}")


if __name__ == "__main__":
    main()
