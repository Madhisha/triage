from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_complaint_label(text: str) -> str:
    text = "" if pd.isna(text) else str(text).strip().lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())
    return text if text else "unknown"


def load_inputs(base_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "overall_before": base_dir / "top_complaints_overall_before.csv",
        "overall_after": base_dir / "top_complaints_overall_after.csv",
        "class_before": base_dir / "top_complaints_classwise_before.csv",
        "class_after": base_dir / "top_complaints_classwise_after.csv",
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        missing_txt = "\n".join(missing)
        raise FileNotFoundError(f"Missing required input files:\n{missing_txt}")

    data = {k: pd.read_csv(v) for k, v in files.items()}
    data["overall_before"] = ensure_numeric(data["overall_before"], ["rank", "count", "percent"])
    data["overall_after"] = ensure_numeric(data["overall_after"], ["rank", "count", "percent"])
    data["class_before"] = ensure_numeric(data["class_before"], ["rank", "count", "percent_within_class"])
    data["class_after"] = ensure_numeric(data["class_after"], ["rank", "count", "percent_within_class"])
    return data


def save_overall_before_after_bars(df_before: pd.DataFrame, df_after: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    b = df_before.copy()
    a = df_after.copy()

    b["chiefcomplaint_norm"] = b["chiefcomplaint"].apply(normalize_complaint_label)
    a["chiefcomplaint_norm"] = a["chiefcomplaint"].apply(normalize_complaint_label)

    b = (
        b.groupby("chiefcomplaint_norm", as_index=False)["count"]
        .sum()
        .rename(columns={"chiefcomplaint_norm": "chiefcomplaint", "count": "before_count"})
    )
    a = (
        a.groupby("chiefcomplaint_norm", as_index=False)["count"]
        .sum()
        .rename(columns={"chiefcomplaint_norm": "chiefcomplaint", "count": "after_count"})
    )

    merged = pd.merge(a, b, on="chiefcomplaint", how="outer").fillna(0)
    merged["display_score"] = merged[["before_count", "after_count"]].max(axis=1)
    merged = merged.sort_values("display_score", ascending=False).head(top_n)
    merged = merged.sort_values("after_count", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(merged))
    bar_h = 0.42
    ax.barh(
        y - bar_h / 2,
        merged["before_count"],
        height=bar_h,
        alpha=0.9,
        color="#2b6cb0",
        edgecolor="none",
        label="Before",
    )
    ax.barh(
        y + bar_h / 2,
        merged["after_count"],
        height=bar_h,
        alpha=0.9,
        color="#e07a2d",
        edgecolor="none",
        label="After",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(merged["chiefcomplaint"])
    ax.set_xlabel("Count")
    ax.set_title("Top Chief Complaints: Before vs After Cleaning")
    ax.grid(False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "overall_before_after_bars.png", dpi=180)
    plt.close(fig)


def save_after_classwise_heatmap(df_class_after: pd.DataFrame, output_dir: Path, top_n_per_class: int = 10) -> None:
    limited = (
        df_class_after.sort_values(["acuity_class", "rank"], ascending=[True, True])
        .groupby("acuity_class", group_keys=False)
        .head(top_n_per_class)
    )

    pivot = limited.pivot_table(
        index="chiefcomplaint",
        columns="acuity_class",
        values="percent_within_class",
        aggfunc="max",
        fill_value=0,
    )

    # Keep rows with strongest signal first.
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, 12))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.2, linecolor="white", ax=ax)
    ax.set_title("After Cleaning: Classwise Complaint Share Heatmap (%)")
    ax.set_xlabel("Acuity Class")
    ax.set_ylabel("Chief Complaint")
    plt.tight_layout()
    plt.savefig(output_dir / "classwise_after_heatmap.png", dpi=180)
    plt.close(fig)


def save_after_classwise_panels(df_class_after: pd.DataFrame, output_dir: Path, top_n: int = 10) -> None:
    class_order = [c for c in ["1", "2", "3"] if c in set(df_class_after["acuity_class"].astype(str))]
    if not class_order:
        class_order = sorted(df_class_after["acuity_class"].astype(str).unique().tolist())

    fig, axes = plt.subplots(len(class_order), 1, figsize=(12, 4 * len(class_order)), squeeze=False)

    for idx, cls in enumerate(class_order):
        ax = axes[idx, 0]
        subset = (
            df_class_after[df_class_after["acuity_class"].astype(str) == cls]
            .sort_values("count", ascending=False)
            .head(top_n)
            .sort_values("count", ascending=True)
        )
        sns.barplot(
            data=subset,
            x="count",
            y="chiefcomplaint",
            hue="chiefcomplaint",
            dodge=False,
            legend=False,
            ax=ax,
            palette="crest",
        )
        ax.set_title(f"Class {cls}: Top {top_n} Complaints After Cleaning")
        ax.set_xlabel("Count")
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig(output_dir / "classwise_after_panels.png", dpi=180)
    plt.close(fig)


def save_slope_change_plot(df_before: pd.DataFrame, df_after: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    b = df_before[["chiefcomplaint", "count"]].rename(columns={"count": "before_count"})
    a = df_after[["chiefcomplaint", "count"]].rename(columns={"count": "after_count"})
    merged = pd.merge(b, a, on="chiefcomplaint", how="inner")
    merged["max_count"] = merged[["before_count", "after_count"]].max(axis=1)
    merged = merged.sort_values("max_count", ascending=False).head(top_n)
    merged = merged.sort_values("after_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    for i, row in merged.reset_index(drop=True).iterrows():
        ax.plot([0, 1], [row["before_count"], row["after_count"]], color="gray", alpha=0.6, linewidth=1)
        ax.scatter([0], [row["before_count"]], color="#4575b4", s=35)
        ax.scatter([1], [row["after_count"]], color="#d7301f", s=35)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Count")
    ax.set_title("Complaint Frequency Change (Common Complaints)")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_before_after_slope.png", dpi=180)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="talk")
    data = load_inputs(base_dir)

    save_overall_before_after_bars(data["overall_before"], data["overall_after"], output_dir)
    save_after_classwise_heatmap(data["class_after"], output_dir)
    save_after_classwise_panels(data["class_after"], output_dir)
    save_slope_change_plot(data["overall_before"], data["overall_after"], output_dir)

    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()