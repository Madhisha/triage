import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOP_N = 20
MAJORITY_THRESHOLDS = [50, 60, 65, 70, 75, 80, 85, 90]


def clean_text(text: str) -> str:
    if text is None:
        return "unknown"
    text = str(text).strip().lower()
    if not text:
        return "unknown"
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())
    return text if text else "unknown"


def normalize_raw_text(text: str) -> str:
    if text is None:
        return "unknown"
    text = str(text).strip()
    return text if text else "unknown"


def iter_rows(csv_files: Iterable[Path]) -> Iterable[dict]:
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


def parse_acuity(value: str) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip()
    if not value:
        return "unknown"
    try:
        return str(int(float(value)))
    except ValueError:
        return value


def collect_counts(
    csv_files: List[Path], cleaned: bool
) -> Tuple[Counter, Dict[str, Counter], Dict[str, Counter], int]:
    overall = Counter()
    classwise = defaultdict(Counter)
    complaint_class = defaultdict(Counter)
    total_rows = 0

    for row in iter_rows(csv_files):
        complaint_raw = row.get("chiefcomplaint")
        complaint = clean_text(complaint_raw) if cleaned else normalize_raw_text(complaint_raw)
        acuity = parse_acuity(row.get("acuity"))

        overall[complaint] += 1
        classwise[acuity][complaint] += 1
        complaint_class[complaint][acuity] += 1
        total_rows += 1

    return overall, classwise, complaint_class, total_rows


def build_majority_rows(complaint_class: Dict[str, Counter]) -> List[dict]:
    rows = []
    for complaint, class_counter in complaint_class.items():
        total = sum(class_counter.values())
        if total == 0:
            continue
        majority_class, majority_count = class_counter.most_common(1)[0]
        majority_pct = majority_count / total * 100.0
        rows.append(
            {
                "chiefcomplaint": complaint,
                "total_count": total,
                "majority_class": majority_class,
                "majority_count": majority_count,
                "majority_percent": majority_pct,
            }
        )

    rows.sort(key=lambda x: (-x["majority_percent"], -x["total_count"], x["chiefcomplaint"]))
    return rows


def count_above_thresholds(majority_rows: List[dict], thresholds: List[int]) -> Dict[int, int]:
    result = {}
    for threshold in thresholds:
        result[threshold] = sum(1 for row in majority_rows if row["majority_percent"] > threshold)
    return result


def write_overall_csv(path: Path, counts: Counter, total: int, top_n: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "chiefcomplaint", "count", "percent"])
        for idx, (complaint, count) in enumerate(counts.most_common(top_n), start=1):
            pct = (count / total * 100.0) if total else 0.0
            writer.writerow([idx, complaint, count, f"{pct:.2f}"])


def write_classwise_csv(path: Path, classwise: Dict[str, Counter], top_n: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["acuity_class", "rank", "chiefcomplaint", "count", "percent_within_class"])

        for acuity in sorted(classwise.keys(), key=lambda x: (x == "unknown", x)):
            counter = classwise[acuity]
            class_total = sum(counter.values())
            for idx, (complaint, count) in enumerate(counter.most_common(top_n), start=1):
                pct = (count / class_total * 100.0) if class_total else 0.0
                writer.writerow([acuity, idx, complaint, count, f"{pct:.2f}"])


def write_majority_by_complaint_csv(path: Path, majority_rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "chiefcomplaint",
                "total_count",
                "majority_class",
                "majority_count",
                "majority_percent",
            ]
        )
        for row in majority_rows:
            writer.writerow(
                [
                    row["chiefcomplaint"],
                    row["total_count"],
                    row["majority_class"],
                    row["majority_count"],
                    f"{row['majority_percent']:.2f}",
                ]
            )


def write_majority_threshold_counts_csv(path: Path, threshold_counts: Dict[int, int]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold_rule", "complaints_count"])
        for threshold in sorted(threshold_counts.keys()):
            writer.writerow([f"majority_percent > {threshold}", threshold_counts[threshold]])


def format_top_lines(counter: Counter, total: int, top_n: int) -> List[str]:
    lines = []
    for idx, (complaint, count) in enumerate(counter.most_common(top_n), start=1):
        pct = (count / total * 100.0) if total else 0.0
        lines.append(f"{idx:>2}. {complaint} | count={count} | pct={pct:.2f}%")
    return lines


def write_summary_report(
    path: Path,
    before_overall: Counter,
    before_classwise: Dict[str, Counter],
    before_total: int,
    after_overall: Counter,
    after_classwise: Dict[str, Counter],
    after_total: int,
    before_majority_rows: List[dict],
    after_majority_rows: List[dict],
    before_threshold_counts: Dict[int, int],
    after_threshold_counts: Dict[int, int],
    top_n: int,
) -> None:
    lines = []
    lines.append("Chief Complaint Frequency Analysis")
    lines.append("=" * 34)
    lines.append("")
    lines.append("Scope")
    lines.append("- Before preprocessing: raw_data/triage_{train,valid,test}.csv (raw chiefcomplaint text)")
    lines.append("- After preprocessing: ml_model/relabelled_data/triage_{train,valid,test}_relabeled.csv")
    lines.append("  with chief complaint cleaning rules from ml_model/ml_preprocess.py")
    lines.append("")
    lines.append(f"Rows analyzed (before): {before_total}")
    lines.append(f"Rows analyzed (after):  {after_total}")
    lines.append(f"Unique complaints (before): {len(before_overall)}")
    lines.append(f"Unique complaints (after):  {len(after_overall)}")
    lines.append("")
    lines.append("Majority-class percentage by complaint")
    lines.append("- Full list saved in majority_by_complaint_before.csv and majority_by_complaint_after.csv")
    lines.append("- Threshold counts use strict greater-than, e.g., majority_percent > 65")
    lines.append("")
    lines.append("Complaints above majority thresholds (BEFORE preprocessing)")
    for threshold in MAJORITY_THRESHOLDS:
        lines.append(f"- > {threshold}%: {before_threshold_counts.get(threshold, 0)} complaints")
    lines.append("")
    lines.append("Complaints above majority thresholds (AFTER preprocessing)")
    for threshold in MAJORITY_THRESHOLDS:
        lines.append(f"- > {threshold}%: {after_threshold_counts.get(threshold, 0)} complaints")
    lines.append("")
    lines.append("Top 20 complaints by majority percentage BEFORE preprocessing")
    for idx, row in enumerate(before_majority_rows[:20], start=1):
        lines.append(
            f"{idx:>2}. {row['chiefcomplaint']} | majority_class={row['majority_class']} | "
            f"majority_pct={row['majority_percent']:.2f}% | total={row['total_count']}"
        )
    lines.append("")
    lines.append("Top 20 complaints by majority percentage AFTER preprocessing")
    for idx, row in enumerate(after_majority_rows[:20], start=1):
        lines.append(
            f"{idx:>2}. {row['chiefcomplaint']} | majority_class={row['majority_class']} | "
            f"majority_pct={row['majority_percent']:.2f}% | total={row['total_count']}"
        )
    lines.append("")
    lines.append(f"Top {top_n} overall complaints BEFORE preprocessing")
    lines.extend(format_top_lines(before_overall, before_total, top_n))
    lines.append("")
    lines.append(f"Top {top_n} overall complaints AFTER preprocessing")
    lines.extend(format_top_lines(after_overall, after_total, top_n))
    lines.append("")

    lines.append(f"Top {top_n} classwise complaints BEFORE preprocessing")
    for acuity in sorted(before_classwise.keys(), key=lambda x: (x == "unknown", x)):
        class_total = sum(before_classwise[acuity].values())
        lines.append(f"Class {acuity} (n={class_total})")
        lines.extend(format_top_lines(before_classwise[acuity], class_total, top_n))
        lines.append("")

    lines.append(f"Top {top_n} classwise complaints AFTER preprocessing")
    for acuity in sorted(after_classwise.keys(), key=lambda x: (x == "unknown", x)):
        class_total = sum(after_classwise[acuity].values())
        lines.append(f"Class {acuity} (n={class_total})")
        lines.extend(format_top_lines(after_classwise[acuity], class_total, top_n))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # Prefer the repository root: .../pw2 (two levels above this file).
    # Fallback to current working directory if the script is copied elsewhere.
    script_path = Path(__file__).resolve()
    repo_candidate = script_path.parents[2]
    project_root = repo_candidate if (repo_candidate / "raw_data").exists() else Path.cwd()

    raw_files = [
        project_root / "raw_data" / "triage_train.csv",
        project_root / "raw_data" / "triage_valid.csv",
        project_root / "raw_data" / "triage_test.csv",
    ]
    after_files = [
        project_root / "ml_model" / "relabelled_data" / "triage_train_relabeled.csv",
        project_root / "ml_model" / "relabelled_data" / "triage_valid_relabeled.csv",
        project_root / "ml_model" / "relabelled_data" / "triage_test_relabeled.csv",
    ]

    missing = [p for p in raw_files + after_files if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_str}")

    output_dir = project_root / "analysis" / "text_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    before_overall, before_classwise, before_complaint_class, before_total = collect_counts(raw_files, cleaned=False)
    after_overall, after_classwise, after_complaint_class, after_total = collect_counts(after_files, cleaned=True)

    before_majority_rows = build_majority_rows(before_complaint_class)
    after_majority_rows = build_majority_rows(after_complaint_class)
    before_threshold_counts = count_above_thresholds(before_majority_rows, MAJORITY_THRESHOLDS)
    after_threshold_counts = count_above_thresholds(after_majority_rows, MAJORITY_THRESHOLDS)

    write_overall_csv(output_dir / "top_complaints_overall_before.csv", before_overall, before_total, TOP_N)
    write_overall_csv(output_dir / "top_complaints_overall_after.csv", after_overall, after_total, TOP_N)
    write_classwise_csv(output_dir / "top_complaints_classwise_before.csv", before_classwise, TOP_N)
    write_classwise_csv(output_dir / "top_complaints_classwise_after.csv", after_classwise, TOP_N)
    write_majority_by_complaint_csv(output_dir / "majority_by_complaint_before.csv", before_majority_rows)
    write_majority_by_complaint_csv(output_dir / "majority_by_complaint_after.csv", after_majority_rows)
    write_majority_threshold_counts_csv(output_dir / "majority_threshold_counts_before.csv", before_threshold_counts)
    write_majority_threshold_counts_csv(output_dir / "majority_threshold_counts_after.csv", after_threshold_counts)

    write_summary_report(
        output_dir / "chief_complaint_analysis_summary.txt",
        before_overall,
        before_classwise,
        before_total,
        after_overall,
        after_classwise,
        after_total,
        before_majority_rows,
        after_majority_rows,
        before_threshold_counts,
        after_threshold_counts,
        TOP_N,
    )

    print(f"Saved analysis files to: {output_dir}")


if __name__ == "__main__":
    main()