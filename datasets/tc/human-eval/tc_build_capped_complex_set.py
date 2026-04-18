#!/usr/bin/env python3
import argparse
import ast
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


NONNR_FILES = [
    "it2_nonNR_tc.tsv",
    "it3_nonNR_tc.tsv",
    "it4_nonNR_tc.tsv",
    "it5_nonNR_tc.tsv",
]

COMPLEX_FILES = [
    "it3_complex_1_tc.tsv",
    "it4_complex_1_tc.tsv",
    "it4_complex_2_tc.tsv",
    "it5_complex_1_tc.tsv",
    "it5_complex_2_tc.tsv",
    "it5_complex_3_tc.tsv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy nonNR TC files and cap each complex TC file to at most N rows per plan."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative paths are resolved from here.",
    )
    parser.add_argument(
        "--input-dir",
        default="datasets/tc",
        help="Directory containing the original TC files.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/tc/capped_complex_plan5",
        help="Directory to write the new TC set into.",
    )
    parser.add_argument(
        "--max-per-plan",
        type=int,
        default=5,
        help="Maximum sampled rows per plan for each complex file.",
    )
    parser.add_argument(
        "--low-support-threshold",
        type=int,
        default=4,
        help="Print final sampled plans at or below this count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def resolve_dir(root, directory):
    path = Path(directory)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def parse_plan(raw_answer):
    if raw_answer is None:
        return "MISSING_ANSWER"
    text = str(raw_answer).strip()
    if not text:
        return "EMPTY_ANSWER"
    try:
        data = ast.literal_eval(text)
    except Exception:
        return "INVALID_ANSWER"
    if not isinstance(data, dict):
        return "NON_DICT_ANSWER"
    plan = data.get("plan")
    if plan is None:
        return "MISSING_PLAN"
    plan_text = str(plan).strip()
    return plan_text if plan_text else "EMPTY_PLAN"


def read_tsv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def copy_nonnr_files(input_dir, output_dir):
    copied = []
    for file_name in NONNR_FILES:
        source = input_dir / file_name
        if not source.exists():
            raise FileNotFoundError(f"Missing nonNR file: {source}")
        target = output_dir / file_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(file_name)
    return copied


def cap_complex_file(source_path, target_path, max_per_plan, seed):
    fieldnames, rows = read_tsv_rows(source_path)
    plan_to_rows = defaultdict(list)
    for idx, row in enumerate(rows):
        copied = dict(row)
        copied["_row_order"] = idx
        plan_to_rows[parse_plan(row.get("answer"))].append(copied)

    rng = random.Random(seed)
    sampled_rows = []
    original_counts = Counter()
    sampled_counts = Counter()

    for plan in sorted(plan_to_rows):
        plan_rows = list(plan_to_rows[plan])
        original_counts[plan] = len(plan_rows)
        rng.shuffle(plan_rows)
        kept = plan_rows[:max_per_plan]
        sampled_counts[plan] = len(kept)
        sampled_rows.extend(kept)

    sampled_rows.sort(key=lambda row: row["_row_order"])
    cleaned_rows = []
    for row in sampled_rows:
        copied = dict(row)
        copied.pop("_row_order", None)
        cleaned_rows.append(copied)

    write_tsv(target_path, fieldnames, cleaned_rows)
    return len(rows), len(cleaned_rows), original_counts, sampled_counts


def build_manifest_row(file_name, original_count, sampled_count, output_dir, root):
    return {
        "file_name": file_name,
        "original_count": original_count,
        "sampled_count": sampled_count,
        "output_path": str((output_dir / file_name).relative_to(root)),
    }


def build_plan_summary_rows(file_name, original_count, sampled_count, original_counts, sampled_counts):
    rows = [
        {
            "file_name": file_name,
            "plan": "__TOTAL__",
            "original_count": original_count,
            "sampled_count": sampled_count,
            "original_ratio_in_file": "1.0000",
            "sampled_ratio_in_file": "1.0000" if sampled_count else "0.0000",
        }
    ]
    for plan, count in sorted(original_counts.items(), key=lambda item: (-item[1], item[0])):
        sampled = sampled_counts.get(plan, 0)
        rows.append(
            {
                "file_name": file_name,
                "plan": plan,
                "original_count": count,
                "sampled_count": sampled,
                "original_ratio_in_file": f"{count / original_count:.4f}" if original_count else "0.0000",
                "sampled_ratio_in_file": f"{sampled / sampled_count:.4f}" if sampled_count else "0.0000",
            }
        )
    return rows


def print_low_support_plans(summary_rows, threshold):
    flagged = [
        row
        for row in summary_rows
        if row["plan"] != "__TOTAL__" and int(row["sampled_count"]) <= threshold
    ]
    if not flagged:
        print(f"No plans at or below {threshold} rows in the final set.")
        return

    print(f"Plans at or below {threshold} rows in the final set:")
    for row in sorted(flagged, key=lambda item: (item["file_name"], int(item["sampled_count"]), item["plan"])):
        print(f"  {row['file_name']}\t{row['plan']}\t{row['sampled_count']}")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    input_dir = resolve_dir(root, args.input_dir)
    output_dir = resolve_dir(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_nonnr_files(input_dir, output_dir)

    manifest_rows = []
    summary_rows = []

    for file_name in COMPLEX_FILES:
        source = input_dir / file_name
        if not source.exists():
            raise FileNotFoundError(f"Missing complex file: {source}")
        target = output_dir / file_name
        original_count, sampled_count, original_counts, sampled_counts = cap_complex_file(
            source,
            target,
            args.max_per_plan,
            args.seed,
        )
        manifest_rows.append(
            build_manifest_row(file_name, original_count, sampled_count, output_dir, root)
        )
        summary_rows.extend(
            build_plan_summary_rows(
                file_name,
                original_count,
                sampled_count,
                original_counts,
                sampled_counts,
            )
        )

    for file_name in NONNR_FILES:
        fieldnames, rows = read_tsv_rows(output_dir / file_name)
        _ = fieldnames
        plan_counts = Counter(parse_plan(row.get("answer")) for row in rows)
        manifest_rows.append(
            build_manifest_row(file_name, len(rows), len(rows), output_dir, root)
        )
        summary_rows.extend(
            build_plan_summary_rows(
                file_name,
                len(rows),
                len(rows),
                plan_counts,
                plan_counts,
            )
        )

    write_tsv(
        output_dir / "capped_manifest.tsv",
        ["file_name", "original_count", "sampled_count", "output_path"],
        sorted(manifest_rows, key=lambda item: item["file_name"]),
    )
    write_tsv(
        output_dir / "capped_plan_summary.tsv",
        ["file_name", "plan", "original_count", "sampled_count", "original_ratio_in_file", "sampled_ratio_in_file"],
        sorted(summary_rows, key=lambda item: (item["file_name"], item["plan"] != "__TOTAL__", item["plan"])),
    )

    total_sampled = sum(int(row["sampled_count"]) for row in manifest_rows)
    print(f"Output directory: {output_dir}")
    print(f"Total rows in final set: {total_sampled}")
    print_low_support_plans(summary_rows, args.low_support_threshold)


if __name__ == "__main__":
    main()
