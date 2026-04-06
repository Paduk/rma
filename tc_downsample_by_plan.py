#!/usr/bin/env python3
import argparse
import ast
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_TARGETS = {
    "it3_nonNR_tc.tsv": 250,
    "it3_complex_1_tc.tsv": 250,
    "it4_nonNR_tc.tsv": 160,
    "it4_complex_1_tc.tsv": 160,
    "it4_complex_2_tc.tsv": 160,
    "it5_nonNR_tc.tsv": 120,
    "it5_complex_1_tc.tsv": 120,
    "it5_complex_2_tc.tsv": 120,
    "it5_complex_3_tc.tsv": 120,
}
MIN_PLAN_SAMPLES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Downsample TC TSV files while preserving per-file plan distribution."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative paths are resolved from here.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/tc/downsampled_plan_preserved",
        help="Directory to write downsampled TSV files and summaries into.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Override target count as file_name=count. Can be used multiple times.",
    )
    parser.add_argument(
        "--min-plan-samples",
        type=int,
        default=MIN_PLAN_SAMPLES,
        help="Minimum guaranteed sampled rows for each plan when the source plan count is at least this value.",
    )
    return parser.parse_args()


def parse_target_overrides(items):
    overrides = {}
    for item in items:
        file_name, value = item.split("=", 1)
        overrides[file_name.strip()] = int(value)
    return overrides


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
    value = data.get("plan")
    if value is None:
        return "MISSING_PLAN"
    plan = str(value).strip()
    return plan if plan else "EMPTY_PLAN"


def read_tsv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    return fieldnames, rows


def build_minimum_allocations(plan_counts, min_plan_samples):
    minimums = {}
    for plan, count in plan_counts.items():
        if count >= min_plan_samples:
            minimums[plan] = min_plan_samples
        else:
            minimums[plan] = count
    return minimums


def allocate_plan_targets(plan_counts, target_total, min_plan_samples):
    total = sum(plan_counts.values())
    if target_total >= total:
        return dict(plan_counts)

    minimums = build_minimum_allocations(plan_counts, min_plan_samples)
    required_total = sum(minimums.values())
    if required_total > target_total:
        raise ValueError(
            f"Target {target_total} is too small. Minimum required rows to satisfy per-plan floor is {required_total}."
        )

    plans = sorted(plan_counts)
    allocations = dict(minimums)
    remaining_total = target_total - required_total
    adjusted_counts = {
        plan: max(plan_counts[plan] - minimums[plan], 0)
        for plan in plans
    }
    adjustable_total = sum(adjusted_counts.values())

    if remaining_total <= 0 or adjustable_total <= 0:
        return allocations

    remainders = []
    for plan in plans:
        exact = remaining_total * adjusted_counts[plan] / adjustable_total
        floor_value = math.floor(exact)
        addable = min(floor_value, adjusted_counts[plan])
        allocations[plan] += addable
        remainder = exact - floor_value
        remainders.append((remainder, adjusted_counts[plan], plan))

    leftover = target_total - sum(allocations.values())
    if leftover <= 0:
        return allocations

    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
    while leftover > 0:
        progressed = False
        for _, _, plan in remainders:
            if allocations[plan] < plan_counts[plan]:
                allocations[plan] += 1
                leftover -= 1
                progressed = True
                if leftover == 0:
                    break
        if not progressed:
            break

    return allocations


def sample_rows_by_plan(rows, target_total, rng, min_plan_samples):
    plan_to_rows = defaultdict(list)
    for row in rows:
        plan_to_rows[row["_plan"]].append(row)

    plan_counts = Counter({plan: len(plan_rows) for plan, plan_rows in plan_to_rows.items()})
    allocations = allocate_plan_targets(plan_counts, target_total, min_plan_samples)

    sampled_rows = []
    for plan in sorted(plan_to_rows):
        plan_rows = list(plan_to_rows[plan])
        rng.shuffle(plan_rows)
        sampled_rows.extend(plan_rows[: allocations[plan]])

    rng.shuffle(sampled_rows)
    return sampled_rows, plan_counts, Counter(allocations)


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(file_name, turn, original_count, target_count, sampled_count, plan_counts, sampled_plan_counts):
    rows = [
        {
            "file_name": file_name,
            "turn": turn,
            "plan": "__TOTAL__",
            "original_count": original_count,
            "sampled_count": sampled_count,
            "target_count": target_count,
            "original_ratio_in_file": "1.0000",
            "sampled_ratio_in_file": "1.0000" if sampled_count else "0.0000",
        }
    ]
    for plan, original_plan_count in sorted(plan_counts.items(), key=lambda item: (-item[1], item[0])):
        sampled_plan_count = sampled_plan_counts.get(plan, 0)
        rows.append(
            {
                "file_name": file_name,
                "turn": turn,
                "plan": plan,
                "original_count": original_plan_count,
                "sampled_count": sampled_plan_count,
                "target_count": target_count,
                "original_ratio_in_file": f"{original_plan_count / original_count:.4f}" if original_count else "0.0000",
                "sampled_ratio_in_file": f"{sampled_plan_count / sampled_count:.4f}" if sampled_count else "0.0000",
            }
        )
    return rows


def print_low_support_plans(per_plan_rows, threshold):
    flagged_rows = [
        row
        for row in per_plan_rows
        if row["plan"] != "__TOTAL__" and int(row["sampled_count"]) <= threshold
    ]

    if not flagged_rows:
        print(f"No sampled plans at or below {threshold} rows.")
        return

    print(f"Plans at or below {threshold} sampled rows:")
    for row in sorted(flagged_rows, key=lambda item: (item["file_name"], int(item["sampled_count"]), item["plan"])):
        print(f"  {row['file_name']}\t{row['plan']}\t{row['sampled_count']}")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    targets = dict(DEFAULT_TARGETS)
    targets.update(parse_target_overrides(args.target))

    manifest_rows = []
    per_plan_rows = []

    for file_name in sorted(targets):
        input_path = root / "datasets" / "tc" / file_name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing TC file: {input_path}")

        fieldnames, rows = read_tsv_rows(input_path)
        target_count = targets[file_name]
        original_count = len(rows)
        actual_target = min(target_count, original_count)
        turn = f"turn{file_name.split('_')[0].replace('it', '')}"

        enriched_rows = []
        for idx, row in enumerate(rows):
            copied = dict(row)
            copied["_row_order"] = idx
            copied["_plan"] = parse_plan(row.get("answer"))
            enriched_rows.append(copied)

        rng = random.Random(args.seed)
        sampled_rows, plan_counts, sampled_plan_counts = sample_rows_by_plan(
            enriched_rows,
            actual_target,
            rng,
            args.min_plan_samples,
        )
        sampled_rows.sort(key=lambda row: row["_row_order"])

        cleaned_rows = []
        for row in sampled_rows:
            copied = dict(row)
            copied.pop("_row_order", None)
            copied.pop("_plan", None)
            cleaned_rows.append(copied)

        output_path = output_dir / file_name
        write_tsv(output_path, fieldnames, cleaned_rows)

        manifest_rows.append(
            {
                "file_name": file_name,
                "turn": turn,
                "original_count": original_count,
                "target_count": target_count,
                "sampled_count": len(cleaned_rows),
                "output_path": str(output_path.relative_to(root)),
            }
        )
        per_plan_rows.extend(
            build_summary_rows(
                file_name=file_name,
                turn=turn,
                original_count=original_count,
                target_count=target_count,
                sampled_count=len(cleaned_rows),
                plan_counts=plan_counts,
                sampled_plan_counts=sampled_plan_counts,
            )
        )

    write_tsv(
        output_dir / "downsample_manifest.tsv",
        ["file_name", "turn", "original_count", "target_count", "sampled_count", "output_path"],
        manifest_rows,
    )
    write_tsv(
        output_dir / "downsample_plan_summary.tsv",
        [
            "file_name",
            "turn",
            "plan",
            "original_count",
            "sampled_count",
            "target_count",
            "original_ratio_in_file",
            "sampled_ratio_in_file",
        ],
        per_plan_rows,
    )

    total_original = sum(row["original_count"] for row in manifest_rows)
    total_sampled = sum(row["sampled_count"] for row in manifest_rows)
    print(f"Original rows across targeted files: {total_original}")
    print(f"Sampled rows across targeted files: {total_sampled}")
    print(f"Output directory: {output_dir}")
    print_low_support_plans(per_plan_rows, 3)


if __name__ == "__main__":
    main()
