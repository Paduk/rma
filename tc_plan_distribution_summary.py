#!/usr/bin/env python3
import argparse
import ast
import csv
from collections import Counter
from pathlib import Path
import re


DEFAULT_FILE_GROUPS = {
    "base": [
        "datasets/tc/it2_nonNR_tc.tsv",
        "datasets/tc/it3_nonNR_tc.tsv",
        "datasets/tc/it4_nonNR_tc.tsv",
        "datasets/tc/it5_nonNR_tc.tsv",
    ],
    "complex": [
        "datasets/tc/it3_complex_1_tc.tsv",
        "datasets/tc/it4_complex_1_tc.tsv",
        "datasets/tc/it4_complex_2_tc.tsv",
        "datasets/tc/it5_complex_1_tc.tsv",
        "datasets/tc/it5_complex_2_tc.tsv",
        "datasets/tc/it5_complex_3_tc.tsv",
    ],
}

TURN_PATTERN = re.compile(r"^it(\d+)_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize TC TSV counts by file, turn, and answer.plan."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root. Relative file paths are resolved from here.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/result/tc_plan_distribution",
        help="Directory to write TSV summaries into.",
    )
    parser.add_argument(
        "--tc-dir",
        help="If set, summarize every it*_tc.tsv in this directory instead of the default file groups.",
    )
    parser.add_argument(
        "--include-all-tc",
        action="store_true",
        help="Include every top-level it*_tc.tsv under datasets/tc in addition to the default file groups.",
    )
    return parser.parse_args()


def extract_turn_from_name(file_name):
    match = TURN_PATTERN.match(Path(file_name).name)
    if not match:
        return "unknown"
    return f"turn{match.group(1)}"


def parse_answer_plan(raw_answer):
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


def collect_default_files(root):
    seen = set()
    records = []
    for group_name, rel_paths in DEFAULT_FILE_GROUPS.items():
        for rel_path in rel_paths:
            path = (root / rel_path).resolve()
            if path in seen:
                continue
            seen.add(path)
            records.append(
                {
                    "group": group_name,
                    "relative_path": rel_path,
                    "path": path,
                    "file_name": path.name,
                    "turn": extract_turn_from_name(path.name),
                }
            )
    return records


def collect_all_tc_files(root, existing_records):
    seen = {record["path"] for record in existing_records}
    for path in sorted((root / "datasets" / "tc").glob("it*_tc.tsv")):
        resolved = path.resolve()
        if resolved in seen:
            continue
        yield {
            "group": "extra",
            "relative_path": str(path.relative_to(root)),
            "path": resolved,
            "file_name": path.name,
            "turn": extract_turn_from_name(path.name),
        }


def collect_tc_files_from_dir(root, tc_dir):
    tc_dir_path = Path(tc_dir)
    if not tc_dir_path.is_absolute():
        tc_dir_path = (root / tc_dir_path).resolve()
    else:
        tc_dir_path = tc_dir_path.resolve()

    if not tc_dir_path.exists():
        raise FileNotFoundError(f"TC directory does not exist: {tc_dir_path}")

    records = []
    for path in sorted(tc_dir_path.glob("it*_tc.tsv")):
        records.append(
            {
                "group": tc_dir_path.name,
                "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "path": path.resolve(),
                "file_name": path.name,
                "turn": extract_turn_from_name(path.name),
            }
        )
    return records


def load_rows(file_record):
    path = file_record["path"]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader, start=1):
            yield {
                "group": file_record["group"],
                "relative_path": file_record["relative_path"],
                "file_name": file_record["file_name"],
                "turn": file_record["turn"],
                "row_idx": row_idx,
                "plan": parse_answer_plan(row.get("answer")),
            }


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def ratio(count, total):
    if total == 0:
        return "0.0000"
    return f"{count / total:.4f}"


def build_outputs(records, output_dir):
    all_rows = []
    file_counts = Counter()
    turn_counts = Counter()
    group_counts = Counter()
    plan_counts = Counter()
    plan_by_file = Counter()
    plan_by_turn = Counter()
    plan_by_group = Counter()

    for record in records:
        for row in load_rows(record):
            all_rows.append(row)
            file_counts[row["file_name"]] += 1
            turn_counts[row["turn"]] += 1
            group_counts[row["group"]] += 1
            plan_counts[row["plan"]] += 1
            plan_by_file[(row["file_name"], row["plan"])] += 1
            plan_by_turn[(row["turn"], row["plan"])] += 1
            plan_by_group[(row["group"], row["plan"])] += 1

    total_count = len(all_rows)
    turn345_total = sum(turn_counts[turn] for turn in ("turn3", "turn4", "turn5"))

    summary_rows = []
    summary_rows.append(
        {
            "scope_type": "overall",
            "scope_value": "all_selected_tc",
            "count": total_count,
            "ratio_vs_parent": ratio(total_count, total_count),
        }
    )
    for group in sorted(group_counts):
        summary_rows.append(
            {
                "scope_type": "group",
                "scope_value": group,
                "count": group_counts[group],
                "ratio_vs_parent": ratio(group_counts[group], total_count),
            }
        )
    for turn in sorted(turn_counts, key=lambda x: (x == "unknown", x)):
        summary_rows.append(
            {
                "scope_type": "turn",
                "scope_value": turn,
                "count": turn_counts[turn],
                "ratio_vs_parent": ratio(turn_counts[turn], total_count),
            }
        )
    for record in sorted(records, key=lambda item: item["file_name"]):
        summary_rows.append(
            {
                "scope_type": "file",
                "scope_value": record["file_name"],
                "count": file_counts[record["file_name"]],
                "ratio_vs_parent": ratio(file_counts[record["file_name"]], total_count),
            }
        )
    for plan, count in plan_counts.most_common():
        summary_rows.append(
            {
                "scope_type": "plan",
                "scope_value": plan,
                "count": count,
                "ratio_vs_parent": ratio(count, total_count),
            }
        )

    detail_rows = []
    for record in sorted(records, key=lambda item: item["file_name"]):
        file_total = file_counts[record["file_name"]]
        file_plan_pairs = [
            (plan, count)
            for (file_name, plan), count in plan_by_file.items()
            if file_name == record["file_name"]
        ]
        for plan, count in sorted(file_plan_pairs, key=lambda item: (-item[1], item[0])):
            detail_rows.append(
                {
                    "group": record["group"],
                    "turn": record["turn"],
                    "file_name": record["file_name"],
                    "plan": plan,
                    "count": count,
                    "ratio_in_file": ratio(count, file_total),
                    "ratio_in_turn": ratio(count, turn_counts[record["turn"]]),
                    "ratio_overall": ratio(count, total_count),
                }
            )

    turn_plan_rows = []
    for turn in sorted(turn_counts, key=lambda x: (x == "unknown", x)):
        turn_total = turn_counts[turn]
        pairs = [
            (plan, count)
            for (turn_name, plan), count in plan_by_turn.items()
            if turn_name == turn
        ]
        for plan, count in sorted(pairs, key=lambda item: (-item[1], item[0])):
            turn_plan_rows.append(
                {
                    "turn": turn,
                    "plan": plan,
                    "count": count,
                    "ratio_in_turn": ratio(count, turn_total),
                    "ratio_overall": ratio(count, total_count),
                }
            )

    group_plan_rows = []
    for group in sorted(group_counts):
        group_total = group_counts[group]
        pairs = [
            (plan, count)
            for (group_name, plan), count in plan_by_group.items()
            if group_name == group
        ]
        for plan, count in sorted(pairs, key=lambda item: (-item[1], item[0])):
            group_plan_rows.append(
                {
                    "group": group,
                    "plan": plan,
                    "count": count,
                    "ratio_in_group": ratio(count, group_total),
                    "ratio_overall": ratio(count, total_count),
                }
            )

    scale_target_turns = ["turn3", "turn4", "turn5"]
    scale_target_counts = [turn_counts[turn] for turn in scale_target_turns if turn_counts[turn] > 0]
    equalized_target = min(scale_target_counts) if scale_target_counts else 0

    turn_scaling_rows = []
    for turn in ("turn2", "turn3", "turn4", "turn5"):
        count = turn_counts[turn]
        turn_scaling_rows.append(
            {
                "turn": turn,
                "count": count,
                "ratio_overall": ratio(count, total_count),
                "ratio_within_turn3_4_5_pool": ratio(count, turn345_total) if turn in {"turn3", "turn4", "turn5"} else "",
                "equalized_target_count_for_turn3_4_5": equalized_target if turn in {"turn3", "turn4", "turn5"} else "",
                "downsample_ratio_to_equalize_turn3_4_5": ratio(equalized_target, count) if turn in {"turn3", "turn4", "turn5"} and count else "",
            }
        )

    write_tsv(
        output_dir / "summary.tsv",
        ["scope_type", "scope_value", "count", "ratio_vs_parent"],
        summary_rows,
    )
    write_tsv(
        output_dir / "detail.tsv",
        ["group", "turn", "file_name", "plan", "count", "ratio_in_file", "ratio_in_turn", "ratio_overall"],
        detail_rows,
    )
    write_tsv(
        output_dir / "turn_plan_summary.tsv",
        ["turn", "plan", "count", "ratio_in_turn", "ratio_overall"],
        turn_plan_rows,
    )
    write_tsv(
        output_dir / "group_plan_summary.tsv",
        ["group", "plan", "count", "ratio_in_group", "ratio_overall"],
        group_plan_rows,
    )
    write_tsv(
        output_dir / "turn_scaling_reference.tsv",
        [
            "turn",
            "count",
            "ratio_overall",
            "ratio_within_turn3_4_5_pool",
            "equalized_target_count_for_turn3_4_5",
            "downsample_ratio_to_equalize_turn3_4_5",
        ],
        turn_scaling_rows,
    )

    return {
        "total_count": total_count,
        "turn_counts": turn_counts,
        "group_counts": group_counts,
        "file_counts": file_counts,
        "output_dir": output_dir,
    }


def print_console_summary(stats):
    print(f"Total selected TC rows: {stats['total_count']}")
    print("Turn counts:")
    for turn in ("turn2", "turn3", "turn4", "turn5"):
        print(f"  {turn}: {stats['turn_counts'][turn]}")
    print("Group counts:")
    for group in sorted(stats["group_counts"]):
        print(f"  {group}: {stats['group_counts'][group]}")
    print(f"Output directory: {stats['output_dir']}")


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()

    if args.tc_dir:
        records = collect_tc_files_from_dir(root, args.tc_dir)
    else:
        records = collect_default_files(root)
        if args.include_all_tc:
            records.extend(list(collect_all_tc_files(root, records)))

    missing_paths = [record["relative_path"] for record in records if not record["path"].exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing TC files: {', '.join(missing_paths)}")

    stats = build_outputs(records, output_dir)
    print_console_summary(stats)


if __name__ == "__main__":
    main()
