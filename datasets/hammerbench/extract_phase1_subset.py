#!/usr/bin/env python3

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract a HammerBench multi-turn subset by data type and minimum turn id, "
            "and print validation stats."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/hj153lee/new-rma/datasets/hammerbench/en/multi-turn.json"),
        help="Path to HammerBench multi-turn.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/hj153lee/new-rma/datasets/hammerbench/en/"
            "multi-turn.phase1_external_mQsA.turn_gt_1.json"
        ),
        help="Path to write the filtered subset JSON",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=["External", "mQsA"],
        help="HammerBench data types to keep",
    )
    parser.add_argument(
        "--min-turn",
        type=int,
        default=2,
        help="Keep samples with turn-id >= this value",
    )
    return parser.parse_args()


def parse_sample_id(sample_id):
    try:
        data_type, conversation_id, turn_id = sample_id.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Unexpected HammerBench id format: {sample_id}") from exc
    return data_type, conversation_id, int(turn_id)


def normalize_type_name(data_type):
    return data_type.lower().replace("_", "-")


def write_json(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    keep_types = set(args.types)
    filtered_rows = []
    rows_by_type = defaultdict(list)
    snapshots_by_type = Counter()
    conversations_by_type = defaultdict(set)

    for row in rows:
        data_type, conversation_id, turn_id = parse_sample_id(row["id"])
        if data_type not in keep_types or turn_id < args.min_turn:
            continue
        filtered_rows.append(row)
        rows_by_type[data_type].append(row)
        snapshots_by_type[data_type] += 1
        conversations_by_type[data_type].add(conversation_id)

    write_json(args.output, filtered_rows)

    total_conversations = len(
        {
            (parse_sample_id(row["id"])[0], parse_sample_id(row["id"])[1])
            for row in filtered_rows
        }
    )
    all_turns_valid = all(parse_sample_id(row["id"])[2] >= args.min_turn for row in filtered_rows)
    only_target_types = all(parse_sample_id(row["id"])[0] in keep_types for row in filtered_rows)

    print(f"input_path\t{args.input}")
    print(f"output_path\t{args.output}")
    print(f"target_types\t{','.join(args.types)}")
    print(f"min_turn\t{args.min_turn}")
    print(f"rows_in_input\t{len(rows)}")
    print(f"rows_in_output\t{len(filtered_rows)}")
    print(f"unique_conversations\t{total_conversations}")
    print(f"validation_all_turns_gte_min\t{all_turns_valid}")
    print(f"validation_only_target_types\t{only_target_types}")

    for data_type in sorted(snapshots_by_type):
        per_type_output = args.output.with_name(
            f"{args.output.stem}.{normalize_type_name(data_type)}{args.output.suffix}"
        )
        write_json(per_type_output, rows_by_type[data_type])
        print(
            "type_stats\t"
            f"{data_type}\t"
            f"output_path={per_type_output}\t"
            f"snapshots={snapshots_by_type[data_type]}\t"
            f"conversations={len(conversations_by_type[data_type])}"
        )


if __name__ == "__main__":
    main()
