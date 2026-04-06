import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict


DEFAULT_HEADERS = [
    "conversation_history",
    "query",
    "rewrited_query",
    "answer",
    "unique_idx",
    "refered_turn",
    "candidates",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a complex JSONL file into new per-refered-turn TSV files"
    )
    parser.add_argument("--input", required=True, help="Source complex JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory to write itX_complex_Y_tc.tsv files")
    parser.add_argument("--turn", type=int, required=False, help="Turn number, e.g. 5 for it5_complex_*.tsv")
    parser.add_argument(
        "--api",
        default="apis/api_v3.0.1.jsonl",
        help="API jsonl used to build candidates if missing",
    )
    parser.add_argument(
        "--headers",
        nargs="+",
        default=DEFAULT_HEADERS,
        help="TSV header order to write",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .bak copy first when output file already exists",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show write count without writing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate sampling")
    return parser.parse_args()


def infer_turn_from_input(path):
    base = os.path.basename(path)
    match = re.match(r"it(\d+)_", base)
    if not match:
        raise ValueError(f"Unable to infer turn from input filename: {base}")
    return int(match.group(1))


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_api_plans(path):
    plans = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                plans.append(json.loads(line)["plan"])
    return plans


def ensure_candidates(record, api_plans):
    if "candidates" in record and record["candidates"]:
        return record["candidates"]

    answer = record.get("answer", {})
    plan = answer.get("plan")
    if not plan:
        return []

    others = [p for p in api_plans if p != plan]
    sample_size = min(4, len(others))
    sampled = random.sample(others, sample_size)
    sampled.append(plan)
    random.shuffle(sampled)
    return sampled


def convert_record(record, headers, api_plans):
    answer = record.get("answer", {})
    converted = {
        "conversation_history": str(record.get("conversation_history", [])),
        "query": record.get("query", ""),
        "rewrited_query": record.get("rewrited_query", ""),
        "answer": str(answer),
        "unique_idx": record.get("unique_idx", ""),
        "refered_turn": str(record.get("refered_turn", "")),
        "candidates": str(ensure_candidates(record, api_plans)),
    }
    return {header: converted.get(header, "") for header in headers}


def main():
    args = parse_args()
    random.seed(args.seed)

    turn = args.turn if args.turn is not None else infer_turn_from_input(args.input)
    api_plans = read_api_plans(args.api)
    source_rows = read_jsonl(args.input)

    grouped_rows = defaultdict(list)
    skipped_rows = 0
    for record in source_rows:
        ref = record.get("refered_turn")
        if ref in (None, "", "NONE"):
            skipped_rows += 1
            continue
        grouped_rows[int(ref)].append(record)

    print(f"input_rows: {len(source_rows)}")
    print(f"skipped_rows_without_refered_turn: {skipped_rows}")

    touched_files = []
    for ref, records in sorted(grouped_rows.items()):
        output_path = os.path.join(args.output_dir, f"it{turn}_complex_{ref}_tc.tsv")
        output_rows = [convert_record(record, args.headers, api_plans) for record in records]
        print(f"refered_turn={ref} output={output_path}")
        print(f"  write_rows={len(output_rows)}")
        touched_files.append((output_path, output_rows))

    if args.dry_run:
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for output_path, output_rows in touched_files:
        if args.backup and os.path.exists(output_path):
            backup_path = f"{output_path}.bak"
            shutil.copy2(output_path, backup_path)
            print(f"backup_created: {backup_path}")

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=args.headers, delimiter="\t")
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()
