import argparse
import csv
import json
import os
import random
import shutil


DEFAULT_HEADERS = [
    "conversation_history",
    "query",
    "rewrited_query",
    "answer",
    "unique_idx",
    "candidates",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a JSONL file into a new TSV file without appending"
    )
    parser.add_argument("--input", required=True, help="Source JSONL file")
    parser.add_argument("--output", required=True, help="Target TSV file to create")
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
        "candidates": str(ensure_candidates(record, api_plans)),
    }
    return {header: converted.get(header, "") for header in headers}


def main():
    args = parse_args()
    random.seed(args.seed)

    api_plans = read_api_plans(args.api)
    source_rows = read_jsonl(args.input)
    output_rows = [convert_record(record, args.headers, api_plans) for record in source_rows]

    print(f"input_rows: {len(source_rows)}")
    print(f"write_rows: {len(output_rows)}")
    print(f"output_tsv: {args.output}")

    if args.dry_run:
        return

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if args.backup and os.path.exists(args.output):
        backup_path = f"{args.output}.bak"
        shutil.copy2(args.output, backup_path)
        print(f"backup_created: {backup_path}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=args.headers, delimiter="\t")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
