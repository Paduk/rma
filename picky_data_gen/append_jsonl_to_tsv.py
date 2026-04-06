import argparse
import csv
import json
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Append generated JSONL rows into an existing TSV")
    parser.add_argument("--input", required=True, help="Source JSONL file")
    parser.add_argument("--output", required=True, help="Target TSV file to append")
    parser.add_argument("--api", default="apis/api_v3.0.1.jsonl", help="API jsonl used to build candidates if missing")
    parser.add_argument("--dedup-key", default="unique_idx", help="Row key used to avoid duplicate append")
    parser.add_argument("--backup", action="store_true", help="Create .bak copy before append")
    parser.add_argument("--dry-run", action="store_true", help="Show append count without writing")
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
    converted = {}
    converted["conversation_history"] = str(record.get("conversation_history", []))
    converted["query"] = record.get("query", "")
    converted["rewrited_query"] = record.get("rewrited_query", "")
    converted["answer"] = str(answer)
    converted["unique_idx"] = record.get("unique_idx", "")
    converted["candidates"] = str(ensure_candidates(record, api_plans))

    row = {}
    for header in headers:
        row[header] = converted.get(header, "")
    return row


def reindex_duplicate_key(key, existing_keys):
    parts = key.split("-")
    if not parts:
        return key

    last = parts[-1]
    try:
        int(last)
    except ValueError:
        candidate = f"{key}-100"
        next_num = 100
        while candidate in existing_keys:
            next_num += 1
            candidate = f"{key}-{next_num}"
        return candidate

    next_num = 100
    while True:
        candidate_parts = parts[:-1] + [str(next_num)]
        candidate = "-".join(candidate_parts)
        if candidate not in existing_keys:
            return candidate
        next_num += 1


def read_existing_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames
    return headers, rows


def main():
    args = parse_args()
    random.seed(args.seed)

    headers, existing_rows = read_existing_tsv(args.output)
    existing_keys = {row.get(args.dedup_key, "") for row in existing_rows}
    api_plans = read_api_plans(args.api)
    source_rows = read_jsonl(args.input)

    append_rows = []
    renamed_duplicates = 0

    for record in source_rows:
        key = record.get(args.dedup_key, "")
        if key in existing_keys:
            new_key = reindex_duplicate_key(key, existing_keys)
            record = dict(record)
            record[args.dedup_key] = new_key
            key = new_key
            renamed_duplicates += 1

        append_rows.append(convert_record(record, headers, api_plans))
        existing_keys.add(key)

    print(f"input_rows: {len(source_rows)}")
    print(f"append_rows: {len(append_rows)}")
    print(f"renamed_duplicates: {renamed_duplicates}")
    print(f"output_tsv: {args.output}")

    if args.dry_run:
        return

    if args.backup:
        backup_path = f"{args.output}.bak"
        shutil.copy2(args.output, backup_path)
        print(f"backup_created: {backup_path}")

    with open(args.output, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
        for row in append_rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
