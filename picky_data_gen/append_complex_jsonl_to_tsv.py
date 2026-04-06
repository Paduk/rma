import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Append complex JSONL rows into per-refered-turn TSV files")
    parser.add_argument("--input", required=True, help="Source complex JSONL file")
    parser.add_argument("--output-dir", required=True, help="Directory containing itX_complex_Y_tc.tsv files")
    parser.add_argument("--turn", type=int, required=False, help="Turn number, e.g. 5 for it5_complex_*.tsv")
    parser.add_argument("--api", default="apis/api_v3.0.1.jsonl", help="API jsonl used to build candidates if missing")
    parser.add_argument("--dedup-key", default="unique_idx", help="Row key used to avoid duplicate append")
    parser.add_argument("--backup", action="store_true", help="Create .bak copy before append")
    parser.add_argument("--dry-run", action="store_true", help="Show append count without writing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate sampling")
    return parser.parse_args()


def infer_turn_from_input(path):
    base = os.path.basename(path)
    m = re.match(r"it(\d+)_", base)
    if not m:
        raise ValueError(f"Unable to infer turn from input filename: {base}")
    return int(m.group(1))


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
        candidate = "-".join(parts[:-1] + [str(next_num)])
        if candidate not in existing_keys:
            return candidate
        next_num += 1


def read_existing_tsv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        headers = reader.fieldnames
    return headers, rows


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
    for record in source_rows:
        ref = record.get("refered_turn")
        if ref in (None, "", "NONE"):
            continue
        grouped_rows[int(ref)].append(record)

    total_append = 0
    total_renamed = 0
    touched_files = []

    for ref, records in sorted(grouped_rows.items()):
        output_path = os.path.join(args.output_dir, f"it{turn}_complex_{ref}_tc.tsv")
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Missing target TSV: {output_path}")

        headers, existing_rows = read_existing_tsv(output_path)
        existing_keys = {row.get(args.dedup_key, "") for row in existing_rows}

        append_rows = []
        renamed = 0
        for record in records:
            key = record.get(args.dedup_key, "")
            if key in existing_keys:
                record = dict(record)
                record[args.dedup_key] = reindex_duplicate_key(key, existing_keys)
                key = record[args.dedup_key]
                renamed += 1

            append_rows.append(convert_record(record, headers, api_plans))
            existing_keys.add(key)

        print(f"refered_turn={ref} output={output_path}")
        print(f"  input_rows={len(records)}")
        print(f"  append_rows={len(append_rows)}")
        print(f"  renamed_duplicates={renamed}")

        total_append += len(append_rows)
        total_renamed += renamed
        touched_files.append((output_path, headers, append_rows))

    print(f"total_append_rows={total_append}")
    print(f"total_renamed_duplicates={total_renamed}")

    if args.dry_run:
        return

    if args.backup:
        for output_path, _, _ in touched_files:
            backup_path = f"{output_path}.bak"
            shutil.copy2(output_path, backup_path)
            print(f"backup_created: {backup_path}")

    for output_path, headers, append_rows in touched_files:
        if not append_rows:
            continue
        with open(output_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            for row in append_rows:
                writer.writerow(row)


if __name__ == "__main__":
    main()
