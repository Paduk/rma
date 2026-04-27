#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge rewrite prediction JSONL into canonical eval JSONL records so "
            "planner-from-rewrite prompts can be generated."
        )
    )
    parser.add_argument(
        "--eval",
        type=Path,
        required=True,
        help="Canonical eval JSONL path",
    )
    parser.add_argument(
        "--rewrite-preds",
        type=Path,
        required=True,
        help="Rewrite prediction JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output merged eval JSONL path. Defaults to <eval>.with_model_rewrite.jsonl",
    )
    parser.add_argument(
        "--field-name",
        default="model_rewritten_query",
        help="Field name to store the merged rewritten query",
    )
    parser.add_argument(
        "--confirmed-arguments-field-name",
        default="model_confirmed_arguments",
        help="Field name to store optional structured rewrite confirmed arguments",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def derive_output_path(eval_path):
    return eval_path.with_name(f"{eval_path.stem}.with_model_rewrite.jsonl")


def main():
    args = parse_args()
    if not args.eval.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval}")
    if not args.rewrite_preds.exists():
        raise FileNotFoundError(f"Rewrite prediction file not found: {args.rewrite_preds}")

    eval_records = read_jsonl(args.eval)
    rewrite_preds = read_jsonl(args.rewrite_preds)
    pred_map = {row["id"]: row for row in rewrite_preds}

    merged = []
    matched = 0
    usable = 0
    usable_confirmed_arguments = 0
    for record in eval_records:
        copied = dict(record)
        pred = pred_map.get(record["id"])
        if pred is not None:
            matched += 1
        rewritten_query = None
        confirmed_arguments = None
        if pred is not None and pred.get("parse_ok") and pred.get("rewritten_query"):
            rewritten_query = pred["rewritten_query"]
            usable += 1
            if isinstance(pred.get("confirmed_arguments"), dict):
                confirmed_arguments = pred["confirmed_arguments"]
                usable_confirmed_arguments += 1
        copied[args.field_name] = rewritten_query
        copied[args.confirmed_arguments_field_name] = confirmed_arguments
        merged.append(copied)

    output_path = args.output or derive_output_path(args.eval)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")

    print(f"eval_path\t{args.eval}")
    print(f"rewrite_preds_path\t{args.rewrite_preds}")
    print(f"output_path\t{output_path}")
    print(f"field_name\t{args.field_name}")
    print(f"confirmed_arguments_field_name\t{args.confirmed_arguments_field_name}")
    print(f"eval_records\t{len(eval_records)}")
    print(f"matched_predictions\t{matched}")
    print(f"usable_rewritten_queries\t{usable}")
    print(f"usable_confirmed_arguments\t{usable_confirmed_arguments}")


if __name__ == "__main__":
    main()
