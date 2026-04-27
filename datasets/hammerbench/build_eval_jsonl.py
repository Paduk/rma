#!/usr/bin/env python3

import argparse
import copy
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a HammerBench subset JSON array into canonical eval JSONL records "
            "for baseline/rewrite/planner experiments."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input HammerBench subset JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output eval JSONL path. Defaults to <input>.eval.jsonl",
    )
    return parser.parse_args()


def parse_sample_id(sample_id):
    try:
        data_type, conversation_id, turn_id = sample_id.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Unexpected HammerBench id format: {sample_id}") from exc
    return data_type, conversation_id, int(turn_id)


def sanitize_tools(tools):
    sanitized = copy.deepcopy(tools)
    for tool in sanitized:
        parameters = tool.get("parameters")
        if isinstance(parameters, dict):
            parameters.pop("required", None)
    return sanitized


def derive_default_output_path(input_path):
    if input_path.suffix == ".json":
        return input_path.with_suffix(".eval.jsonl")
    return input_path.with_name(f"{input_path.name}.eval.jsonl")


def build_record(row):
    messages = row["messages"]
    if len(messages) < 2:
        raise ValueError(f"Sample {row['id']} has too few messages: {len(messages)}")
    if messages[-1]["role"] != "function call":
        raise ValueError(f"Sample {row['id']} last message is not a function call")
    if messages[-2]["role"] != "user":
        raise ValueError(f"Sample {row['id']} current message is not a user turn")

    data_type, conversation_id, turn_id = parse_sample_id(row["id"])
    return {
        "id": row["id"],
        "data_type": data_type,
        "conversation_id": conversation_id,
        "turn_id": turn_id,
        "prior_history": messages[:-2],
        "current_user_utterance": messages[-2]["content"],
        "planner_history": messages[:-1],
        "tool_schema": sanitize_tools(row["tools"]),
        "gold_call": messages[-1]["content"],
        "oracle_rewritten_query": None,
    }


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    output_path = args.output or derive_default_output_path(args.input)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    records = [build_record(row) for row in rows]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")

    type_counts = Counter(record["data_type"] for record in records)
    turn_ids_ok = all(record["turn_id"] >= 2 for record in records)
    gold_names_ok = all(isinstance(record["gold_call"].get("name"), str) for record in records)
    no_required_ok = all(
        "required" not in tool.get("parameters", {})
        for record in records
        for tool in record["tool_schema"]
    )

    print(f"input_path\t{args.input}")
    print(f"output_path\t{output_path}")
    print(f"records_written\t{len(records)}")
    print(f"validation_turn_id_present\t{turn_ids_ok}")
    print(f"validation_gold_call_name_present\t{gold_names_ok}")
    print(f"validation_required_removed\t{no_required_ok}")
    for data_type in sorted(type_counts):
        print(f"type_stats\t{data_type}\tcount={type_counts[data_type]}")


if __name__ == "__main__":
    main()
