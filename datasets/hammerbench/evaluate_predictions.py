#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HammerBench planner prediction JSONL against gold function calls."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Prediction JSONL path produced by run_prompt_jsonl_ollama.py",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional JSON summary output path",
    )
    parser.add_argument(
        "--rows-out",
        type=Path,
        help="Optional per-row metrics JSONL output path",
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


def normalize_value(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(val) for key, val in sorted(value.items())}
    return value


def normalize_call(call):
    if not isinstance(call, dict):
        return {"name": None, "arguments": {}}
    name = call.get("name")
    if isinstance(name, str):
        name = name.strip()
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": name,
        "arguments": normalize_value(arguments),
    }


def make_argument_items(arguments):
    if not isinstance(arguments, dict):
        return set()
    items = set()
    for key, value in sorted(arguments.items()):
        normalized = normalize_value(value)
        items.add((key, json.dumps(normalized, ensure_ascii=True, sort_keys=True)))
    return items


def compute_argument_metrics(gold_arguments, pred_arguments):
    gold_items = make_argument_items(gold_arguments)
    pred_items = make_argument_items(pred_arguments)
    matched = len(gold_items & pred_items)

    if not pred_items and not gold_items:
        precision = recall = f1 = 1.0
    else:
        precision = matched / len(pred_items) if pred_items else 0.0
        recall = matched / len(gold_items) if gold_items else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

    return {
        "gold_arg_count": len(gold_items),
        "pred_arg_count": len(pred_items),
        "matched_arg_count": matched,
        "arg_precision": precision,
        "arg_recall": recall,
        "arg_f1": f1,
        "arg_exact_match": gold_items == pred_items,
    }


def select_pred_call(row):
    if isinstance(row.get("pred_call"), dict):
        return row["pred_call"]
    if isinstance(row.get("parsed_output"), dict):
        return row["parsed_output"]
    return None


def evaluate_row(row):
    gold = normalize_call(row.get("gold_call"))
    pred = normalize_call(select_pred_call(row))
    parse_ok = bool(row.get("parse_ok"))
    api_correct = parse_ok and pred["name"] == gold["name"]
    arg_metrics = compute_argument_metrics(gold["arguments"], pred["arguments"])
    final_call_exact_match = api_correct and arg_metrics["arg_exact_match"]

    return {
        "id": row.get("id"),
        "data_type": row.get("data_type"),
        "turn_id": row.get("turn_id"),
        "mode": row.get("mode"),
        "model": row.get("model"),
        "parse_ok": parse_ok,
        "gold_name": gold["name"],
        "pred_name": pred["name"],
        "api_correct": api_correct,
        "arg_exact_match": arg_metrics["arg_exact_match"],
        "arg_precision": arg_metrics["arg_precision"],
        "arg_recall": arg_metrics["arg_recall"],
        "arg_f1": arg_metrics["arg_f1"],
        "gold_arg_count": arg_metrics["gold_arg_count"],
        "pred_arg_count": arg_metrics["pred_arg_count"],
        "matched_arg_count": arg_metrics["matched_arg_count"],
        "final_call_exact_match": final_call_exact_match,
    }


def compute_summary(rows):
    count = len(rows)
    parse_ok_count = sum(1 for row in rows if row["parse_ok"])
    parsed_rows = [row for row in rows if row["parse_ok"]]

    def avg(field, subset):
        if not subset:
            return 0.0
        return sum(row[field] for row in subset) / len(subset)

    return {
        "rows": count,
        "parse_ok_count": parse_ok_count,
        "parse_ok_rate": avg("parse_ok", rows),
        "api_accuracy": avg("api_correct", rows),
        "api_accuracy_on_parsed": avg("api_correct", parsed_rows),
        "argument_exact_match": avg("arg_exact_match", rows),
        "argument_exact_match_on_parsed": avg("arg_exact_match", parsed_rows),
        "argument_slot_precision": avg("arg_precision", rows),
        "argument_slot_recall": avg("arg_recall", rows),
        "argument_slot_f1": avg("arg_f1", rows),
        "final_call_exact_match": avg("final_call_exact_match", rows),
        "final_call_exact_match_on_parsed": avg("final_call_exact_match", parsed_rows),
    }


def print_summary_block(title, summary):
    print(title)
    for key in (
        "rows",
        "parse_ok_count",
        "parse_ok_rate",
        "api_accuracy",
        "api_accuracy_on_parsed",
        "argument_exact_match",
        "argument_exact_match_on_parsed",
        "argument_slot_precision",
        "argument_slot_recall",
        "argument_slot_f1",
        "final_call_exact_match",
        "final_call_exact_match_on_parsed",
    ):
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}\t{value:.4f}")
        else:
            print(f"{key}\t{value}")


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    raw_rows = read_jsonl(args.input)
    if not raw_rows:
        raise ValueError(f"No rows found in {args.input}")

    modes = {row.get("mode") for row in raw_rows}
    if modes == {"rewrite"}:
        raise ValueError(
            "This evaluator is for planner predictions, not rewrite-only predictions."
        )

    evaluated_rows = [evaluate_row(row) for row in raw_rows]
    overall_summary = compute_summary(evaluated_rows)

    by_type = defaultdict(list)
    for row in evaluated_rows:
        by_type[row["data_type"]].append(row)
    by_type_summary = {
        data_type: compute_summary(rows)
        for data_type, rows in sorted(by_type.items())
    }

    print(f"input_path\t{args.input}")
    print(f"modes\t{','.join(sorted(str(mode) for mode in modes if mode is not None))}")
    print_summary_block("overall", overall_summary)
    for data_type, summary in by_type_summary.items():
        print_summary_block(f"by_type={data_type}", summary)

    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_path": str(args.input),
            "modes": sorted(str(mode) for mode in modes if mode is not None),
            "overall": overall_summary,
            "by_type": by_type_summary,
        }
        args.summary_out.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.rows_out:
        args.rows_out.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_out.open("w", encoding="utf-8") as f:
            for row in evaluated_rows:
                f.write(json.dumps(row, ensure_ascii=True))
                f.write("\n")


if __name__ == "__main__":
    main()
