#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a single per-example audit JSONL by joining HammerBench prompts, "
            "raw generations, parsed generations, gold calls, and row metrics."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "ours"],
        required=True,
        help="Run mode that produced the files.",
    )
    parser.add_argument(
        "--planner-prompts",
        type=Path,
        required=True,
        help="Planner prompt JSONL path.",
    )
    parser.add_argument(
        "--planner-preds",
        type=Path,
        required=True,
        help="Planner prediction JSONL path.",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="Per-row metric JSONL path produced by evaluate_predictions.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output audit JSONL path.",
    )
    parser.add_argument(
        "--rewrite-prompts",
        type=Path,
        default=None,
        help="Rewrite prompt JSONL path for --mode ours.",
    )
    parser.add_argument(
        "--rewrite-preds",
        type=Path,
        default=None,
        help="Rewrite prediction JSONL path for --mode ours.",
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


def read_required_jsonl(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return read_jsonl(path)


def read_optional_jsonl(path, label):
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return read_jsonl(path)


def by_id(rows):
    return {row["id"]: row for row in rows}


def ordered_ids(*row_lists):
    ids = []
    seen = set()
    for rows in row_lists:
        for row in rows:
            row_id = row["id"]
            if row_id not in seen:
                ids.append(row_id)
                seen.add(row_id)
    return ids


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
    name = call.get("name") or call.get("plan")
    if isinstance(name, str):
        name = name.strip()
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return {
        "name": name,
        "arguments": normalize_value(arguments),
    }


def compact_prompt(prompt):
    if not isinstance(prompt, dict):
        return None
    return {
        "mode": prompt.get("mode"),
        "prompt_key": prompt.get("prompt_key"),
        "rewritten_query_field": prompt.get("rewritten_query_field"),
        "confirmed_arguments_field": prompt.get("confirmed_arguments_field"),
        "messages": prompt.get("messages"),
        "system_instruction": prompt.get("system_instruction"),
        "user_input": prompt.get("user_input"),
    }


def compact_prediction(pred):
    if not isinstance(pred, dict):
        return None

    pred_call = pred.get("pred_call")
    if not isinstance(pred_call, dict):
        pred_call = pred.get("parsed_output")

    payload = {
        "model": pred.get("model"),
        "parse_ok": pred.get("parse_ok"),
        "parse_error": pred.get("parse_error"),
        "raw_response": pred.get("raw_response"),
        "parsed_output": pred.get("parsed_output"),
        "pred_call": pred_call,
        "pred_call_normalized": normalize_call(pred_call),
        "finish_reason": pred.get("finish_reason"),
        "prompt_tokens": pred.get("prompt_tokens"),
        "completion_tokens": pred.get("completion_tokens"),
    }
    if "rewritten_query" in pred:
        payload["rewritten_query"] = pred.get("rewritten_query")
    if "confirmed_arguments" in pred:
        payload["confirmed_arguments"] = pred.get("confirmed_arguments")
    return payload


def compact_metrics(row):
    if not isinstance(row, dict):
        return None
    return {
        "parse_ok": row.get("parse_ok"),
        "gold_name": row.get("gold_name"),
        "pred_name": row.get("pred_name"),
        "api_correct": row.get("api_correct"),
        "arg_exact_match": row.get("arg_exact_match"),
        "arg_precision": row.get("arg_precision"),
        "arg_recall": row.get("arg_recall"),
        "arg_f1": row.get("arg_f1"),
        "gold_arg_count": row.get("gold_arg_count"),
        "pred_arg_count": row.get("pred_arg_count"),
        "matched_arg_count": row.get("matched_arg_count"),
        "final_call_exact_match": row.get("final_call_exact_match"),
    }


def first_present(*rows):
    for row in rows:
        if isinstance(row, dict):
            return row
    return {}


def build_audit_record(
    row_id,
    mode,
    planner_prompt,
    planner_pred,
    metrics,
    rewrite_prompt=None,
    rewrite_pred=None,
):
    source = first_present(planner_prompt, planner_pred, rewrite_prompt, rewrite_pred, metrics)
    gold_raw = None
    for row in (planner_prompt, planner_pred, rewrite_prompt, rewrite_pred):
        if isinstance(row, dict) and isinstance(row.get("gold_call"), dict):
            gold_raw = row["gold_call"]
            break

    record = {
        "id": row_id,
        "data_type": source.get("data_type"),
        "turn_id": source.get("turn_id"),
        "mode": mode,
        "gold_raw": gold_raw,
        "gold_parsed": normalize_call(gold_raw),
        "planner": {
            "prompt": compact_prompt(planner_prompt),
            "generation": compact_prediction(planner_pred),
        },
        "metrics": compact_metrics(metrics),
    }

    if mode == "ours":
        record["rewrite"] = {
            "prompt": compact_prompt(rewrite_prompt),
            "generation": compact_prediction(rewrite_pred),
        }
        rewritten_query = None
        if isinstance(rewrite_pred, dict):
            rewritten_query = rewrite_pred.get("rewritten_query")
        record["model_rewritten_query"] = rewritten_query
        confirmed_arguments = None
        if isinstance(rewrite_pred, dict):
            confirmed_arguments = rewrite_pred.get("confirmed_arguments")
        record["model_confirmed_arguments"] = confirmed_arguments

    return record


def main():
    args = parse_args()
    if args.mode == "ours" and (args.rewrite_prompts is None or args.rewrite_preds is None):
        raise ValueError("--rewrite-prompts and --rewrite-preds are required for --mode ours")

    planner_prompts = read_required_jsonl(args.planner_prompts, "planner prompts")
    planner_preds = read_required_jsonl(args.planner_preds, "planner predictions")
    metric_rows = read_required_jsonl(args.rows, "metric rows")
    rewrite_prompts = read_optional_jsonl(args.rewrite_prompts, "rewrite prompts")
    rewrite_preds = read_optional_jsonl(args.rewrite_preds, "rewrite predictions")

    planner_prompt_map = by_id(planner_prompts)
    planner_pred_map = by_id(planner_preds)
    metric_map = by_id(metric_rows)
    rewrite_prompt_map = by_id(rewrite_prompts)
    rewrite_pred_map = by_id(rewrite_preds)

    # Audit rows should represent records actually sent to a model. Prompt files
    # may contain the full eval set even when --limit was used for generation.
    ids = ordered_ids(rewrite_preds, planner_preds, metric_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row_id in ids:
            record = build_audit_record(
                row_id=row_id,
                mode=args.mode,
                planner_prompt=planner_prompt_map.get(row_id),
                planner_pred=planner_pred_map.get(row_id),
                metrics=metric_map.get(row_id),
                rewrite_prompt=rewrite_prompt_map.get(row_id),
                rewrite_pred=rewrite_pred_map.get(row_id),
            )
            f.write(json.dumps(record, ensure_ascii=True))
            f.write("\n")

    print(f"mode\t{args.mode}")
    print(f"output_path\t{args.output}")
    print(f"audit_rows\t{len(ids)}")
    print(f"planner_prompt_rows\t{len(planner_prompts)}")
    print(f"planner_prediction_rows\t{len(planner_preds)}")
    print(f"metric_rows\t{len(metric_rows)}")
    if args.mode == "ours":
        print(f"rewrite_prompt_rows\t{len(rewrite_prompts)}")
        print(f"rewrite_prediction_rows\t{len(rewrite_preds)}")


if __name__ == "__main__":
    main()
