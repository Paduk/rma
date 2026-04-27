#!/usr/bin/env python3

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


DEFAULT_HOST = "http://localhost:11436"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HammerBench prompt JSONL records against an Ollama chat model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input prompt JSONL path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output prediction JSONL path. Defaults to <input>.pred.jsonl",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Ollama host URL",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=512,
        help="Maximum generation tokens",
    )
    parser.add_argument(
        "--num-parallel",
        type=int,
        default=1,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of input records to process",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a few prompts and skip Ollama requests",
    )
    parser.add_argument(
        "--dry-run-limit",
        type=int,
        default=2,
        help="Examples to print in --dry-run mode",
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


def derive_output_path(input_path):
    return input_path.with_name(f"{input_path.stem}.pred.jsonl")


def extract_first_json_object(text):
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def requires_confirmed_arguments(record):
    return record.get("mode") == "rewrite" and str(
        record.get("prompt_key") or ""
    ).startswith("structured_")


def parse_model_output(record, raw_text):
    mode = record["mode"]
    parsed = extract_first_json_object(raw_text)
    if parsed is None:
        return None, "no_json_object_found"

    if mode == "rewrite":
        rewritten_query = (
            parsed.get("rewritten_query")
            or parsed.get("rewrited_query")
            or parsed.get("query")
        )
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            return None, "missing_rewritten_query"
        confirmed_arguments = parsed.get("confirmed_arguments")
        if confirmed_arguments is None:
            if requires_confirmed_arguments(record):
                return None, "missing_confirmed_arguments"
            return {
                "rewritten_query": rewritten_query.strip(),
            }, None
        if not isinstance(confirmed_arguments, dict):
            return None, "confirmed_arguments_not_object"
        return {
            "rewritten_query": rewritten_query.strip(),
            "confirmed_arguments": confirmed_arguments,
        }, None

    name = parsed.get("name") or parsed.get("plan")
    arguments = parsed.get("arguments", {})
    if not isinstance(name, str) or not name.strip():
        return None, "missing_name"
    if not isinstance(arguments, dict):
        return None, "arguments_not_object"
    return {
        "name": name.strip(),
        "arguments": arguments,
    }, None


def request_ollama_chat(host, model, messages, temperature, num_predict, timeout):
    response = requests.post(
        f"{host.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["message"]["content"]


def predict_one(idx, record, args):
    raw_text = request_ollama_chat(
        host=args.host,
        model=args.model,
        messages=record["messages"],
        temperature=args.temperature,
        num_predict=args.num_predict,
        timeout=args.timeout,
    )
    parsed_output, parse_error = parse_model_output(record, raw_text)
    result = {
        "id": record["id"],
        "data_type": record["data_type"],
        "turn_id": record["turn_id"],
        "mode": record["mode"],
        "model": args.model,
        "parse_ok": parse_error is None,
        "parse_error": parse_error,
        "raw_response": raw_text,
        "parsed_output": parsed_output,
        "gold_call": record.get("gold_call"),
    }
    if record["mode"] == "rewrite":
        result["rewritten_query"] = (
            parsed_output.get("rewritten_query") if parsed_output else None
        )
        result["confirmed_arguments"] = (
            parsed_output.get("confirmed_arguments") if parsed_output else None
        )
    else:
        result["pred_call"] = parsed_output
    return idx, result


def main():
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    output_path = args.output or derive_output_path(args.input)

    if args.dry_run:
        for record in records[: args.dry_run_limit]:
            print(f"id\t{record['id']}")
            print(f"mode\t{record['mode']}")
            print("messages")
            print(json.dumps(record["messages"], ensure_ascii=True, indent=2))
            print("---")
        print(f"records_in_input\t{len(records)}")
        print("dry_run\tTrue")
        return

    results = [None] * len(records)
    if args.num_parallel <= 1:
        for idx, record in enumerate(records):
            _, result = predict_one(idx, record, args)
            results[idx] = result
    else:
        with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
            futures = {
                executor.submit(predict_one, idx, record, args): idx
                for idx, record in enumerate(records)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=True))
            f.write("\n")

    parse_ok_count = sum(1 for result in results if result["parse_ok"])
    print(f"input_path\t{args.input}")
    print(f"output_path\t{output_path}")
    print(f"model\t{args.model}")
    print(f"records_processed\t{len(results)}")
    print(f"parse_ok\t{parse_ok_count}")
    print(f"parse_failed\t{len(results) - parse_ok_count}")


if __name__ == "__main__":
    main()
