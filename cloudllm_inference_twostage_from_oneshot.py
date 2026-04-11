import argparse
import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "api_v3.0.1.jsonl"
DEFAULT_MODEL_NAME = "o4-mini"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a two-stage pipeline by feeding oneshot-generated rewrites into planning-only cloud inference."
    )
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="Path to a oneshot result TSV that contains generation['rewrited_query'].",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Cloud model alias for stage-2 planning. Example: o4-mini, gpt-5-mini.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to tool schema file. Supports api_v3.0.1.jsonl or simple_api.json.",
    )
    parser.add_argument(
        "--o",
        required=True,
        help="Output TSV path.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=min(8, (os.cpu_count() or 1) * 5),
        help="Maximum concurrent cloud requests.",
    )
    return parser.parse_args()


def read_apis(api_file: Path):
    with open(api_file, encoding="utf-8") as f:
        if api_file.suffix == ".jsonl":
            out = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                api_data = json.loads(line)
                for key in ("examples", "returns", "next_turn_plans"):
                    api_data.pop(key, None)
                out[api_data["plan"]] = api_data
            return out
        return json.load(f)


def build_api_str_from_candidates(candidates, apis) -> str:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list.")

    lines = []
    for plan_name in candidates:
        api_data = apis[plan_name].copy()
        lines.append(f"{plan_name}: {api_data}")
    return "\n".join(lines)


def render_messages_as_plain_text(messages, add_generation_prompt: bool) -> str:
    sections = []
    for message in messages:
        role = str(message.get("role", "")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    if add_generation_prompt:
        sections.append("Assistant:\n")
    return "\n\n".join(sections)


def print_first_inference_preview(*, script_name: str, file_name: str, prompt: str, raw: str):
    print("\n# First Inference Preview")
    print(f"script: {script_name}")
    print(f"file: {file_name}")
    print("prompt:")
    print(prompt or "<empty>")
    print()
    print("response:")
    print(raw or "<empty>")
    print()


def disable_per_request_cost_print(generate_response):
    if hasattr(generate_response, "print_cost"):
        generate_response.print_cost = False


def print_file_cumulative_cost(*, generate_response, test_key: str, file_name: str):
    total_cost = getattr(generate_response, "total_cost", None)
    if total_cost is None:
        return
    print(
        f"[Cost] test_key={test_key} file={file_name} cumulative_cost=${total_cost:.6f} USD",
        flush=True,
    )


def parse_literal(raw_value, field_name: str):
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError(f"Unsupported type for {field_name}: {type(raw_value).__name__}")
    return ast.literal_eval(raw_value)


def extract_json_from_markdown(text):
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("JSON block not found.")

        return json.loads(json_str)
    except Exception:
        return None


def parse_response_json(text):
    text = re.sub(r".*?```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Valid JSON not found")
    return json.loads(match.group())


def parse_planning_response(raw_text: str):
    try:
        parsed = ast.literal_eval(raw_text)
    except Exception:
        parsed = extract_json_from_markdown(raw_text)
        if not isinstance(parsed, dict):
            parsed = parse_response_json(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a dict.")

    result = {
        "plan": parsed.get("plan"),
        "arguments": parsed.get("arguments", {}),
    }
    if not isinstance(result["arguments"], dict):
        raise ValueError("Parsed arguments field is not a dict.")
    return result


def extract_turn_from_filename(file_name):
    match = re.search(r"it(\d+)", file_name or "")
    if not match:
        return None
    return int(match.group(1))


def compute_plan_macro(df, metric="all"):
    if df.empty:
        return float("nan")

    gt_plan_series = df["gt"].apply(
        lambda x: x.get("plan") if isinstance(x, dict) else None
    )
    metric_by_plan = (
        pd.DataFrame({"gt_plan": gt_plan_series, metric: df[metric]})
        .dropna(subset=["gt_plan"])
        .groupby("gt_plan")[metric]
        .apply(lambda sub: sub.eq("pass").mean())
    )
    return metric_by_plan.mean()


def compute_file_accuracy(df, metric="all"):
    if df.empty:
        return float("nan")
    return df[metric].eq("pass").mean()


def get_turn_file_sort_key(file_name):
    turn = extract_turn_from_filename(file_name)
    complex_match = re.search(r"complex_(\d+)", file_name or "")
    if complex_match:
        return (turn if turn is not None else 999, 0, -int(complex_match.group(1)))
    return (turn if turn is not None else 999, 1, file_name)


def print_turn_macro_summary(df, title=None, metric="all"):
    if df.empty or "turn" not in df.columns:
        return

    file_names = sorted(df["file"].dropna().unique().tolist(), key=get_turn_file_sort_key)
    rows = []
    for turn, sub in sorted(df.dropna(subset=["turn"]).groupby("turn"), key=lambda x: x[0]):
        turn_macro = compute_plan_macro(sub, metric=metric)
        row = {
            "turn": int(turn),
            "samples": len(sub),
            "plan_macro_all": round(turn_macro, 4),
        }
        turn_files = set(sub["file"].dropna().tolist())
        for file_name in file_names:
            if file_name in turn_files:
                row[file_name] = round(
                    compute_file_accuracy(sub[sub["file"] == file_name], metric=metric),
                    4,
                )
            else:
                row[file_name] = None
        rows.append(row)

    if not rows:
        return

    turn_df = pd.DataFrame(rows)
    ordered_cols = ["turn", "samples", "plan_macro_all"] + file_names
    turn_df = turn_df[ordered_cols]
    if title:
        print(f"\n# Turn-wise Plan Macro for {title}\n")
    else:
        print("\n# Turn-wise Plan Macro\n")
    print(turn_df.fillna("N/A").to_string(index=False))


def print_eval(df, title=None, test_type=None, detail=False):
    metrics = ("plan", "arguments", "all")
    if title:
        print(f"\n## Performance for {title}\n")

    metric_rows = []
    for col in metrics:
        if col == "all":
            acc = compute_plan_macro(df, metric=col)
            label = f"{col.title():<10} Macro Accuracy"
        else:
            acc = df[col].eq("pass").mean()
            label = f"{col.title():<10} Accuracy"
        metric_rows.append((label, acc))
        print(f"{label} : {acc * 100:.2f}%")

    print("-" * 40)

    log_path = PROJECT_ROOT / "logs" / "cloud_inference_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        if title:
            f.write(f"\n## Performance for {title}, {test_type}\n")
        for label, acc in metric_rows:
            f.write(f"{label} : {acc * 100:.2f}%\n")
        f.write("-" * 40 + "\n")

    if detail:
        df["gt_plan"] = df["gt"].apply(lambda x: x.get("plan"))
        detail_df = (
            df.groupby("gt_plan")[list(metrics)]
            .apply(lambda sub: sub.eq("pass").mean().round(2))
            .reset_index()
        )
        print(detail_df.to_string(index=False))
        detail_df["macro_by_plan"] = detail_df[metrics].mean(axis=1).round(2)
        print("\n# Plan Macro Accuracy")
        print(detail_df.to_string(index=False))


def init_error_log(error_log_path: Path):
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(error_log_path, "w", encoding="utf-8"):
        pass


def classify_raw_error(raw: str) -> str:
    stripped = (raw or "").strip()
    if not stripped:
        return "empty_raw"
    if "\n{" in stripped or "}\n{" in stripped:
        return "two_json_objects_newline"
    if re.search(r"\}\s*,\s*\{", stripped):
        return "two_objects_comma_split"

    missing_braces = stripped.count("{") - stripped.count("}")
    if missing_braces > 0:
        return f"truncated_missing_{missing_braces}_brace"
    if missing_braces < 0:
        return f"extra_{abs(missing_braces)}_closing_brace"

    if re.search(r'"[A-Za-z0-9_]+"\s*:\s*-?\d+"', stripped):
        return "number_then_stray_quote"
    if re.search(r'"[A-Za-z0-9_]+"\s*:\s*(true|false|null)"', stripped):
        return "bool_or_null_then_stray_quote"
    return "other_malformed_json"


def append_error_log(
    error_log_path: Path,
    *,
    test_key: str,
    file_name: str,
    row_idx: int,
    model_name: str,
    error: Exception,
    raw: str,
):
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_key": test_key,
        "file": file_name,
        "row_idx": row_idx,
        "model_name": model_name,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "raw_error_type": classify_raw_error(raw),
        "raw": raw,
    }
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False))
        f.write("\n")


def build_rewrite_system_message(api_str: str) -> str:
    return (
        "Given a user query and a list of available tools, "
        "select the most appropriate tool and generate its arguments. "
        "Only use parameter values that are explicitly stated or can be reasonably inferred "
        "from the query. Return compact JSON only with keys "
        "\"plan\" and \"arguments\". Always include both keys. The value of "
        "\"arguments\" must always be an object. If no tool matches the request, "
        "set \"plan\" to \"None\" and \"arguments\" to {}.\n"
        f"<|tool|>{api_str}<|/tool|>"
    )


def build_stage2_prompt(example, apis, stage1_rewrite: str) -> str:
    candidates = parse_literal(example["candidates"], "candidates")
    api_str = build_api_str_from_candidates(candidates, apis)
    messages = [
        {"role": "system", "content": build_rewrite_system_message(api_str)},
        {"role": "user", "content": f"User Query: {stage1_rewrite}"},
    ]
    return render_messages_as_plain_text(
        messages=messages,
        add_generation_prompt=True,
    )


def extract_stage1_rewrite(example):
    stage1_generation = parse_literal(example["generation"], "generation")
    if not isinstance(stage1_generation, dict):
        raise ValueError("generation is not a dict")

    stage1_rewrite = stage1_generation.get("rewrited_query")
    if not isinstance(stage1_rewrite, str) or not stage1_rewrite.strip():
        raise ValueError("generation['rewrited_query'] is missing")

    return stage1_generation, stage1_rewrite


def process_example(
    example,
    *,
    row_idx,
    file_name,
    apis,
    generate_response,
    model_name,
):
    prompt = ""
    raw = ""
    gt = {}
    stage1_generation = {}
    stage1_rewrite = ""
    error_payload = None

    try:
        stage1_generation, stage1_rewrite = extract_stage1_rewrite(example)
        prompt = build_stage2_prompt(example, apis, stage1_rewrite)
        response = generate_response("", [prompt])[0]
        raw = response.get("text", "")
        if not raw:
            raise ValueError("Empty response text")

        result = parse_planning_response(raw)
        gt = parse_literal(example["gt"], "gt")
        plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
        arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
        all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
    except Exception as e:
        result = {"error": str(e)}
        plan_res = "fail"
        arg_res = "fail"
        all_res = "fail"
        error_payload = {
            "test_key": example.get("test_key", "unknown"),
            "file_name": file_name,
            "row_idx": row_idx,
            "model_name": model_name,
            "error": e,
            "raw": raw,
        }

    row = {
        "row_idx": row_idx,
        "test_key": example.get("test_key"),
        "conversation_history": example.get("conversation_history"),
        "query": example.get("query"),
        "source_rewrited_query": example.get("rewrited_query"),
        "rewrited_query": stage1_rewrite,
        "candidates": example.get("candidates"),
        "stage1_raw": example.get("raw"),
        "stage1_generation": stage1_generation,
        "raw": raw,
        "generation": result,
        "gt": gt,
        "plan": plan_res,
        "arguments": arg_res,
        "all": all_res,
        "file": file_name,
        "turn": extract_turn_from_filename(file_name),
    }
    return row, error_payload, prompt, raw


def main():
    args = parse_args()
    from utils.frequently_used_tools import get_model_name

    input_path = Path(args.input_tsv)
    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = out_path.with_name(f"{out_path.stem}.errors.jsonl")
    init_error_log(error_log_path)

    apis = read_apis(Path(args.tools_path))
    model_name, generate_response = get_model_name(args.model)
    disable_per_request_cost_print(generate_response)

    print(f"input_tsv: {input_path}")
    print(f"model_name: {model_name}")
    print("prompt_style: plain_text_messages")
    print(f"error_log_path: {error_log_path}")

    df = pd.read_csv(input_path, sep="\t", dtype=str)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")

    df["_source_order"] = range(len(df))
    file_groups = (
        df[["test_key", "file"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    all_results = []
    split_results = {}
    printed_first_prompt = False
    printed_first_raw = False

    for test_key, file_name in file_groups:
        if test_key not in split_results:
            split_results[test_key] = []

        print(f"\n# Running split/file: {test_key}/{file_name}")
        file_df = df[(df["test_key"] == test_key) & (df["file"] == file_name)].copy()
        file_df = file_df.sort_values("_source_order").reset_index(drop=True)
        examples = file_df.to_dict("records")

        file_results = []
        max_workers = max(1, args.max_workers)

        if examples:
            first_row, first_error_payload, first_prompt, first_raw = process_example(
                examples[0],
                row_idx=0,
                file_name=file_name,
                apis=apis,
                generate_response=generate_response,
                model_name=model_name,
            )
            if not printed_first_prompt and not printed_first_raw:
                print_first_inference_preview(
                    script_name="cloudllm_inference_twostage_from_oneshot.py",
                    file_name=file_name,
                    prompt=first_prompt,
                    raw=first_raw,
                )
                printed_first_prompt = True
                printed_first_raw = True
            file_results.append(first_row)
            split_results[test_key].append(first_row)
            if first_error_payload is not None:
                append_error_log(error_log_path, **first_error_payload)
                print(
                    f"\n[inference_error] test_key={test_key} file={file_name} row=0",
                    flush=True,
                )
                print(f"type={type(first_error_payload['error']).__name__}", flush=True)
                print(f"message={first_error_payload['error']}", flush=True)
                if first_prompt:
                    print("prompt:", flush=True)
                    print(first_prompt, flush=True)
                if first_raw:
                    print("raw:", flush=True)
                    print(first_raw, flush=True)
                else:
                    print("raw: <empty>", flush=True)
                print(flush=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_example,
                    example,
                    row_idx=row_idx,
                    file_name=file_name,
                    apis=apis,
                    generate_response=generate_response,
                    model_name=model_name,
                ): row_idx
                for row_idx, example in enumerate(examples[1:], start=1)
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {test_key}/{file_name}",
            ):
                row, error_payload, prompt, raw = future.result()
                file_results.append(row)
                split_results[test_key].append(row)
                if error_payload is not None:
                    append_error_log(error_log_path, **error_payload)
                    print(
                        f"\n[inference_error] test_key={test_key} file={file_name} row={error_payload['row_idx']}",
                        flush=True,
                    )
                    print(f"type={type(error_payload['error']).__name__}", flush=True)
                    print(f"message={error_payload['error']}", flush=True)
                    if prompt:
                        print("prompt:", flush=True)
                        print(prompt, flush=True)
                    if raw:
                        print("raw:", flush=True)
                        print(raw, flush=True)
                    else:
                        print("raw: <empty>", flush=True)
                    print(flush=True)

        df_file = pd.DataFrame(file_results)
        if not df_file.empty:
            df_file = df_file.sort_values("row_idx").reset_index(drop=True)
        split_results[test_key] = sorted(
            split_results[test_key],
            key=lambda row: (str(row.get("file")), row.get("row_idx", 0)),
        )
        print_eval(df_file, title=f"{test_key}/{file_name}", test_type=model_name)
        print_file_cumulative_cost(
            generate_response=generate_response,
            test_key=test_key,
            file_name=file_name,
        )
        all_results.extend(df_file.to_dict("records"))

    result_df = pd.DataFrame(all_results)
    for test_key, rows in split_results.items():
        df_split = pd.DataFrame(rows)
        print_eval(df_split, title=f"split={test_key}", test_type=model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(sorted(split_results.keys()))
    print_eval(result_df, title=f"combined={combined_title}", test_type=model_name)
    print_turn_macro_summary(result_df, title=f"combined={combined_title}", metric="all")
    result_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
