import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from train.llama_prompts import (
    SFT_HISTORY_INFERENCE_LLAMA,
    SFT_HISTORY_INFERENCE_PHI4,
    SFT_HISTORY_INFERENCE_QWEN25,
    SFT_HISTORY_INFERENCE_QWEN3,
    SFT_REWRITE_INFERENCE_LLAMA,
    SFT_REWRITE_INFERENCE_PHI4,
    SFT_REWRITE_INFERENCE_QWEN25,
    SFT_REWRITE_INFERENCE_QWEN3,
    ZERO_HISTORY_INFERENCE_LLAMA,
    ZERO_HISTORY_INFERENCE_PHI4,
    ZERO_HISTORY_INFERENCE_QWEN25,
    ZERO_HISTORY_INFERENCE_QWEN3,
    ZERO_REWRITE_INFERENCE_LLAMA,
    ZERO_REWRITE_INFERENCE_PHI4,
    ZERO_REWRITE_INFERENCE_QWEN25,
    ZERO_REWRITE_INFERENCE_QWEN3,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_HOST = "http://localhost:11436"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stage-2 planning by feeding oneshot result TSV rewrites into the "
            "same prompt/model presets used by ollama_inference_multi.py."
        )
    )
    parser.add_argument(
        "--input_tsv",
        required=True,
        help="Path to a oneshot result TSV.",
    )
    parser.add_argument(
        "--t",
        required=True,
        help="Planning preset key, reusing ollama_inference_multi.py test_type_config.",
    )
    parser.add_argument(
        "--o",
        required=True,
        help="Output TSV path.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json or api jsonl.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Ollama.",
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=512,
        help="Maximum generation tokens for Ollama.",
    )
    parser.add_argument(
        "--rewrite_source",
        choices=["auto", "column", "generation"],
        default="auto",
        help=(
            "Which rewrite to feed into stage-2 planning. "
            "'generation' uses generation['rewrited_query'], "
            "'column' uses the top-level rewrited_query column, "
            "'auto' prefers generation then falls back to the column."
        ),
    )
    return parser.parse_args()


def read_apis(api_file: Path):
    with open(api_file, encoding="utf-8") as f:
        if api_file.suffix == ".json":
            return json.load(f)

        out = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for key in ("examples", "returns", "next_turn_plans"):
                data.pop(key, None)
            out[data["plan"]] = data
        return out


def parse_literal(raw_value, field_name: str):
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError(f"Unsupported type for {field_name}: {type(raw_value).__name__}")
    return ast.literal_eval(raw_value)


def parse_response_json(text):
    text = re.sub(r".*?```(?:json)?\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"\s*```.*", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Valid JSON not found")
    return json.loads(match.group())


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


def generate_text(prompt, model, host, temperature=0.0, num_predict=512):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "format": "json",
                "num_predict": num_predict,
            },
            "stream": False,
        },
        timeout=300,
    )

    if response.status_code == 200:
        data = response.json()
        return data["response"]
    raise RuntimeError(f"API request failed: {response.text}")


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

    log_path = PROJECT_ROOT / "logs" / "ollama_inference_log.txt"
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
        print("\n# Plan별 Macro Accuracy")
        print(detail_df.to_string(index=False))


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


def build_test_type_config():
    return {
        "base-phi4": {
            "model_name": "phi4-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "rewrite-phi4": {
            "model_name": "phi4-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_PHI4,
            "prompt_mode": "rewrite",
        },
        "base-llama3": {
            "model_name": "llama3-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "rewrite-llama3": {
            "model_name": "llama3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
            "prompt_mode": "rewrite",
        },
        "base-qwen3": {
            "model_name": "qwen3-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3": {
            "model_name": "qwen3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-1.7b": {
            "model_name": "qwen3-base-1.7b:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-1.7b": {
            "model_name": "qwen3-rewrite-1.7b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-0.6b": {
            "model_name": "qwen3-base-0.6b:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-0.6b": {
            "model_name": "qwen3-rewrite-0.6b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "base-qwen2.5": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        },
        "rewrite-qwen2.5": {
            "model_name": "qwen2.5-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
            "prompt_mode": "rewrite",
        },
        "base-qwen2.5-base": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        },
        "rewrite-qwen2.5-base": {
            "model_name": "qwen2.5-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_QWEN25,
            "prompt_mode": "rewrite",
        },
        "base-phi4-base": {
            "model_name": "phi4-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "rewrite-phi4-base": {
            "model_name": "phi4-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_PHI4,
            "prompt_mode": "rewrite",
        },
        "base-llama3-base": {
            "model_name": "llama-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "rewrite-llama3-base": {
            "model_name": "llama-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_LLAMA,
            "prompt_mode": "rewrite",
        },
        "base-qwen3-base": {
            "model_name": "qwen3-base:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "rewrite-qwen3-base": {
            "model_name": "qwen3-base:latest",
            "prompt_template": ZERO_REWRITE_INFERENCE_QWEN3,
            "prompt_mode": "rewrite",
        },
        "new-base-qwen3": {
            "model_name": "qwen3-qwen-history-all_linear:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_QWEN3,
            "prompt_mode": "base",
        },
        "new-base-phi4": {
            "model_name": "phi4-phi-history-1st:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e4": {
            "model_name": "phi4-phi-history-1st-e4:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e5": {
            "model_name": "phi4-phi-history-1st-e5:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-phi4-e6": {
            "model_name": "phi4-phi-history-1st:latest",
            "prompt_template": SFT_HISTORY_INFERENCE_PHI4,
            "prompt_mode": "base",
        },
        "new-base-llama3": {
            "model_name": "llama3-llama-history-all_linear:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_LLAMA,
            "prompt_mode": "base",
        },
        "new-base-qwen2.5": {
            "model_name": "qwen25-qwen-history-all_linear:latest",
            "prompt_template": ZERO_HISTORY_INFERENCE_QWEN25,
            "prompt_mode": "base",
        },
    }


def normalize_test_key(raw_value):
    value = str(raw_value).strip() if raw_value is not None else ""
    return value or "input"


def normalize_file_name(raw_value, input_path: Path):
    value = str(raw_value).strip() if raw_value is not None else ""
    return value or input_path.name


def resolve_rewrite_query(example, rewrite_source):
    source_rewrite = example.get("rewrited_query")
    if isinstance(source_rewrite, str):
        source_rewrite = source_rewrite.strip()

    stage1_generation = {}
    generation_error = None
    raw_generation = example.get("generation")
    if raw_generation not in (None, ""):
        try:
            parsed_generation = parse_literal(raw_generation, "generation")
            if not isinstance(parsed_generation, dict):
                raise ValueError("generation is not a dict")
            stage1_generation = parsed_generation
        except Exception as exc:
            generation_error = exc

    generated_rewrite = stage1_generation.get("rewrited_query")
    if isinstance(generated_rewrite, str):
        generated_rewrite = generated_rewrite.strip()

    if rewrite_source == "column":
        if source_rewrite:
            return stage1_generation, source_rewrite, "column"
        raise ValueError("Top-level rewrited_query column is missing")

    if rewrite_source == "generation":
        if generated_rewrite:
            return stage1_generation, generated_rewrite, "generation"
        if generation_error is not None:
            raise generation_error
        raise ValueError("generation['rewrited_query'] is missing")

    if generated_rewrite:
        return stage1_generation, generated_rewrite, "generation"
    if source_rewrite:
        return stage1_generation, source_rewrite, "column"
    if generation_error is not None:
        raise generation_error
    raise ValueError("No usable rewrited_query found in generation or top-level column")


def build_prompt_and_metadata(example, *, apis, prompt_template, prompt_mode, rewrite_source):
    candidates = parse_literal(example["candidates"], "candidates")
    api_str = ""
    for plan in candidates:
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"

    stage1_generation = {}
    rewrite_query = example.get("rewrited_query")
    rewrite_source_used = "column"

    if prompt_mode == "base":
        prompt = prompt_template.format(
            tools=api_str,
            conversation_history=example["conversation_history"],
            data=example["query"],
        )
    elif prompt_mode == "rewrite":
        stage1_generation, rewrite_query, rewrite_source_used = resolve_rewrite_query(
            example,
            rewrite_source=rewrite_source,
        )
        prompt = prompt_template.format(
            tools=api_str,
            data=rewrite_query,
        )
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    gt = parse_literal(example["gt"], "gt")
    if not isinstance(gt, dict):
        raise ValueError("gt is not a dict")

    return {
        "prompt": prompt,
        "gt": gt,
        "candidates": candidates,
        "stage1_generation": stage1_generation,
        "rewrite_query": rewrite_query,
        "rewrite_source_used": rewrite_source_used,
    }


def main():
    args = parse_args()
    input_path = Path(args.input_tsv).expanduser().resolve()
    out_path = Path(args.o).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = build_test_type_config().get(args.t)
    if config is None:
        valid_test_types = ", ".join(sorted(build_test_type_config().keys()))
        raise ValueError(f"Invalid test type: {args.t}. Available test types: {valid_test_types}")

    model_name = config["model_name"]
    prompt_template = config["prompt_template"]
    prompt_mode = config["prompt_mode"]
    apis = read_apis(Path(args.tools_path).expanduser().resolve())

    print(f"input_tsv: {input_path}")
    print(f"model_name: {model_name}")
    print(f"host: {args.host}")
    print(f"prompt_mode: {prompt_mode}")
    print(f"rewrite_source: {args.rewrite_source}")

    df = pd.read_csv(input_path, sep="\t", dtype=str)
    if df.empty:
        raise ValueError(f"No rows found in {input_path}")

    required_columns = {"candidates", "gt"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in input TSV: {', '.join(missing_columns)}")

    if prompt_mode == "base":
        for column in ("conversation_history", "query"):
            if column not in df.columns:
                raise ValueError(f"Missing required column for base mode: {column}")

    df["_source_order"] = range(len(df))
    df["_normalized_test_key"] = (
        df["test_key"].apply(normalize_test_key)
        if "test_key" in df.columns
        else "input"
    )
    df["_normalized_file"] = (
        df["file"].apply(lambda value: normalize_file_name(value, input_path))
        if "file" in df.columns
        else input_path.name
    )

    file_groups = (
        df[["_normalized_test_key", "_normalized_file"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )

    all_results = []
    split_results = {}

    for test_key, file_name in file_groups:
        if test_key not in split_results:
            split_results[test_key] = []

        print(f"\n# Running split/file: {test_key}/{file_name}")
        file_df = df[
            (df["_normalized_test_key"] == test_key) & (df["_normalized_file"] == file_name)
        ].copy()
        file_df = file_df.sort_values("_source_order").reset_index(drop=True)
        examples = file_df.to_dict("records")

        file_results = []

        for row_idx, example in enumerate(
            tqdm(examples, desc=f"Processing {test_key}/{file_name}")
        ):
            prompt = ""
            raw = ""
            gt = {}
            result = {}
            stage1_generation = {}
            rewrite_query = example.get("rewrited_query")
            rewrite_source_used = "column"

            try:
                prepared = build_prompt_and_metadata(
                    example,
                    apis=apis,
                    prompt_template=prompt_template,
                    prompt_mode=prompt_mode,
                    rewrite_source=args.rewrite_source,
                )
                prompt = prepared["prompt"]
                gt = prepared["gt"]
                stage1_generation = prepared["stage1_generation"]
                rewrite_query = prepared["rewrite_query"]
                rewrite_source_used = prepared["rewrite_source_used"]

                if row_idx == 0:
                    print(prompt)

                raw = generate_text(
                    prompt=prompt,
                    model=model_name,
                    host=args.host,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                )
                result = parse_planning_response(raw)

                plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
            except Exception as e:
                result = {"error": str(e)}
                print(f"Error: {e}, {raw}")
                plan_res = "fail"
                arg_res = "fail"
                all_res = "fail"

            row = {
                "row_idx": row_idx,
                "test_key": test_key,
                "conversation_history": example.get("conversation_history"),
                "query": example.get("query"),
                "source_rewrited_query": example.get("rewrited_query"),
                "rewrited_query": rewrite_query,
                "rewrite_source": rewrite_source_used,
                "candidates": example.get("candidates"),
                "stage1_generation": stage1_generation,
                "generation": result,
                "gt": gt,
                "plan": plan_res,
                "arguments": arg_res,
                "all": all_res,
                "file": file_name,
                "turn": extract_turn_from_filename(file_name),
            }
            file_results.append(row)
            split_results[test_key].append(row)

        df_file = pd.DataFrame(file_results)
        if not df_file.empty:
            df_file = df_file.sort_values("row_idx").reset_index(drop=True)
        print_eval(df_file, title=f"{test_key}/{file_name}", test_type=model_name)
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
