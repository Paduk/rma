import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from train.llama_prompts import SFT_REWRITE_INFERENCE_PHI4


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "result" / "oneshot_test"
DEFAULT_API_FILE = BASE_DIR / "apis" / "simple_api.json"
DEFAULT_LOG_FILE = BASE_DIR / "logs" / "ollama_inference_log.txt"
MODEL_NAME = "phi4-rewrite:latest"
PROMPT_TEMPLATE = SFT_REWRITE_INFERENCE_PHI4


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate oneshot rewrite datasets with phi4-rewrite.")
    parser.add_argument("--input_file", type=str, required=True, help="Input TSV file path")
    parser.add_argument("--o", type=str, default="", help="Output TSV path")
    parser.add_argument("--host", type=str, default="http://localhost:11436", help="Ollama host URL")
    parser.add_argument(
        "--api_file",
        type=str,
        default=str(DEFAULT_API_FILE),
        help="API definition JSON file",
    )
    return parser.parse_args()


def read_apis(api_file: Path):
    with open(api_file, encoding="utf-8") as f:
        return json.load(f)


def append_log_line(log_file: Path, text: str):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def parse_literal(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception:
            return text


def pick_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return default
    if pd.isna(value):
        return default
    if isinstance(value, str) and not value.strip():
        return default
    return value


def extract_json_from_markdown(text):
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    json_match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not json_match:
        raise ValueError("JSON block not found")
    return json.loads(json_match.group())


def parse_model_output(raw):
    for parser in (ast.literal_eval, json.loads, extract_json_from_markdown):
        try:
            parsed = parser(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError("Failed to parse model output as dict")


def generate_text(prompt, *, model, host):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "format": "json",
                "num_predict": 512,
            },
            "stream": False,
        },
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.text}")
    return response.json()["response"]


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


def print_eval(df, *, title, log_file, test_type):
    metrics = ("plan", "arguments", "all")
    metric_rows = []

    print(f"\n## Performance for {title}\n")
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

    append_log_line(log_file, f"\n## Performance for {title}, {test_type}")
    for label, acc in metric_rows:
        append_log_line(log_file, f"{label} : {acc * 100:.2f}%")
    append_log_line(log_file, "-" * 40)


def print_turn_macro_summary(df, *, title):
    if df.empty or "turn" not in df.columns:
        return

    file_names = sorted(df["source_file"].dropna().unique().tolist(), key=get_turn_file_sort_key)
    rows = []
    for turn, sub in sorted(df.dropna(subset=["turn"]).groupby("turn"), key=lambda x: x[0]):
        turn_macro = compute_plan_macro(sub, metric="all")
        row = {
            "turn": int(turn),
            "samples": len(sub),
            "plan_macro_all": round(turn_macro, 4),
        }
        source_files = set(sub["source_file"].dropna().tolist())
        for file_name in file_names:
            if file_name in source_files:
                row[file_name] = round(compute_file_accuracy(sub[sub["source_file"] == file_name], metric="all"), 4)
            else:
                row[file_name] = None
        rows.append(row)

    if not rows:
        return

    print(f"\n# Turn-wise Plan Macro for {title}\n")
    turn_df = pd.DataFrame(rows)
    ordered_cols = ["turn", "samples", "plan_macro_all"] + file_names
    print(turn_df[ordered_cols].fillna("N/A").to_string(index=False))


def build_api_str(row, apis):
    candidates = parse_literal(row.get("candidates"))

    if isinstance(candidates, str):
        candidates = [candidates]

    selected_plans = []
    if isinstance(candidates, list):
        selected_plans = [plan for plan in candidates if isinstance(plan, str) and plan in apis]

    if not selected_plans:
        selected_plans = list(apis.keys())

    lines = [f"{plan}: {apis[plan]}" for plan in selected_plans]
    return "\n".join(lines) + "\n"


def build_prompt(row, apis):
    api_str = build_api_str(row, apis)
    rewrited_query = pick_value(row.get("rewrited_query"))
    if rewrited_query is None:
        raise ValueError("`rewrited_query` is missing")
    return PROMPT_TEMPLATE.format(
        tools=api_str,
        data=rewrited_query,
    )


def resolve_output_path(input_file: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    return DEFAULT_OUTPUT_DIR / f"{input_file.stem}.tsv"


def evaluate_file(df, *, dataset_name, apis, host):
    file_results = []
    prompt_preview_printed = False

    for row_idx, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}")
    ):
        raw = ""
        gt = {}
        test_key = pick_value(row.get("test_key"), "unknown")
        source_file = pick_value(row.get("file"), dataset_name)
        turn = pick_value(row.get("turn"))

        try:
            prompt = build_prompt(row, apis)
            if not prompt_preview_printed:
                print(prompt)
                prompt_preview_printed = True

            raw = generate_text(prompt, model=MODEL_NAME, host=host)
            result = parse_model_output(raw)

            gt = parse_literal(row.get("gt"))
            if not isinstance(gt, dict):
                raise ValueError("`gt` must be a dict-like value")

            plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
            arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
            all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
        except Exception as e:
            result = {"error": str(e)}
            print(f"Error in {dataset_name} row {row_idx}: {e}")
            plan_res = "fail"
            arg_res = "fail"
            all_res = "fail"

        file_results.append(
            {
                "dataset_name": dataset_name,
                "test_key": test_key,
                "conversation_history": row.get("conversation_history"),
                "query": row.get("query"),
                "rewrited_query": row.get("rewrited_query"),
                "candidates": row.get("candidates"),
                "generation": result,
                "gt": gt,
                "plan": plan_res,
                "arguments": arg_res,
                "all": all_res,
                "source_file": source_file,
                "turn": extract_turn_from_filename(str(source_file)) if turn is None else turn,
            }
        )

    return file_results


def main():
    args = parse_args()

    input_file = Path(args.input_file)
    output_path = resolve_output_path(input_file, args.o)
    api_file = Path(args.api_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not api_file.exists():
        raise FileNotFoundError(f"API file not found: {api_file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    apis = read_apis(api_file)

    print(MODEL_NAME)
    print(args.host)
    print(f"input_file: {input_file}")
    print(f"output_file: {output_path}")

    print(f"\n# Running dataset: {input_file.name}")
    df = pd.read_csv(input_file, sep="\t", keep_default_na=False)
    dataset_results = evaluate_file(
        df,
        dataset_name=input_file.name,
        apis=apis,
        host=args.host,
    )

    dataset_df = pd.DataFrame(dataset_results)
    print_eval(dataset_df, title=f"dataset={input_file.name}", log_file=DEFAULT_LOG_FILE, test_type=MODEL_NAME)
    print_turn_macro_summary(dataset_df, title=f"dataset={input_file.name}")

    for test_key, rows in sorted(dataset_df.groupby("test_key"), key=lambda x: x[0]):
        split_df = pd.DataFrame(rows)
        print_eval(split_df, title=f"split={test_key}", log_file=DEFAULT_LOG_FILE, test_type=MODEL_NAME)
        print_turn_macro_summary(split_df, title=f"split={test_key}")

    dataset_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
