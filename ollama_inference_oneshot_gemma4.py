import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from generation_backends import (
    add_text_generation_backend_args,
    build_text_generation_backend_from_args,
)
from oneshot_qwen_prompt import (
    build_oneshot_messages,
    render_chat_template,
    resolve_prompt_tokenizer_name,
)


DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_HOST = "http://localhost:11437"
DEFAULT_MODEL_NAME = "gemma-oneshot-rma-gemma4:latest"
DEFAULT_PROMPT_TOKENIZER_NAME = "google/gemma-4-E2B-it"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Gemma 4 one-shot RMA models with shared Ollama/HF generation backends."
        )
    )
    add_text_generation_backend_args(parser, default_host=DEFAULT_HOST)
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Legacy Ollama model name alias. If --model is unset, this value is used for --backend ollama.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--prompt_tokenizer_name",
        default=DEFAULT_PROMPT_TOKENIZER_NAME,
        help="HF tokenizer used to render the chat template.",
    )
    parser.add_argument(
        "--test_key",
        required=True,
        help="Comma-separated test split keys. Example: base,complex",
    )
    parser.add_argument(
        "--o",
        required=True,
        help="Output TSV path.",
    )
    return parser.parse_args()


def read_simple_apis(api_file: Path):
    with open(api_file, encoding="utf-8") as f:
        return json.load(f)


def extract_turn_from_filename(file_name):
    match = re.search(r"it(\d+)", file_name)
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
    complex_match = re.search(r"complex_(\d+)", file_name)
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


def parse_test_keys(test_key_arg, data_files):
    test_keys = [key.strip() for key in test_key_arg.split(",") if key.strip()]
    if not test_keys:
        raise ValueError("No valid test_key values were provided.")

    deduped = []
    for key in test_keys:
        if key not in data_files:
            raise ValueError(
                f"Invalid test_key: {key}. Available keys: {', '.join(sorted(data_files.keys()))}"
            )
        if key not in deduped:
            deduped.append(key)
    return deduped


def build_data_files():
    return {
        "base": [
            "datasets/tc/scale/it2_nonNR_tc.tsv",
            # "datasets/tc/scale/it3_nonNR_tc.tsv",
            # "datasets/tc/scale/it4_nonNR_tc.tsv",
            # "datasets/tc/scale/it5_nonNR_tc.tsv",
        ],
        "complex": [
            "datasets/tc/scale/it3_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_1_tc.tsv",
            "datasets/tc/scale/it5_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_3_tc.tsv",
        ],
        "swap": [
            "datasets/tc/it2_nonNR_tc_swapped_backup.tsv",
            "datasets/tc/it5_nonNR_tc_swapped_backup.tsv",
        ],
    }


def parse_literal(raw_value, field_name: str):
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError(f"Unsupported type for {field_name}: {type(raw_value).__name__}")
    return ast.literal_eval(raw_value)


def load_prompt_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to render the chat template during inference."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def build_oneshot_prompt(example, apis, prompt_tokenizer, prompt_model_name: str) -> str:
    candidates = parse_literal(example["candidates"], "candidates")
    messages = build_oneshot_messages(
        conversation_history=example["conversation_history"],
        query=example["query"],
        candidates=candidates,
        apis=apis,
    )
    return render_chat_template(
        tokenizer=prompt_tokenizer,
        messages=messages,
        add_generation_prompt=True,
        model_name=prompt_model_name,
    )


def extract_json_from_markdown(text):
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("JSON 블록을 찾을 수 없습니다.")

        return json.loads(json_str)

    except Exception as e:
        print("JSON 추출/파싱 실패:", e)
        return None


def parse_oneshot_response(raw_text: str):
    try:
        parsed = ast.literal_eval(raw_text)
    except Exception:
        parsed = extract_json_from_markdown(raw_text)

    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a dict.")

    result = {
        "rewrited_query": parsed.get("rewrited_query"),
        "plan": parsed.get("plan"),
        "arguments": parsed.get("arguments", {}),
    }
    if not isinstance(result["arguments"], dict):
        raise ValueError("Parsed arguments field is not a dict.")
    return result


def main():
    args = parse_args()
    if not args.model:
        args.model = args.model_name

    apis = read_simple_apis(Path(args.tools_path))
    generation_backend = build_text_generation_backend_from_args(
        args,
        default_model_name=DEFAULT_MODEL_NAME,
    )
    active_model_name = generation_backend.model_label
    prompt_tokenizer_name = resolve_prompt_tokenizer_name(
        model_name=args.model or args.model_name or DEFAULT_MODEL_NAME,
        prompt_tokenizer_name=args.prompt_tokenizer_name,
    )
    prompt_tokenizer = load_prompt_tokenizer(prompt_tokenizer_name)
    data_files = build_data_files()
    test_keys = parse_test_keys(args.test_key, data_files)
    print(f"backend: {generation_backend.backend_name}")
    print(f"generation_model: {active_model_name}")
    if args.backend == "ollama":
        print(f"host: {args.host}")
    print(f"prompt_tokenizer_name: {prompt_tokenizer_name}")
    print(f"test_keys: {test_keys}")

    all_results = []
    split_results = {key: [] for key in test_keys}
    printed_first_prompt = False
    printed_first_raw = False

    for test_key in test_keys:
        print(f"\n# Running split: {test_key}")
        print(data_files[test_key])
        for rel_path in data_files[test_key]:
            file_path = PROJECT_ROOT / rel_path
            df = pd.read_csv(file_path, sep="\t", dtype=str)
            examples = df.to_dict("records")

            file_results = []

            for row_idx, example in enumerate(
                tqdm(examples, desc=f"Processing {test_key}/{file_path.name}")
            ):
                prompt = ""
                raw = ""
                gt = {}

                try:
                    prompt = build_oneshot_prompt(
                        example,
                        apis,
                        prompt_tokenizer=prompt_tokenizer,
                        prompt_model_name=prompt_tokenizer_name,
                    )
                    if not printed_first_prompt:
                        print("first_inference_prompt:")
                        print(prompt)
                        print()
                        printed_first_prompt = True

                    raw = generation_backend.generate_text(
                        prompt=prompt,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
                        response_format_json=True,
                    )
                    if not printed_first_raw:
                        print("first_inference_raw:")
                        print(raw)
                        print()
                        printed_first_raw = True

                    result = parse_oneshot_response(raw)

                    gt = parse_literal(example["answer"], "answer")
                    plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                    arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                    all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
                except Exception as e:
                    result = {"error": str(e)}
                    print(
                        f"\n[inference_error] test_key={test_key} file={file_path.name} row={row_idx}",
                        flush=True,
                    )
                    print(f"type={type(e).__name__}", flush=True)
                    print(f"message={e}", flush=True)
                    if prompt:
                        print("prompt:", flush=True)
                        print(prompt, flush=True)
                    if raw:
                        print("raw:", flush=True)
                        print(raw, flush=True)
                    else:
                        print("raw: <empty>", flush=True)
                    print(flush=True)
                    plan_res = "fail"
                    arg_res = "fail"
                    all_res = "fail"

                row = {
                    "test_key": test_key,
                    "backend_name": generation_backend.backend_name,
                    "model_name": active_model_name,
                    "conversation_history": example.get("conversation_history"),
                    "query": example.get("query"),
                    "rewrited_query": example.get("rewrited_query"),
                    "candidates": example.get("candidates"),
                    "raw": raw,
                    "generation": result,
                    "gt": gt,
                    "plan": plan_res,
                    "arguments": arg_res,
                    "all": all_res,
                    "file": file_path.name,
                    "turn": extract_turn_from_filename(file_path.name),
                }
                file_results.append(row)
                split_results[test_key].append(row)

            df_file = pd.DataFrame(file_results)
            print_eval(df_file, title=f"{test_key}/{file_path.name}", test_type=active_model_name)
            all_results.extend(file_results)

    result_df = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=active_model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(result_df, title=f"combined={combined_title}", test_type=active_model_name)
    print_turn_macro_summary(result_df, title=f"combined={combined_title}", metric="all")

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
