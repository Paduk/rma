import argparse
import ast
import json
import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from train.gemma4_multitask_prompting import (
    DEFAULT_GEMMA4_PROMPT_TOKENIZER_NAME,
    render_planning_prompt,
    render_rewrite_prompt,
)
from utils.generation_backends import (
    add_text_generation_backend_args,
    build_text_generation_backend_from_args,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_API_PATH = PROJECT_ROOT / "apis" / "simple_api.json"


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Rewrite first, then run planning evaluation for Gemma 4 with shared Ollama/HF backends."
    )
    parser.add_argument(
        "--model_family",
        type=str,
        required=True,
        help="one of: gemma4, gemma4-multitask, gemma4-separate",
    )
    parser.add_argument("--test_key", type=str, required=True, help="dataset split key(s), comma separated")
    parser.add_argument("--o", type=str, required=True, help="output TSV path")
    parser.add_argument(
        "--api",
        type=str,
        default=str(DEFAULT_API_PATH),
        help="Path to simple_api.json",
    )
    add_text_generation_backend_args(
        parser,
        default_host="http://localhost:11436",
        prefix="rewrite",
        default_num_predict=200,
    )
    add_text_generation_backend_args(
        parser,
        default_host="http://localhost:11436",
        prefix="plan",
        default_num_predict=512,
    )
    parser.add_argument("--rewrite_model_name", type=str, default=None, help="Override rewrite-stage Ollama model name")
    parser.add_argument("--plan_model_name", type=str, default=None, help="Override planning-stage Ollama model name")
    parser.add_argument(
        "--prompt_tokenizer_name",
        type=str,
        default=DEFAULT_GEMMA4_PROMPT_TOKENIZER_NAME,
        help="HF tokenizer used to reproduce the Gemma 4 multitask training chat template.",
    )
    return parser


def read_apis(api_file, simple=False):
    with open(api_file, encoding="utf-8") as f:
        if simple:
            return json.load(f)
        out = {}
        for line in f:
            data = json.loads(line)
            for k in ("examples", "returns", "next_turn_plans"):
                data.pop(k, None)
            out[data["plan"]] = data
        return out

def extract_json_from_markdown(text):
    try:
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = json_match.group() if json_match else None

        if not json_str:
            raise ValueError("JSON block not found")

        return json.loads(json_str)
    except Exception:
        return None


def parse_model_output(raw):
    if raw is None:
        raise ValueError("Empty response")

    try:
        return ast.literal_eval(raw)
    except Exception:
        result = extract_json_from_markdown(raw)
        if result is None:
            raise ValueError(f"Failed to parse model output: {raw}")
        return result


def parse_rewrite_output_inference_style(raw):
    try:
        return ast.literal_eval(raw)["rewrited_query"]
    except Exception:
        return raw.split("rewrited_query")[1].split(": ")[1].split("}")[0][1:]


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
        row = {
            "turn": int(turn),
            "samples": len(sub),
            "plan_macro_all": round(compute_plan_macro(sub, metric=metric), 4),
        }
        turn_files = set(sub["file"].dropna().tolist())
        for file_name in file_names:
            if file_name in turn_files:
                row[file_name] = round(compute_file_accuracy(sub[sub["file"] == file_name], metric=metric), 4)
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
        detail_df["macro_by_plan"] = detail_df[list(metrics)].mean(axis=1).round(2)
        print("\n# Plan별 Macro Accuracy")
        print(detail_df.to_string(index=False))


def parse_test_keys(test_key_arg, data_files):
    test_keys = [key.strip() for key in test_key_arg.split(",") if key.strip()]
    if not test_keys:
        raise ValueError("`--test_key` is required. Example: --test_key base or --test_key base,complex")

    deduped = []
    for key in test_keys:
        if key not in data_files:
            raise ValueError(f"Invalid test_key: {key}. Available keys: {', '.join(sorted(data_files.keys()))}")
        if key not in deduped:
            deduped.append(key)
    return deduped


def load_prompt_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to render the Gemma 4 chat template during inference."
        ) from exc

    return AutoTokenizer.from_pretrained(model_name)


def get_model_configs(model_family):
    config = {
        "gemma4": {
            "model_name": "gemma-multitask-rma-gemma4:latest",
            "plan_model_name": "gemma-multitask-rma-gemma4:latest",
        },
        "gemma4-multitask": {
            "model_name": "gemma-multitask-rma-gemma4:latest",
            "plan_model_name": "gemma-multitask-rma-gemma4:latest",
        },
        "gemma4-separate": {
            "model_name": "gemma-4-e2b-it-rewrite-lm_only_lora:latest",
            "plan_model_name": "gemma4-rma:latest",
        },
    }
    if model_family not in config:
        raise ValueError(
            "Invalid model_family: "
            f"{model_family}. Available keys: {', '.join(sorted(config.keys()))}"
        )
    return config[model_family]


def get_data_files():
    return {
        "base": [
            "datasets/tc/scale/it2_nonNR_tc.tsv",
            "datasets/tc/scale/it3_nonNR_tc.tsv",
            "datasets/tc/scale/it4_nonNR_tc.tsv",
            "datasets/tc/scale/it5_nonNR_tc.tsv",
        ],
        "complex": [
            "datasets/tc/scale/it3_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_1_tc.tsv",
            "datasets/tc/scale/it4_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_1_tc.tsv",
            "datasets/tc/scale/it5_complex_2_tc.tsv",
            "datasets/tc/scale/it5_complex_3_tc.tsv",
        ],
        "manual": [
            "datasets/tc/manual/turn2.tsv",
            "datasets/tc/manual/turn3.tsv",
            "datasets/tc/manual/turn4.tsv",
            "datasets/tc/manual/turn5.tsv",
        ],
        "advanced_manual": [
            "datasets/tc/manual/ad/turn2.tsv",
            "datasets/tc/manual/ad/turn3.tsv",
            "datasets/tc/manual/ad/turn4.tsv",
            "datasets/tc/manual/ad/turn5.tsv",
        ],
        "manual_rewrited": [
            "datasets/tc/manual/llama_rewrited/turn2.tsv",
            "datasets/tc/manual/llama_rewrited/turn3.tsv",
            "datasets/tc/manual/llama_rewrited/turn4.tsv",
            "datasets/tc/manual/llama_rewrited/turn5.tsv",
        ],
    }


def build_rewrite_prompt(example, prompt_tokenizer):
    return render_rewrite_prompt(
        prompt_tokenizer,
        conversation_history=example["conversation_history"],
        query=example["query"],
    )


def build_plan_prompt(example, apis, prompt_tokenizer, rewritten_query):
    api_str = ""
    for plan in ast.literal_eval(example["candidates"]):
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"

    return render_planning_prompt(
        prompt_tokenizer,
        tools=api_str,
        rewritten_query=rewritten_query,
    )


def main():
    args = build_arg_parser().parse_args()
    if not args.rewrite_model and args.rewrite_model_name:
        args.rewrite_model = args.rewrite_model_name
    if not args.plan_model and args.plan_model_name:
        args.plan_model = args.plan_model_name

    model_config = get_model_configs(args.model_family)
    prompt_tokenizer = load_prompt_tokenizer(args.prompt_tokenizer_name)
    rewrite_backend = build_text_generation_backend_from_args(
        args,
        default_model_name=model_config["model_name"],
        prefix="rewrite",
    )
    plan_backend = build_text_generation_backend_from_args(
        args,
        default_model_name=model_config["plan_model_name"],
        prefix="plan",
    )
    rewrite_model_name = rewrite_backend.model_label
    plan_model_name = plan_backend.model_label
    sft_apis = read_apis(args.api, simple=True)
    data_files = get_data_files()
    test_keys = parse_test_keys(args.test_key, data_files)

    print(f"rewrite backend: {rewrite_backend.backend_name}")
    print(f"rewrite model: {rewrite_model_name}")
    if args.rewrite_backend == "ollama":
        print(f"rewrite host: {args.rewrite_host}")
    print(f"planning backend: {plan_backend.backend_name}")
    print(f"planning model: {plan_model_name}")
    if args.plan_backend == "ollama":
        print(f"planning host: {args.plan_host}")
    print(f"Selected test_keys: {test_keys}")

    all_results = []
    split_results = {key: [] for key in test_keys}

    for test_key in test_keys:
        print(f"\n# Running split: {test_key}")
        print(data_files[test_key])
        for file_path in data_files[test_key]:
            ds = load_dataset("csv", data_files={"tc": [file_path]}, delimiter="\t")["tc"]
            file_results = []

            for ex in tqdm(ds, desc=f"Processing {test_key}/{os.path.basename(file_path)}"):
                gt = ex["answer"]
                if isinstance(gt, str):
                    gt = ast.literal_eval(gt)

                rewrite_prompt = build_rewrite_prompt(ex, prompt_tokenizer)
                rewrite_raw = ""
                rewrite_result = None
                rewrite_error = ""
                rewritten_query = ""
                plan_prompt = ""
                plan_raw = ""
                plan_result = {}
                plan_res = "fail"
                arg_res = "fail"
                all_res = "fail"

                try:
                    rewrite_raw = rewrite_backend.generate_text(
                        rewrite_prompt,
                        temperature=args.rewrite_temperature,
                        num_predict=args.rewrite_num_predict,
                        response_format_json=True,
                        stop=["}"],
                    )
                    rewrite_parse_input = rewrite_raw
                    if rewrite_parse_input and not rewrite_parse_input.rstrip().endswith("}"):
                        rewrite_parse_input = rewrite_parse_input + "}"
                    rewritten_query = parse_rewrite_output_inference_style(rewrite_parse_input)
                    rewrite_result = {"rewrited_query": rewritten_query}
                    if not rewritten_query:
                        raise ValueError(f"Missing rewrited_query in rewrite result: {rewrite_raw}")
                except Exception as e:
                    rewrite_error = str(e)
                    print(f"Rewrite error: {e}, raw={rewrite_raw}")

                if not rewrite_error:
                    try:
                        plan_prompt = build_plan_prompt(
                            ex,
                            sft_apis,
                            prompt_tokenizer,
                            rewritten_query,
                        )
                        plan_raw = plan_backend.generate_text(
                            plan_prompt,
                            temperature=args.plan_temperature,
                            num_predict=args.plan_num_predict,
                            response_format_json=True,
                        )
                        plan_result = parse_model_output(plan_raw)
                        plan_res = "pass" if plan_result.get("plan") == gt.get("plan") else "fail"
                        arg_res = "pass" if plan_result.get("arguments") == gt.get("arguments") else "fail"
                        all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
                    except Exception as e:
                        plan_result = {"error": str(e)}
                        print(f"Planning error: {e}, raw={plan_raw}")

                row = {
                    "test_key": test_key,
                    "rewrite_backend_name": rewrite_backend.backend_name,
                    "rewrite_model_name": rewrite_model_name,
                    "planning_backend_name": plan_backend.backend_name,
                    "planning_model_name": plan_model_name,
                    "conversation_history": ex.get("conversation_history"),
                    "query": ex.get("query"),
                    "gt_rewrited_query": ex.get("rewrited_query"),
                    "generated_rewrited_query": rewritten_query,
                    "rewrite_prompt": rewrite_prompt,
                    "rewrite_raw": rewrite_raw,
                    "rewrite_generation": rewrite_result if rewrite_result is not None else {"error": rewrite_error, "raw": rewrite_raw},
                    "rewrite_error": rewrite_error,
                    "candidates": ex.get("candidates"),
                    "planning_prompt": plan_prompt,
                    "planning_raw": plan_raw,
                    "planning_generation": plan_result,
                    "gt": gt,
                    "plan": plan_res,
                    "arguments": arg_res,
                    "all": all_res,
                    "file": os.path.basename(file_path),
                    "turn": extract_turn_from_filename(os.path.basename(file_path)),
                }
                file_results.append(row)
                split_results[test_key].append(row)

            df_file = pd.DataFrame(file_results)
            print_eval(
                df_file,
                title=f"{test_key}/{os.path.basename(file_path)} | model={args.model_family}",
                test_type=f"{rewrite_model_name} -> {plan_model_name}",
            )
            all_results.extend(file_results)

    result = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(
            df_split,
            title=f"split={test_key} | model={args.model_family}",
            test_type=f"{rewrite_model_name} -> {plan_model_name}",
        )
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(
        result,
        title=f"combined={combined_title} | model={args.model_family}",
        test_type=f"{rewrite_model_name} -> {plan_model_name}",
    )
    print_turn_macro_summary(result, title=f"combined={combined_title}", metric="all")
    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
