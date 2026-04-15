import argparse
import ast
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"
TRAIN_DIR = PROJECT_ROOT / "train"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from oneshot_qwen_prompt import (
    build_oneshot_messages,
    render_chat_template,
    render_messages_as_plain_text,
    resolve_prompt_tokenizer_name,
)
from rma_model_profiles import RMA_MODEL_PROFILES

DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_HOST = "http://localhost:11436"
DEFAULT_MODEL_NAME = "qwen3-oneshot:latest"
DEFAULT_PROMPT_TOKENIZER_NAME = None
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
GLM_STOP_SEQUENCES = [
    "<|observation|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]
PROFILE_STOP_SEQUENCES = {
    "glm-edge-4b": GLM_STOP_SEQUENCES,
}
ONESHOT_SYSTEM_PROMPT = (
    "Given a conversation history, a user query, and a list of available tools, "
    "first rewrite the query by resolving ambiguous references using the "
    "conversation history. Then select the most appropriate tool and generate "
    "its arguments. Return compact JSON only with keys "
    "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
    "keys. The value of \"arguments\" must always be an object. If no tool "
    "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}."
)
"""
python3 /home/hj153lee/RMA/ollama_inference_oneshot.py \
--model_name phi4-oneshot-e6:latest \
--test_key base,complex \
--o /home/hj153lee/RMA/datasets/result/260406-ablation/phi4-e6-oneshot.tsv \
--host http://localhost:21436
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one-shot RMA models via Ollama with model-matched chat templates."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RMA_MODEL_PROFILES.keys()),
        default=None,
        help=(
            "Named model profile. If provided, uses the profile HF tokenizer for prompt "
            "rendering and infers the default Ollama one-shot model name unless "
            "--model_name is also set."
        ),
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Ollama model name to query.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Ollama host URL.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--prompt_tokenizer_name",
        default=DEFAULT_PROMPT_TOKENIZER_NAME,
        help=(
            "HF tokenizer used to render the chat template. If omitted, use --profile "
            "when present; otherwise infer it from --model_name."
        ),
    )
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for profile tokenizers without chat_template.",
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
    parser.add_argument(
        "--num_predict",
        type=int,
        default=512,
        help="Maximum generation tokens for Ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Ollama.",
    )
    parser.add_argument(
        "--multi",
        "--num_parallel",
        dest="num_parallel",
        nargs="?",
        type=int,
        const=4,
        default=1,
        help=(
            "Number of concurrent Ollama requests. Omit or set to 1 for the "
            "current sequential behavior; use --multi 3~5 for parallel calls."
        ),
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


def infer_model_slug(model_name: str) -> str:
    lower_name = model_name.lower()
    if "qwen3" in lower_name:
        return "qwen3"
    if "qwen2.5" in lower_name or "qwen25" in lower_name:
        return "qwen25"
    if "phi-4" in lower_name or "phi4" in lower_name:
        return "phi4"
    if "llama" in lower_name:
        return "llama3"
    if "gemma" in lower_name:
        return "gemma"
    sanitized = re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return sanitized or "model"


def infer_oneshot_ollama_model_name(prompt_model_name: str) -> str:
    return f"{infer_model_slug(prompt_model_name)}-oneshot:latest"


def infer_profile_from_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    lower_name = model_name.lower()
    for profile_name, profile in sorted(
        RMA_MODEL_PROFILES.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        profile_key = profile_name.lower()
        model_slug = infer_model_slug(profile.model_name)
        model_tail = re.sub(
            r"[^a-z0-9]+",
            "-",
            profile.model_name.split("/")[-1].lower(),
        ).strip("-")
        match_keys = [profile_key, model_tail]
        if profile_name in {"qwen", "phi", "llama", "gemma"} or model_slug not in {
            "qwen3",
            "qwen25",
            "phi4",
            "llama3",
            "gemma",
        }:
            match_keys.append(model_slug)
        if any(match_key and match_key in lower_name for match_key in match_keys):
            return profile_name
    return None


def resolve_prompt_model_name(args, model_name: str) -> str:
    if args.prompt_tokenizer_name:
        return args.prompt_tokenizer_name
    if args.profile:
        return RMA_MODEL_PROFILES[args.profile].model_name
    return resolve_prompt_tokenizer_name(
        model_name=model_name,
        prompt_tokenizer_name=args.prompt_tokenizer_name,
    )


def resolve_ollama_model_name(args, prompt_model_name: str) -> str:
    if args.model_name:
        return args.model_name
    if args.profile:
        return infer_oneshot_ollama_model_name(prompt_model_name)
    return DEFAULT_MODEL_NAME


def get_profile_stop_sequences(profile: str | None):
    if not profile:
        return None
    return PROFILE_STOP_SEQUENCES.get(profile)


def truncate_at_stop_markers(text: str, stop_markers=None) -> str:
    if not text or not stop_markers:
        return text
    stop_positions = [
        text.find(marker)
        for marker in stop_markers
        if marker and text.find(marker) >= 0
    ]
    if not stop_positions:
        return text
    return text[: min(stop_positions)]


def is_llama_prompt_model(model_name: str) -> bool:
    return model_name == LLAMA_MODEL_NAME


def build_llama_prompts(
    api_str: str,
    conversation_history: str,
    query: str,
    target_json: str,
):
    user_content = (
        f"Tools: {api_str}\n"
        f"Conversation History: {conversation_history}\n"
        f"User Query: {query}"
    )
    prompt_prefix = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{ONESHOT_SYSTEM_PROMPT}\n"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )
    prompt = f"{prompt_prefix}{target_json}<|eot_id|>\n"
    return prompt_prefix, prompt


def render_model_messages(
    tokenizer,
    messages,
    add_generation_prompt: bool,
    model_name: str,
    chat_template_fallback: str,
):
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer=tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            model_name=model_name,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "for base-model one-shot SFT."
        )

    return render_messages_as_plain_text(messages, add_generation_prompt=add_generation_prompt)


def build_oneshot_prompt(
    example,
    apis,
    prompt_tokenizer,
    prompt_model_name: str,
    chat_template_fallback: str,
    source_file: str | None = None,
) -> str:
    candidates = parse_literal(example["candidates"], "candidates")
    api_str = "\n".join(f"{plan_name}: {apis[plan_name].copy()}" for plan_name in candidates)
    if is_llama_prompt_model(prompt_model_name):
        prompt_prefix, _ = build_llama_prompts(
            api_str=api_str,
            conversation_history=example["conversation_history"],
            query=example["query"],
            target_json="",
        )
        return prompt_prefix

    messages = build_oneshot_messages(
        conversation_history=example["conversation_history"],
        query=example["query"],
        candidates=candidates,
        apis=apis,
    )
    return render_model_messages(
        tokenizer=prompt_tokenizer,
        messages=messages,
        add_generation_prompt=True,
        model_name=prompt_model_name,
        chat_template_fallback=chat_template_fallback,
    )


def generate_text(prompt, model, host, temperature=0.0, num_predict=512, stop=None):
    options = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if stop:
        options["stop"] = stop
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "options": options,
            "stream": False,
        },
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.text}")
    return response.json()["response"]


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


def parse_oneshot_response(raw_text: str, stop_markers=None):
    parse_text = truncate_at_stop_markers(raw_text, stop_markers).strip()
    try:
        parsed = ast.literal_eval(parse_text)
    except Exception:
        parsed = extract_json_from_markdown(parse_text)

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


def build_oneshot_result_row(
    test_key,
    example,
    raw,
    result,
    gt,
    plan_res,
    arg_res,
    all_res,
    file_name,
):
    return {
        "test_key": test_key,
        "conversation_history": example.get("conversation_history"),
        "query": example.get("query"),
        "rewrited_query": example.get("rewrited_query"),
        "candidates": example.get("candidates"),
        "raw_generation": raw,
        "parse_error": result.get("error") if isinstance(result, dict) else None,
        "generation": result,
        "gt": gt,
        "plan": plan_res,
        "arguments": arg_res,
        "all": all_res,
        "file": file_name,
        "turn": extract_turn_from_filename(file_name),
    }


def build_oneshot_error_info(test_key, file_name, row_idx, exc, prompt, raw):
    return {
        "test_key": test_key,
        "file_name": file_name,
        "row_idx": row_idx,
        "type": type(exc).__name__,
        "message": str(exc),
        "prompt": prompt,
        "raw": raw,
    }


def print_oneshot_inference_error(error_info):
    print(
        (
            f"\n[inference_error] test_key={error_info['test_key']} "
            f"file={error_info['file_name']} row={error_info['row_idx']}"
        ),
        flush=True,
    )
    print(f"type={error_info['type']}", flush=True)
    print(f"message={error_info['message']}", flush=True)
    if error_info["prompt"]:
        print("prompt:", flush=True)
        print(error_info["prompt"], flush=True)
    if error_info["raw"]:
        print("raw:", flush=True)
        print(error_info["raw"], flush=True)
    else:
        print("raw: <empty>", flush=True)
    print(flush=True)


def evaluate_oneshot_example(
    row_idx,
    example,
    prompt,
    test_key,
    file_name,
    model_name,
    host,
    temperature,
    num_predict,
    stop_markers,
):
    raw = ""
    gt = {}
    error_info = None

    try:
        raw = generate_text(
            prompt=prompt,
            model=model_name,
            host=host,
            temperature=temperature,
            num_predict=num_predict,
            stop=stop_markers,
        )
        result = parse_oneshot_response(raw, stop_markers=stop_markers)

        gt = parse_literal(example["answer"], "answer")
        plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
        arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
        all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
    except Exception as e:
        result = {"error": str(e)}
        error_info = build_oneshot_error_info(
            test_key=test_key,
            file_name=file_name,
            row_idx=row_idx,
            exc=e,
            prompt=prompt,
            raw=raw,
        )
        plan_res = "fail"
        arg_res = "fail"
        all_res = "fail"

    row = build_oneshot_result_row(
        test_key=test_key,
        example=example,
        raw=raw,
        result=result,
        gt=gt,
        plan_res=plan_res,
        arg_res=arg_res,
        all_res=all_res,
        file_name=file_name,
    )
    return row_idx, row, error_info


def main():
    args = parse_args()
    if args.num_parallel < 1:
        raise ValueError("--multi/--num_parallel must be >= 1.")

    apis = read_simple_apis(Path(args.tools_path))
    if not args.profile and args.model_name:
        args.profile = infer_profile_from_model_name(args.model_name)
    bootstrap_model_name = args.model_name or DEFAULT_MODEL_NAME
    prompt_model_name = resolve_prompt_model_name(args, model_name=bootstrap_model_name)
    model_name = resolve_ollama_model_name(args, prompt_model_name=prompt_model_name)
    stop_markers = get_profile_stop_sequences(args.profile)
    prompt_tokenizer = load_prompt_tokenizer(prompt_model_name)
    data_files = build_data_files()
    test_keys = parse_test_keys(args.test_key, data_files)
    print(f"profile: {args.profile}")
    print(f"model_name: {model_name}")
    print(f"host: {args.host}")
    print(f"prompt_tokenizer_name: {prompt_model_name}")
    print(f"chat_template_fallback: {args.chat_template_fallback}")
    print(f"stop_markers: {stop_markers}")
    print(f"test_keys: {test_keys}")
    if args.num_parallel > 1:
        print(f"num_parallel: {args.num_parallel}")

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

            if args.num_parallel <= 1:
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
                            prompt_model_name=prompt_model_name,
                            chat_template_fallback=args.chat_template_fallback,
                            source_file=file_path.name,
                        )
                        if not printed_first_prompt:
                            print("first_inference_prompt:")
                            print(prompt)
                            print()
                            printed_first_prompt = True

                        raw = generate_text(
                            prompt=prompt,
                            model=model_name,
                            host=args.host,
                            temperature=args.temperature,
                            num_predict=args.num_predict,
                            stop=stop_markers,
                        )
                        if not printed_first_raw:
                            print("first_inference_raw:")
                            print(raw)
                            print()
                            printed_first_raw = True

                        result = parse_oneshot_response(raw, stop_markers=stop_markers)

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
                        "conversation_history": example.get("conversation_history"),
                        "query": example.get("query"),
                        "rewrited_query": example.get("rewrited_query"),
                        "candidates": example.get("candidates"),
                        "raw_generation": raw,
                        "parse_error": result.get("error") if isinstance(result, dict) else None,
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
            else:
                prepared_examples = []
                rows_by_idx = {}
                error_infos = {}

                for row_idx, example in enumerate(
                    tqdm(examples, desc=f"Preparing {test_key}/{file_path.name}")
                ):
                    prompt = ""
                    try:
                        prompt = build_oneshot_prompt(
                            example,
                            apis,
                            prompt_tokenizer=prompt_tokenizer,
                            prompt_model_name=prompt_model_name,
                            chat_template_fallback=args.chat_template_fallback,
                            source_file=file_path.name,
                        )
                        if not printed_first_prompt:
                            print("first_inference_prompt:")
                            print(prompt)
                            print()
                            printed_first_prompt = True
                    except Exception as e:
                        rows_by_idx[row_idx] = build_oneshot_result_row(
                            test_key=test_key,
                            example=example,
                            raw="",
                            result={"error": str(e)},
                            gt={},
                            plan_res="fail",
                            arg_res="fail",
                            all_res="fail",
                            file_name=file_path.name,
                        )
                        error_infos[row_idx] = build_oneshot_error_info(
                            test_key=test_key,
                            file_name=file_path.name,
                            row_idx=row_idx,
                            exc=e,
                            prompt=prompt,
                            raw="",
                        )
                        continue

                    prepared_examples.append((row_idx, example, prompt))

                with ThreadPoolExecutor(max_workers=args.num_parallel) as executor:
                    futures = [
                        executor.submit(
                            evaluate_oneshot_example,
                            row_idx,
                            example,
                            prompt,
                            test_key,
                            file_path.name,
                            model_name,
                            args.host,
                            args.temperature,
                            args.num_predict,
                            stop_markers,
                        )
                        for row_idx, example, prompt in prepared_examples
                    ]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Processing {test_key}/{file_path.name}",
                    ):
                        row_idx, row, error_info = future.result()
                        rows_by_idx[row_idx] = row
                        if error_info:
                            error_infos[row_idx] = error_info

                file_results = [rows_by_idx[row_idx] for row_idx in range(len(examples))]

                if not printed_first_raw:
                    first_raw = next(
                        (
                            row.get("raw_generation")
                            for row in file_results
                            if row.get("raw_generation")
                        ),
                        None,
                    )
                    if first_raw:
                        print("first_inference_raw:")
                        print(first_raw)
                        print()
                        printed_first_raw = True

                for row_idx in sorted(error_infos):
                    print_oneshot_inference_error(error_infos[row_idx])

                split_results[test_key].extend(file_results)

            df_file = pd.DataFrame(file_results)
            print_eval(df_file, title=f"{test_key}/{file_path.name}", test_type=model_name)
            all_results.extend(file_results)

    result_df = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(result_df, title=f"combined={combined_title}", test_type=model_name)
    print_turn_macro_summary(result_df, title=f"combined={combined_title}", metric="all")

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
