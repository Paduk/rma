import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from oneshot_qwen_prompt import (
    build_api_str_from_candidates,
    build_user_content,
    render_chat_template,
    resolve_prompt_tokenizer_name,
)


DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_HOST = "http://localhost:11436"
DEFAULT_MODEL_NAME = "phi4-oneshot-rewrite-first-block:latest"
DEFAULT_PROMPT_TOKENIZER_NAME = None
PRIMARY_QUERY_FIELD = "rewritten_query"
TAG_SEQUENCE = (PRIMARY_QUERY_FIELD, "plan", "arguments")
TAGGED_ONESHOT_SYSTEM_PROMPT = (
    "Given a conversation history, a user query, and a list of available tools, "
    "first rewrite the query by resolving only ambiguous pronouns or omitted "
    "references using the conversation history. If there are no ambiguous "
    "pronouns or omitted references, rewritten_query may be identical to the "
    "user query. Then, based on the rewritten_query, select the most appropriate "
    "tool and generate its arguments. Return the answer using exactly these "
    "three sections in this order:\n"
    "<rewritten_query>\n"
    "...rewritten query...\n"
    "</rewritten_query>\n"
    "<plan>\n"
    "...tool name or None...\n"
    "</plan>\n"
    "<arguments>\n"
    "...compact JSON object...\n"
    "</arguments>\n"
    "Always include all three sections. The content inside <arguments> must be "
    "a valid compact JSON object. If no tool matches the request, set the plan "
    "to None and the arguments object to {}."
)
"""
python3 /home/hj153lee/new-rma/exp_ollama_oneshot_inference.py \
--model_name phi4-oneshot-rewrite-first-block:latest \
--test_key base,complex \
--o /home/hj153lee/new-rma/datasets/result/260406-ablation/phi4-rewrite-first-block.tsv \
--host http://localhost:11435
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate experimental one-shot RMA models via Ollama using the "
            "rewrite-first block output format."
        )
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
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
        help="HF tokenizer used to render the chat template. If omitted, infer it from --model_name.",
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


def build_tagged_system_message(api_str: str) -> str:
    return f"{TAGGED_ONESHOT_SYSTEM_PROMPT}\n<|tool|>{api_str}<|/tool|>"


def build_oneshot_messages(conversation_history: str, query: str, candidates, apis):
    api_str = build_api_str_from_candidates(candidates, apis)
    return [
        {"role": "system", "content": build_tagged_system_message(api_str)},
        {"role": "user", "content": build_user_content(conversation_history, query)},
    ]


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


def generate_text(prompt, model, host, temperature=0.0, num_predict=512):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
            "stream": False,
        },
        timeout=300,
    )
    if response.status_code != 200:
        raise RuntimeError(f"API request failed: {response.text}")
    return response.json()["response"]


def extract_tagged_section(text: str, tag: str, following_tags: tuple[str, ...]) -> str | None:
    start_match = re.search(rf"<{tag}>", text, re.IGNORECASE)
    if not start_match:
        return None

    start = start_match.end()
    end_match = re.search(rf"</{tag}>", text[start:], re.IGNORECASE | re.DOTALL)
    if end_match:
        end = start + end_match.start()
        return text[start:end].strip()

    next_positions = []
    for next_tag in following_tags:
        next_match = re.search(rf"<{next_tag}>", text[start:], re.IGNORECASE)
        if next_match:
            next_positions.append(start + next_match.start())
    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    if depth > 0 and text[start:].strip().startswith("{"):
        return text[start:] + ("}" * depth)
    return None


def parse_arguments_block(arguments_text: str) -> dict:
    candidate = arguments_text.strip()
    if not candidate:
        raise ValueError("arguments block is empty.")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(candidate)
        if not candidate:
            raise ValueError("No JSON object found in arguments block.")
        parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("arguments block did not parse to a dict.")
    return parsed


def extract_rewritten_query_section(raw_text: str) -> str:
    section = extract_tagged_section(raw_text, PRIMARY_QUERY_FIELD, TAG_SEQUENCE[1:])
    if section is not None:
        return section
    raise ValueError(f"Missing <{PRIMARY_QUERY_FIELD}> section.")


def parse_oneshot_response(raw_text: str):
    rewritten_query = extract_rewritten_query_section(raw_text)
    sections = {PRIMARY_QUERY_FIELD: rewritten_query}
    for idx, tag in enumerate(TAG_SEQUENCE[1:], start=1):
        section = extract_tagged_section(raw_text, tag, TAG_SEQUENCE[idx + 1 :])
        if section is None:
            raise ValueError(f"Missing <{tag}> section.")
        sections[tag] = section

    return {
        PRIMARY_QUERY_FIELD: sections[PRIMARY_QUERY_FIELD],
        "plan": sections["plan"],
        "arguments": parse_arguments_block(sections["arguments"]),
    }


def main():
    args = parse_args()
    apis = read_simple_apis(Path(args.tools_path))
    prompt_tokenizer_name = resolve_prompt_tokenizer_name(
        model_name=args.model_name,
        prompt_tokenizer_name=args.prompt_tokenizer_name,
    )
    prompt_tokenizer = load_prompt_tokenizer(prompt_tokenizer_name)
    data_files = build_data_files()
    test_keys = parse_test_keys(args.test_key, data_files)
    print(f"model_name: {args.model_name}")
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

                    raw = generate_text(
                        prompt=prompt,
                        model=args.model_name,
                        host=args.host,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
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
                    "conversation_history": example.get("conversation_history"),
                    "query": example.get("query"),
                    "rewritten_query": example.get("rewritten_query", example.get("rewrited_query")),
                    "candidates": example.get("candidates"),
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
            print_eval(df_file, title=f"{test_key}/{file_path.name}", test_type=args.model_name)
            all_results.extend(file_results)

    result_df = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=args.model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(result_df, title=f"combined={combined_title}", test_type=args.model_name)
    print_turn_macro_summary(result_df, title=f"combined={combined_title}", metric="all")

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
