import argparse
import ast
import json
import os
import re
from functools import partial
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from train.gemma_prompts import (
    ZERO_HISTORY_INFERENCE_GEMMA,
    ZERO_REWRITE_INFERENCE_GEMMA,
)
from train.gemma4_legacy_prompting import build_legacy_gemma4_prompt
from train.planner_json_utils import normalize_answer_to_dict, serialize_answer_to_json
from utils.generation_backends import (
    add_text_generation_backend_args,
    build_text_generation_backend_from_args,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "http://localhost:11436"
DEFAULT_API_PATH = PROJECT_ROOT / "apis" / "simple_api.json"

DEFAULT_DATA_FILES = {
    "swap": [
        "datasets/tc/it2_nonNR_tc_swapped_backup.tsv",
        "datasets/tc/it5_nonNR_tc_swapped_backup.tsv",
    ],
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
    "manual_rewrited": [
        "datasets/tc/manual/llama_rewrited/turn2.tsv",
        "datasets/tc/manual/llama_rewrited/turn3.tsv",
        "datasets/tc/manual/llama_rewrited/turn4.tsv",
        "datasets/tc/manual/llama_rewrited/turn5.tsv",
    ],
}

DEFAULT_TEST_TYPE_CONFIG = {
    "base-gemma4": {
        "model_name": "gemma-4-e2b-it-history-lm_only_lora:latest",
        "prompt_template": None,
        "prompt_mode": "base",
        "prompt_style": "legacy_chat",
        "response_format_json": False,
    },
    "rewrite-gemma4": {
        "model_name": "gemma-4-e2b-it-rewrite-lm_only_lora:latest",
        "prompt_template": None,
        "prompt_mode": "rewrite",
        "prompt_style": "legacy_chat",
        "response_format_json": False,
    },
    "base-gemma4-base": {
        "model_name": "gemma4:latest",
        "prompt_template": ZERO_HISTORY_INFERENCE_GEMMA,
        "prompt_mode": "base",
        "prompt_style": "zero_shot_template",
        "response_format_json": True,
    },
    "rewrite-gemma4-base": {
        "model_name": "gemma4:latest",
        "prompt_template": ZERO_REWRITE_INFERENCE_GEMMA,
        "prompt_mode": "rewrite",
        "prompt_style": "zero_shot_template",
        "response_format_json": True,
    },
    "new-base-gemma4": {
        "model_name": "gemma-4-e2b-it-history-lm_only_lora:latest",
        "prompt_template": None,
        "prompt_mode": "base",
        "prompt_style": "legacy_chat",
        "response_format_json": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 multi-file inference and evaluation.")
    parser.add_argument("--t", required=True, help="Inference preset key.")
    parser.add_argument("--o", required=True, help="Output TSV path.")
    parser.add_argument("--test_key", required=True, help="Comma-separated test split keys.")
    add_text_generation_backend_args(parser, default_host=DEFAULT_HOST)
    parser.add_argument(
        "--api",
        default=str(DEFAULT_API_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--prompt_tokenizer_name",
        default="google/gemma-4-E2B-it",
        help="HF tokenizer used to reproduce the Gemma 4 training chat template.",
    )
    parser.add_argument(
        "--debug_print_prompt",
        action="store_true",
        help="Print the rendered prompt for debugging.",
    )
    parser.add_argument(
        "--debug_print_raw",
        action="store_true",
        help="Print raw model outputs before parsing.",
    )
    parser.add_argument(
        "--debug_limit",
        type=int,
        default=5,
        help="Maximum number of prompts/raw outputs to print in debug mode.",
    )
    parser.add_argument(
        "--debug_first_file_only",
        action="store_true",
        help="Process only the first file of the first selected split, then stop.",
    )
    parser.add_argument(
        "--debug_case_tsv",
        default=None,
        help="Optional TSV path for per-case debug rows including prompt/raw/gt/result.",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["relaxed", "strict_json"],
        default="relaxed",
        help="Use relaxed post-processing or evaluate only raw valid JSON responses.",
    )
    return parser.parse_args()


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


def quote_bare_keys(dict_like_text: str) -> str:
    return re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', dict_like_text)


def parse_loose_dict(dict_like_text: str):
    candidates = [
        dict_like_text.strip(),
        quote_bare_keys(dict_like_text.strip()),
    ]
    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def normalize_plan_name(plan: str, candidate_plans):
    if not plan:
        return plan
    if not candidate_plans:
        return plan
    if plan in candidate_plans:
        return plan

    plan_lower = plan.lower()
    candidate_map = {candidate.lower(): candidate for candidate in candidate_plans}
    if plan_lower in candidate_map:
        return candidate_map[plan_lower]

    stripped = re.sub(r"^action_", "", plan_lower)
    for candidate in candidate_plans:
        candidate_lower = candidate.lower()
        if candidate_lower == stripped:
            return candidate
        if re.sub(r"^action_", "", candidate_lower) == stripped:
            return candidate
    return plan


def coerce_attachment_value(param_name: str, value):
    if param_name in {"attachments"} and value is not None and not isinstance(value, list):
        return [value]
    return value


def parse_param_list_style(raw_list, params):
    assigned = {}
    used_indices = set()

    i = 0
    while i < len(raw_list) - 1:
        item = raw_list[i]
        next_item = raw_list[i + 1]
        if (
            isinstance(item, str)
            and item in params
            and not (isinstance(next_item, str) and next_item in params)
        ):
            assigned[item] = next_item
            used_indices.add(i)
            used_indices.add(i + 1)
            i += 2
            continue
        i += 1

    leftovers = [
        item
        for idx, item in enumerate(raw_list)
        if idx not in used_indices and not (isinstance(item, str) and item in params)
    ]
    for param in params:
        if param not in assigned and leftovers:
            assigned[param] = leftovers.pop(0)

    for param in list(assigned):
        assigned[param] = coerce_attachment_value(param, assigned[param])

    return assigned if assigned else None


def parse_yaml_like_body(body: str):
    result = {}
    current_list_key = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                continue
            item = stripped[2:].strip()
            if ": " in item:
                _, value = item.split(": ", 1)
                item = value
            try:
                item = ast.literal_eval(item)
            except Exception:
                item = item.strip("'\"")
            result.setdefault(current_list_key, []).append(item)
            continue

        current_list_key = None
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            result[key] = []
            current_list_key = key
            continue
        try:
            parsed_value = ast.literal_eval(value)
        except Exception:
            parsed_value = value.strip("'\"")
        result[key] = parsed_value

    return result if result else None


def parse_yaml_like_plan(text: str, candidate_plans):
    stripped = (text or "").strip()
    plan_match = re.match(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\n(?P<body>(?:.+\n?)*)$",
        stripped,
        re.DOTALL,
    )
    if not plan_match:
        return None
    plan = normalize_plan_name(plan_match.group(1), candidate_plans)
    body = parse_yaml_like_body(plan_match.group("body"))
    if isinstance(body, dict):
        return {"plan": plan, "arguments": body}
    return None


def parse_tool_name_params_dict(parsed, candidate_plans):
    if not isinstance(parsed, dict):
        return None
    if "tool_name" not in parsed or "params" not in parsed:
        return None
    params = parsed.get("params")
    if not isinstance(params, dict):
        return None
    plan = normalize_plan_name(parsed.get("tool_name"), candidate_plans)
    return {"plan": plan, "arguments": params}


def parse_python_call(text: str):
    stripped = text.strip()
    if stripped.startswith("print(") and stripped.endswith(")"):
        stripped = stripped[6:-1].strip()

    try:
        expr = ast.parse(stripped, mode="eval").body
    except Exception:
        return None

    if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
        return None
    if expr.args:
        return None

    arguments = {}
    for kw in expr.keywords:
        if kw.arg is None:
            return None
        try:
            arguments[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            return None
    return {"plan": expr.func.id, "arguments": arguments}


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


def parse_arguments_block(arguments_text: str):
    candidate = arguments_text.strip()
    if not candidate:
        return None
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = "{}"

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(candidate)
        if not candidate:
            return None
        try:
            parsed = json.loads(candidate)
        except Exception:
            return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_tagged_plan_arguments(text: str, candidate_plans):
    plan = extract_tagged_section(text, "plan", ("arguments",))
    arguments_text = extract_tagged_section(text, "arguments", ())
    if plan is None or arguments_text is None:
        return None

    arguments = parse_arguments_block(arguments_text)
    if not isinstance(arguments, dict):
        return None

    return {
        "plan": normalize_plan_name(plan, candidate_plans),
        "arguments": arguments,
    }


def parse_strict_json_result(raw: str, candidate_plans=None):
    stripped = (raw or "").strip()
    if not stripped:
        return None

    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        return None
    if "plan" not in parsed or "arguments" not in parsed:
        return None
    if not isinstance(parsed["arguments"], dict):
        return None
    return parsed


def parse_generation_result(raw: str, candidate_plans=None, apis=None):
    stripped = (raw or "").strip()
    if not stripped:
        return None

    parsed = parse_tagged_plan_arguments(stripped, candidate_plans)
    if isinstance(parsed, dict):
        return parsed

    try:
        parsed = ast.literal_eval(stripped)
        if isinstance(parsed, dict):
            converted = parse_tool_name_params_dict(parsed, candidate_plans)
            if converted is not None:
                return converted
            if "plan" in parsed and "arguments" in parsed:
                parsed["plan"] = normalize_plan_name(parsed["plan"], candidate_plans)
                return parsed
    except Exception:
        pass

    parsed = extract_json_from_markdown(stripped)
    if isinstance(parsed, dict):
        converted = parse_tool_name_params_dict(parsed, candidate_plans)
        if converted is not None:
            return converted
        if "plan" in parsed and "arguments" in parsed:
            parsed["plan"] = normalize_plan_name(parsed["plan"], candidate_plans)
            return parsed

    parsed = parse_python_call(stripped)
    if isinstance(parsed, dict):
        parsed["plan"] = normalize_plan_name(parsed["plan"], candidate_plans)
        return parsed

    plan_match = re.match(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:?\s*(\{.*\})\s*$",
        stripped,
        re.DOTALL,
    )
    if plan_match:
        arguments = parse_loose_dict(plan_match.group(2))
        if isinstance(arguments, dict):
            return {
                "plan": normalize_plan_name(plan_match.group(1), candidate_plans),
                "arguments": arguments,
            }

    list_match = re.match(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\[.*\])\s*$",
        stripped,
        re.DOTALL,
    )
    if list_match:
        try:
            raw_list = ast.literal_eval(list_match.group(2))
        except Exception:
            raw_list = None
        if isinstance(raw_list, list):
            plan = normalize_plan_name(list_match.group(1), candidate_plans)
            params = apis.get(plan, []) if apis else []
            arguments = parse_param_list_style(raw_list, params)
            if isinstance(arguments, dict):
                return {"plan": plan, "arguments": arguments}

    parsed = parse_yaml_like_plan(stripped, candidate_plans)
    if isinstance(parsed, dict):
        return parsed

    return None


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
        detail_df["macro_by_plan"] = detail_df[list(metrics)].mean(axis=1).round(2)
        print(detail_df.to_string(index=False))

def extract_turn_from_filename(file_name):
    match = re.search(r"it(\d+)", file_name)
    if not match:
        return None
    return int(match.group(1))


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


def parse_test_keys(test_key_arg, data_files):
    if not test_key_arg:
        raise ValueError("`--test_key` is required. Example: --test_key base or --test_key base,complex")

    test_keys = [key.strip() for key in test_key_arg.split(",") if key.strip()]
    if not test_keys:
        raise ValueError("No valid test_key values were provided.")

    deduped_test_keys = []
    for key in test_keys:
        if key not in data_files:
            raise ValueError(f"Invalid test_key: {key}. Available keys: {', '.join(sorted(data_files.keys()))}")
        if key not in deduped_test_keys:
            deduped_test_keys.append(key)

    return deduped_test_keys


def preprocess_example_it(example, apis, prompt_template, prompt_mode, prompt_style, tokenizer):
    api_str = ""
    for plan in ast.literal_eval(example["candidates"]):
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"

    if prompt_style == "legacy_chat":
        if tokenizer is None:
            raise ValueError("Gemma 4 legacy chat prompting requires a tokenizer.")
        if prompt_mode == "base":
            prompt = build_legacy_gemma4_prompt(
                tokenizer,
                tools=api_str,
                prompt_mode="base",
                query=example["query"],
                conversation_history=example["conversation_history"],
            )
        elif prompt_mode == "rewrite":
            prompt = build_legacy_gemma4_prompt(
                tokenizer,
                tools=api_str,
                prompt_mode="rewrite",
                query=example["rewrited_query"],
            )
        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    elif prompt_style == "zero_shot_template":
        if prompt_mode == "base":
            prompt = prompt_template.format(
                tools=api_str,
                conversation_history=example["conversation_history"],
                data=example["query"],
            )
        elif prompt_mode == "rewrite":
            prompt = prompt_template.format(
                tools=api_str,
                data=example["rewrited_query"],
            )
        else:
            raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    return {
        "strprompt": prompt,
        "stranswer": serialize_answer_to_json(example["answer"], indent=2),
        "candidates": example["candidates"],
        "rewrited_query": example["rewrited_query"],
        "query": example["query"],
        "conversation_history": example["conversation_history"],
    }


def resolve_test_type_config(test_type: str):
    config = DEFAULT_TEST_TYPE_CONFIG.get(test_type)
    if config is None:
        valid_test_types = ", ".join(sorted(DEFAULT_TEST_TYPE_CONFIG.keys()))
        raise ValueError(f"Invalid test type: {test_type}. Available test types: {valid_test_types}")
    return config


def serialize_for_tsv(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def main():
    args = parse_args()
    sft_apis = read_apis(args.api, simple=True)
    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    debug_case_tsv_path = (
        Path(args.debug_case_tsv)
        if args.debug_case_tsv
        else out_path.with_name(f"{out_path.stem}.first_file_debug.tsv")
        if args.debug_first_file_only
        else None
    )

    config = resolve_test_type_config(args.t)
    prompt_template = config["prompt_template"]
    prompt_mode = config["prompt_mode"]
    prompt_style = config["prompt_style"]
    response_format_json = config["response_format_json"] or args.eval_mode == "strict_json"
    test_keys = parse_test_keys(args.test_key, DEFAULT_DATA_FILES)
    generation_backend = build_text_generation_backend_from_args(
        args,
        default_model_name=config["model_name"],
    )
    active_model_name = generation_backend.model_label
    prompt_tokenizer = None
    if prompt_style == "legacy_chat":
        prompt_tokenizer = AutoTokenizer.from_pretrained(args.prompt_tokenizer_name)

    print(f"backend: {generation_backend.backend_name}")
    print(f"generation_model: {active_model_name}")
    if args.backend == "ollama":
        print(args.host)

    all_results = []
    split_results = {key: [] for key in test_keys}
    debug_rows = []
    executed_test_keys = []
    debug_counter = 0
    print(f"Selected test_keys: {test_keys}")
    for test_key in test_keys:
        if args.debug_first_file_only and executed_test_keys:
            break
        executed_test_keys.append(test_key)
        print(f"\n# Running split: {test_key}")
        print(DEFAULT_DATA_FILES[test_key])
        file_paths = DEFAULT_DATA_FILES[test_key]
        if args.debug_first_file_only:
            file_paths = file_paths[:1]

        for file_path in file_paths:
            ds = load_dataset("csv", data_files={"tc": [file_path]}, delimiter="\t")["tc"]
            proc = ds.map(
                partial(
                    preprocess_example_it,
                    apis=sft_apis,
                    prompt_template=prompt_template,
                    prompt_mode=prompt_mode,
                    prompt_style=prompt_style,
                    tokenizer=prompt_tokenizer,
                )
            )
            if args.debug_print_prompt and debug_counter < args.debug_limit:
                print(proc[0]["strprompt"])
                debug_counter += 1

            file_results = []
            for row_idx, ex in enumerate(
                tqdm(proc, desc=f"Processing {test_key}/{os.path.basename(file_path)}")
            ):
                prompt = ex["strprompt"]
                raw = ""
                gt = {}
                error_message = ""
                parse_ok = False

                try:
                    raw = generation_backend.generate_text(
                        prompt,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
                        response_format_json=response_format_json,
                    )
                    if args.debug_print_raw and debug_counter < args.debug_limit:
                        print(f"[RAW][{test_key}/{os.path.basename(file_path)}#{row_idx}] {raw}")
                        debug_counter += 1

                    candidate_plans = ast.literal_eval(ex["candidates"])
                    if args.eval_mode == "strict_json":
                        result = parse_strict_json_result(
                            raw,
                            candidate_plans=candidate_plans,
                        )
                    else:
                        result = parse_generation_result(
                            raw,
                            candidate_plans=candidate_plans,
                            apis=sft_apis,
                        )
                    if not isinstance(result, dict):
                        raise ValueError("Failed to parse model output into {plan, arguments} dict")
                    parse_ok = True

                    gt = normalize_answer_to_dict(ex["stranswer"])

                    plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                    arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                    all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
                except Exception as e:
                    result = {"error": str(e)}
                    print(f"Error: {e}, {raw}")
                    error_message = str(e)
                    plan_res = "fail"
                    arg_res = "fail"
                    all_res = "fail"

                row = {
                    "test_key": test_key,
                    "backend_name": generation_backend.backend_name,
                    "model_name": active_model_name,
                    "conversation_history": ex.get("conversation_history"),
                    "query": ex.get("query"),
                    "rewrited_query": ex.get("rewrited_query"),
                    "candidates": ex.get("candidates"),
                    "raw": raw,
                    "raw_generated": raw,
                    "generation": result,
                    "gt": gt,
                    "plan": plan_res,
                    "arguments": arg_res,
                    "all": all_res,
                    "file": os.path.basename(file_path),
                    "turn": extract_turn_from_filename(os.path.basename(file_path)),
                }
                file_results.append(row)
                split_results[test_key].append(row)
                if debug_case_tsv_path is not None:
                    debug_rows.append(
                        {
                            "test_key": test_key,
                            "backend_name": generation_backend.backend_name,
                            "model_name": active_model_name,
                            "file": os.path.basename(file_path),
                            "row_idx": row_idx,
                            "turn": extract_turn_from_filename(os.path.basename(file_path)),
                            "query": ex.get("query"),
                            "conversation_history": serialize_for_tsv(ex.get("conversation_history")),
                            "candidates": serialize_for_tsv(ast.literal_eval(ex["candidates"])),
                            "prompt": prompt,
                            "raw": raw,
                            "raw_generated": raw,
                            "parsed_generation": serialize_for_tsv(result),
                            "gt": serialize_for_tsv(gt),
                            "parse_ok": parse_ok,
                            "error_message": error_message,
                            "plan": plan_res,
                            "arguments": arg_res,
                            "all": all_res,
                        }
                    )

            df_file = pd.DataFrame(file_results)
            print_eval(
                df_file,
                title=f"{test_key}/{os.path.basename(file_path)}",
                test_type=active_model_name,
            )
            all_results.extend(file_results)

        if args.debug_first_file_only:
            print(f"[debug] stopped after first file: {os.path.basename(file_paths[0])}")
            break

    result = pd.DataFrame(all_results)
    for test_key in executed_test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=active_model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(executed_test_keys)
    print_eval(result, title=f"combined={combined_title}", test_type=active_model_name)
    print_turn_macro_summary(result, title=f"combined={combined_title}", metric="all")
    result.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")
    if debug_case_tsv_path is not None:
        debug_case_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(debug_rows).to_csv(
            debug_case_tsv_path,
            sep="\t",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"[debug] wrote per-case TSV: {debug_case_tsv_path}")


if __name__ == "__main__":
    main()
