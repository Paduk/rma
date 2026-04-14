import argparse
import ast
import json
import math
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from train.gemma_prompts import SFT_REWRITE_INFERENCE_GEMMA
from train.llama_prompts import (
    SFT_REWRITE_INFERENCE_LLAMA,
    SFT_REWRITE_INFERENCE_PHI4,
    SFT_REWRITE_INFERENCE_QWEN25,
    SFT_REWRITE_INFERENCE_QWEN3,
)
from train.rma_model_profiles import RMA_MODEL_PROFILES


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "result" / "oneshot_test"
DEFAULT_API_FILE = BASE_DIR / "apis" / "simple_api.json"
DEFAULT_LOG_FILE = BASE_DIR / "logs" / "ollama_inference_log.txt"
DEFAULT_REWRITE_TYPE = "rewrite-phi4"
GLM_STOP_SEQUENCES = [
    "<|observation|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]
GENERIC_SYSTEM_REWRITE_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query.\n "
    "<|tool|>{tools}<|/tool|>"
)
GENERIC_PROFILE_NAMES = {
    "glm-edge-1.5b",
    "glm-edge-4b",
    "smollm2-1.7b",
    "smollm2-1.7b-instruct",
    "smollm3-3b",
    "falcon3-1b",
    "falcon3-1b-base",
    "falcon3-3b",
    "falcon3-3b-base",
    "exaone4-1.2b",
    "olmo2-1b",
    "olmo2-1b-instruct",
    "granite3.3-2b",
    "lfm2.5-1.2b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate oneshot rewrite datasets with rewrite-mode SFT models.")
    parser.add_argument("--input_file", type=str, required=True, help="Input TSV file path")
    parser.add_argument("--o", type=str, default="", help="Output TSV path")
    parser.add_argument("--host", type=str, default="http://localhost:11436", help="Ollama host URL")
    parser.add_argument(
        "--rewrite_type",
        "--t",
        dest="rewrite_type",
        type=str,
        default=DEFAULT_REWRITE_TYPE,
        help="Rewrite prompt/model config key, e.g. rewrite-phi4 or rewrite-granite3.3-2b.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RMA_MODEL_PROFILES.keys()),
        default=None,
        help="Shortcut for --rewrite_type rewrite-<profile> when the profile is supported.",
    )
    parser.add_argument(
        "--model",
        "--model_name",
        dest="model",
        type=str,
        default=None,
        help="Ollama model name override.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama sampling temperature")
    parser.add_argument("--num_predict", type=int, default=512, help="Ollama max generation tokens")
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for profile tokenizers without chat_template.",
    )
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


def sanitize_model_slug(model_name: str) -> str:
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


def infer_generic_ollama_model_name(profile_name: str, prompt_model_name: str, train_type: str, prefix: str) -> str:
    model_slug = sanitize_model_slug(prompt_model_name)
    return f"{model_slug}-{profile_name}-{train_type}-{prefix}:latest"


def build_rewrite_type_configs():
    configs = {
        "rewrite-phi4": {
            "model_name": "phi4-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_PHI4,
        },
        "rewrite-phi": {
            "model_name": "phi4-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_PHI4,
        },
        "rewrite-llama3": {
            "model_name": "llama3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
        },
        "rewrite-llama": {
            "model_name": "llama3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
        },
        "rewrite-gemma": {
            "model_name": "gemma-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_GEMMA,
        },
        "rewrite-qwen3": {
            "model_name": "qwen3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
        },
        "rewrite-qwen": {
            "model_name": "qwen3-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
        },
        "rewrite-qwen3-1.7b": {
            "model_name": "qwen3-rewrite-1.7b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
        },
        "rewrite-qwen3-0.6b": {
            "model_name": "qwen3-rewrite-0.6b:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
        },
        "rewrite-qwen2.5": {
            "model_name": "qwen2.5-rewrite:latest",
            "prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
        },
    }

    for profile_name in GENERIC_PROFILE_NAMES:
        profile = RMA_MODEL_PROFILES[profile_name]
        config = {
            "model_name": infer_generic_ollama_model_name(
                profile_name=profile_name,
                prompt_model_name=profile.model_name,
                train_type="rewrite",
                prefix=profile.prefix,
            ),
            "prompt_renderer": "chat_template",
            "prompt_model_name": profile.model_name,
            "chat_template_fallback": "simple",
        }
        if profile_name == "glm-edge-4b":
            config["stop"] = GLM_STOP_SEQUENCES
        configs[f"rewrite-{profile_name}"] = config

    return configs


def profile_to_rewrite_type(profile: str) -> str:
    profile_aliases = {
        "phi": "rewrite-phi4",
        "llama": "rewrite-llama3",
        "qwen": "rewrite-qwen3",
        "gemma": "rewrite-gemma",
    }
    return profile_aliases.get(profile, f"rewrite-{profile}")


def infer_rewrite_type_from_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    lower_name = model_name.lower()
    legacy_patterns = [
        ("rewrite-qwen3-1.7b", ("qwen3-1.7b",)),
        ("rewrite-qwen3-0.6b", ("qwen3-0.6b",)),
        ("rewrite-qwen2.5", ("qwen2.5", "qwen25")),
        ("rewrite-phi4", ("phi4", "phi-4")),
        ("rewrite-llama3", ("llama3", "llama-3")),
        ("rewrite-gemma", ("gemma",)),
        ("rewrite-qwen3", ("qwen3",)),
    ]
    for rewrite_type, patterns in legacy_patterns:
        if any(pattern in lower_name for pattern in patterns):
            return rewrite_type

    for profile_name in sorted(GENERIC_PROFILE_NAMES, key=len, reverse=True):
        profile = RMA_MODEL_PROFILES[profile_name]
        model_tail = re.sub(
            r"[^a-z0-9]+",
            "-",
            profile.model_name.split("/")[-1].lower(),
        ).strip("-")
        match_keys = [profile_name.lower(), model_tail, sanitize_model_slug(profile.model_name)]
        if any(match_key and match_key in lower_name for match_key in match_keys):
            return f"rewrite-{profile_name}"
    return None


def resolve_rewrite_config(args):
    configs = build_rewrite_type_configs()
    rewrite_type = args.rewrite_type
    if args.profile:
        profile_rewrite_type = profile_to_rewrite_type(args.profile)
        if rewrite_type != DEFAULT_REWRITE_TYPE and rewrite_type != profile_rewrite_type:
            raise ValueError(
                f"--profile {args.profile} conflicts with --rewrite_type {rewrite_type}. "
                f"Use --rewrite_type {profile_rewrite_type} or omit --rewrite_type."
            )
        rewrite_type = profile_rewrite_type
    elif args.model and rewrite_type == DEFAULT_REWRITE_TYPE:
        inferred_rewrite_type = infer_rewrite_type_from_model_name(args.model)
        if inferred_rewrite_type:
            rewrite_type = inferred_rewrite_type

    if rewrite_type not in configs:
        valid = ", ".join(sorted(configs.keys()))
        raise ValueError(f"Invalid rewrite_type: {rewrite_type}. Available rewrite types: {valid}")

    config = configs[rewrite_type].copy()
    config["rewrite_type"] = rewrite_type
    config["model_name"] = args.model or config["model_name"]
    if config.get("prompt_renderer") == "chat_template":
        config["chat_template_fallback"] = args.chat_template_fallback
    return config


def render_chat_template(tokenizer, messages, add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_simple_inference_prompt(system_msg: str, user_content: str) -> str:
    return (
        f"System:\n{system_msg}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:\n"
    )


def render_generic_inference_prompt(
    rewrited_query: str,
    api_str: str,
    tokenizer,
    chat_template_fallback: str,
) -> str:
    system_msg = GENERIC_SYSTEM_REWRITE_PROMPT.format(tools=api_str)
    user_content = f"User Query: {rewrited_query}"
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "only for models that were trained with simple fallback."
        )
    return render_simple_inference_prompt(system_msg, user_content)


def load_prompt_tokenizer(config):
    if config.get("prompt_renderer") != "chat_template":
        return None

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config["prompt_model_name"],
        trust_remote_code=True,
    )


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


def resolve_rewrited_query(row):
    generation = parse_literal(row.get("generation"))
    if isinstance(generation, dict):
        generated_rewrite = pick_value(generation.get("rewrited_query"))
        if generated_rewrite is not None:
            return generated_rewrite, "generation"

    fallback_rewrite = pick_value(row.get("rewrited_query"))
    if fallback_rewrite is not None:
        return fallback_rewrite, "rewrited_query"

    raise ValueError("`generation.rewrited_query` and `rewrited_query` are both missing")


def extract_json_from_markdown(text):
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if code_block_match:
        return json.loads(code_block_match.group(1))

    json_match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not json_match:
        raise ValueError("JSON block not found")
    return json.loads(json_match.group())


def truncate_at_stop_markers(text: str, stop_markers=None) -> str:
    if not stop_markers or not isinstance(text, str):
        return text
    cut_idx = len(text)
    for marker in stop_markers:
        if not marker:
            continue
        marker_idx = text.find(marker)
        if marker_idx != -1:
            cut_idx = min(cut_idx, marker_idx)
    return text[:cut_idx].strip()


def parse_model_output(raw, stop_markers=None):
    raw = truncate_at_stop_markers(raw, stop_markers)
    for parser in (ast.literal_eval, json.loads, extract_json_from_markdown):
        try:
            parsed = parser(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError("Failed to parse model output as dict")


def generate_text(prompt, *, model, host, temperature, num_predict, stop=None):
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

    file_names = sorted(df["file"].dropna().unique().tolist(), key=get_turn_file_sort_key)
    rows = []
    for turn, sub in sorted(df.dropna(subset=["turn"]).groupby("turn"), key=lambda x: x[0]):
        turn_macro = compute_plan_macro(sub, metric="all")
        row = {
            "turn": int(turn),
            "samples": len(sub),
            "plan_macro_all": round(turn_macro, 4),
        }
        source_files = set(sub["file"].dropna().tolist())
        for file_name in file_names:
            if file_name in source_files:
                row[file_name] = round(compute_file_accuracy(sub[sub["file"] == file_name], metric="all"), 4)
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


def build_prompt(row, apis, config, prompt_tokenizer):
    api_str = build_api_str(row, apis)
    rewrited_query, rewrited_query_source = resolve_rewrited_query(row)

    if config.get("prompt_renderer") == "chat_template":
        prompt = render_generic_inference_prompt(
            rewrited_query,
            api_str=api_str,
            tokenizer=prompt_tokenizer,
            chat_template_fallback=config.get("chat_template_fallback", "simple"),
        )
        return prompt, rewrited_query, rewrited_query_source

    prompt = config["prompt_template"].format(
        tools=api_str,
        data=rewrited_query,
    )
    return prompt, rewrited_query, rewrited_query_source


def resolve_output_path(input_file: Path, output_arg: str) -> Path:
    if output_arg:
        return Path(output_arg)
    return DEFAULT_OUTPUT_DIR / f"{input_file.stem}.tsv"


def evaluate_file(df, *, dataset_name, apis, config, prompt_tokenizer, host, temperature, num_predict):
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
        prompt_rewrited_query = None
        rewrited_query_source = None

        try:
            prompt, prompt_rewrited_query, rewrited_query_source = build_prompt(
                row,
                apis,
                config,
                prompt_tokenizer,
            )
            if not prompt_preview_printed:
                print(prompt)
                prompt_preview_printed = True

            raw = generate_text(
                prompt,
                model=config["model_name"],
                host=host,
                temperature=temperature,
                num_predict=num_predict,
                stop=config.get("stop"),
            )
            result = parse_model_output(raw, stop_markers=config.get("stop"))

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
                "rewrited_query": prompt_rewrited_query,
                "rewrited_query_source": rewrited_query_source,
                "gold_rewrited_query": row.get("rewrited_query"),
                "oneshot_generation": row.get("generation"),
                "candidates": row.get("candidates"),
                "raw_generation": raw,
                "generation": result,
                "gt": gt,
                "plan": plan_res,
                "arguments": arg_res,
                "all": all_res,
                "file": source_file,
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
    config = resolve_rewrite_config(args)
    prompt_tokenizer = load_prompt_tokenizer(config)

    print(f"rewrite_type: {config['rewrite_type']}")
    print(f"model_name: {config['model_name']}")
    if config.get("prompt_model_name"):
        print(f"prompt_model_name: {config['prompt_model_name']}")
    if config.get("stop"):
        print(f"stop: {config['stop']}")
    print(args.host)
    print(f"input_file: {input_file}")
    print(f"output_file: {output_path}")

    print(f"\n# Running dataset: {input_file.name}")
    df = pd.read_csv(input_file, sep="\t", keep_default_na=False)
    dataset_results = evaluate_file(
        df,
        dataset_name=input_file.name,
        apis=apis,
        config=config,
        prompt_tokenizer=prompt_tokenizer,
        host=args.host,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    dataset_df = pd.DataFrame(dataset_results)
    print_eval(dataset_df, title=f"dataset={input_file.name}", log_file=DEFAULT_LOG_FILE, test_type=config["model_name"])
    print_turn_macro_summary(dataset_df, title=f"dataset={input_file.name}")

    for test_key, rows in sorted(dataset_df.groupby("test_key"), key=lambda x: x[0]):
        split_df = pd.DataFrame(rows)
        print_eval(split_df, title=f"split={test_key}", log_file=DEFAULT_LOG_FILE, test_type=config["model_name"])
        print_turn_macro_summary(split_df, title=f"split={test_key}")

    dataset_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8-sig")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
