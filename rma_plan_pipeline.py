import argparse
import ast
import json
import os
import re

import pandas as pd
import requests
from datasets import load_dataset
from tqdm import tqdm

from train.gemma_prompts import (
    SFT_REWRITE_INFERENCE_GEMMA,
    SFT_RMA_INFERENCE_GEMMA,
)
from train.llama_prompts import (
    SFT_HISTORY_INFERENCE_LLAMA,
    SFT_HISTORY_INFERENCE_PHI4,
    SFT_HISTORY_INFERENCE_QWEN25,
    SFT_HISTORY_INFERENCE_QWEN3,
    SFT_REWRITE_INFERENCE_LLAMA,
    SFT_REWRITE_INFERENCE_PHI4,
    SFT_REWRITE_INFERENCE_QWEN25,
    SFT_REWRITE_INFERENCE_QWEN3,
    SFT_RMA_INFERENCE_LLAMA,
    SFT_RMA_INFERENCE_PHI4,
    SFT_RMA_INFERENCE_QWEN3,
    ZERO_HISTORY_INFERENCE_LLAMA,
    ZERO_HISTORY_INFERENCE_PHI4,
    ZERO_HISTORY_INFERENCE_QWEN25,
    ZERO_HISTORY_INFERENCE_QWEN3,
    ZERO_REWRITE_INFERENCE_LLAMA,
    ZERO_REWRITE_INFERENCE_PHI4,
    ZERO_REWRITE_INFERENCE_QWEN25,
    ZERO_REWRITE_INFERENCE_QWEN3,
)
from train.rma_model_profiles import RMA_MODEL_PROFILES


GLM_STOP_SEQUENCES = [
    "<|observation|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]
RMA_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the query clearly by replacing ambiguous pronouns (like \"it\", "
    "\"that\") with explicit information from the conversation history. Keep "
    "exactly the same sentence structure. Do NOT generate or include any "
    "information, words, or values outside of the provided conversation_history "
    "and query."
)
PLANNING_SYSTEM_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values "
    "that are explicitly stated or can be reasonably inferred from the query.\n "
    "<|tool|>{tools}<|/tool|>"
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Rewrite first, then run planning evaluation")
    parser.add_argument(
        "--model_family",
        type=str,
        default=None,
        help="Legacy model family key, e.g. qwen3, qwen3-0.6b, qwen3-1.7b, llama3, phi4, qwen25.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RMA_MODEL_PROFILES.keys()),
        default=None,
        help="Named model profile for chat-template prompt rendering.",
    )
    parser.add_argument(
        "--rewrite_model",
        "--rewrite_model_name",
        dest="rewrite_model",
        type=str,
        default=None,
        help="Override the stage-1 RMA rewrite Ollama model name.",
    )
    parser.add_argument(
        "--plan_model",
        "--plan_model_name",
        dest="plan_model",
        type=str,
        default=None,
        help="Override the stage-2 planning Ollama model name.",
    )
    parser.add_argument(
        "--rewrite_prompt_tokenizer_name",
        default=None,
        help="HF tokenizer used for stage-1 chat-template prompt rendering.",
    )
    parser.add_argument(
        "--plan_prompt_tokenizer_name",
        default=None,
        help="HF tokenizer used for stage-2 chat-template prompt rendering.",
    )
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for tokenizers without chat_template.",
    )
    parser.add_argument("--test_key", type=str, required=True, help="dataset split key(s), comma separated")
    parser.add_argument("--o", type=str, required=True, help="output TSV path")
    parser.add_argument("--rewrite_host", type=str, default="http://localhost:11436", help="Ollama host for rewrite model")
    parser.add_argument("--plan_host", type=str, default="http://localhost:11435", help="Ollama host for planning model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama sampling temperature")
    parser.add_argument("--rewrite_num_predict", type=int, default=200, help="Max tokens for stage-1 rewrite")
    parser.add_argument("--plan_num_predict", type=int, default=512, help="Max tokens for stage-2 planning")
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


def generate_text(prompt, model, host, num_predict=512, stop=None, temperature=0.0):
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

    if response.status_code == 200:
        return response.json()["response"]
    raise Exception(f"API request failed: {response.text}")


def generate_text_rewrite_inference_style(prompt, model, host):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "format": "json",
                "num_predict": 200,
                "stop": ["}"],
            },
            "stream": False,
        },
    )

    if response.status_code == 200:
        return response.json()["response"]
    raise Exception(f"API request failed: {response.text}")


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


def truncate_at_stop_markers(text, stop_markers=None):
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
    if raw is None:
        raise ValueError("Empty response")

    raw = truncate_at_stop_markers(raw, stop_markers)
    try:
        return ast.literal_eval(raw)
    except Exception:
        result = extract_json_from_markdown(raw)
        if result is None:
            raise ValueError(f"Failed to parse model output: {raw}")
        return result


def parse_rewrite_output_inference_style(raw, stop_markers=None):
    try:
        return parse_model_output(raw, stop_markers=stop_markers)["rewrited_query"]
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

    with open("logs/ollama_inference_log.txt", "a", encoding="utf-8") as f:
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


def sanitize_model_slug(model_name):
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


def infer_profile_ollama_model_name(profile_name, prompt_model_name, train_type, prefix):
    model_slug = sanitize_model_slug(prompt_model_name)
    return f"{model_slug}-{profile_name}-{train_type}-{prefix}:latest"


def get_profile_stop_sequences(profile_name):
    if profile_name == "glm-edge-4b":
        return GLM_STOP_SEQUENCES
    return None


def select_profile_rma_prompt_template(model_name):
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_RMA_INFERENCE_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_RMA_INFERENCE_GEMMA
    if model_name == "microsoft/Phi-4-mini-instruct":
        return SFT_RMA_INFERENCE_PHI4
    if "Qwen/" in model_name:
        return SFT_RMA_INFERENCE_QWEN3
    return None


def select_profile_plan_prompt_template(model_name):
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_REWRITE_INFERENCE_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_REWRITE_INFERENCE_GEMMA
    return None


def get_legacy_model_configs():
    config = {
        "phi4": {
            "model_name": "phi4-rma:latest",
            "prompt_template": SFT_RMA_INFERENCE_PHI4,
            "prompt_renderer": "template",
            "plan_model_name": "phi4-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_PHI4,
            "plan_prompt_renderer": "template",
        },
        "llama3": {
            "model_name": "llama3-rma:latest",
            "prompt_template": SFT_RMA_INFERENCE_LLAMA,
            "prompt_renderer": "template",
            "plan_model_name": "llama3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
            "plan_prompt_renderer": "template",
        },
        "qwen3": {
            "model_name": "qwen3-rma:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen3-1.7b": {
            "model_name": "qwen3-rma-1.7b:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite-1.7b:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen3-0.6b": {
            "model_name": "qwen3-rma-0.6b:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite-0.6b:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen25": {
            "model_name": "qwen25-rma:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen25-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
            "plan_prompt_renderer": "template",
        },
        "gemma": {
            "model_name": "gemma-rma:latest",
            "prompt_template": SFT_RMA_INFERENCE_GEMMA,
            "prompt_renderer": "template",
            "plan_model_name": "gemma-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_GEMMA,
            "plan_prompt_renderer": "template",
        },
        "qwen3-tctraining-e5": {
            "model_name": "qwen3-rma-tctraining-e5:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen3-tctraining-e6": {
            "model_name": "qwen3-rma-tctraining-e6:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen3-pure-e4": {
            "model_name": "qwen3-pure-e4:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen3-pure-e5": {
            "model_name": "qwen3-pure-e5:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "qwen25-pure-e4": {
            "model_name": "qwen25-pure-e4:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen2.5-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
            "plan_prompt_renderer": "template",
        },
        "phi4-new": {
            "model_name": "phi4-new:latest",
            "prompt_template": SFT_RMA_INFERENCE_PHI4,
            "prompt_renderer": "template",
            "plan_model_name": "phi4-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_PHI4,
            "plan_prompt_renderer": "template",
        },
        "llama3-pure-e4": {
            "model_name": "llama3-pure-e4:latest",
            "prompt_template": SFT_RMA_INFERENCE_LLAMA,
            "prompt_renderer": "template",
            "plan_model_name": "llama3-rewrite:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
            "plan_prompt_renderer": "template",
        },
        "qwen3-multitask": {
            "model_name": "qwen3-multitask:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen3-multitask:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN3,
            "plan_prompt_renderer": "template",
        },
        "phi4-multitask": {
            "model_name": "phi4-multitask:latest",
            "prompt_template": SFT_RMA_INFERENCE_PHI4,
            "prompt_renderer": "template",
            "plan_model_name": "phi4-multitask:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_PHI4,
            "plan_prompt_renderer": "template",
        },
        "llama3-multitask": {
            "model_name": "llama3-multitask:latest",
            "prompt_template": SFT_RMA_INFERENCE_LLAMA,
            "prompt_renderer": "template",
            "plan_model_name": "llama3-multitask:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_LLAMA,
            "plan_prompt_renderer": "template",
        },
        "qwen2.5-multitask": {
            "model_name": "qwen25-new:latest",
            "prompt_template": SFT_RMA_INFERENCE_QWEN3,
            "prompt_renderer": "template",
            "plan_model_name": "qwen25-new:latest",
            "plan_prompt_template": SFT_REWRITE_INFERENCE_QWEN25,
            "plan_prompt_renderer": "template",
        }
    }
    return config


def get_profile_model_config(profile_name):
    profile = RMA_MODEL_PROFILES[profile_name]
    prompt_model_name = profile.model_name
    prefix = profile.prefix
    stop = get_profile_stop_sequences(profile_name)
    rma_prompt_template = select_profile_rma_prompt_template(prompt_model_name)
    plan_prompt_template = select_profile_plan_prompt_template(prompt_model_name)
    config = {
        "model_name": infer_profile_ollama_model_name(
            profile_name,
            prompt_model_name,
            train_type="rma",
            prefix=prefix,
        ),
        "plan_model_name": infer_profile_ollama_model_name(
            profile_name,
            prompt_model_name,
            train_type="rewrite",
            prefix=prefix,
        ),
        "rewrite_stop": stop,
        "plan_stop": stop,
    }
    if rma_prompt_template is None:
        config.update(
            {
                "prompt_renderer": "chat_template",
                "prompt_tokenizer_name": prompt_model_name,
            }
        )
    else:
        config.update(
            {
                "prompt_renderer": "template",
                "prompt_template": rma_prompt_template,
            }
        )

    if plan_prompt_template is None:
        config.update(
            {
                "plan_prompt_renderer": "chat_template",
                "plan_prompt_tokenizer_name": prompt_model_name,
            }
        )
    else:
        config.update(
            {
                "plan_prompt_renderer": "template",
                "plan_prompt_template": plan_prompt_template,
            }
        )
    return config


def get_model_configs(model_family):
    config = get_legacy_model_configs()
    if model_family not in config:
        raise ValueError(
            "Invalid model_family: "
            f"{model_family}. Available keys: {', '.join(sorted(config.keys()))}"
        )
    return config[model_family]


def resolve_model_config(args):
    if args.profile and args.model_family:
        raise ValueError("Use either --profile or --model_family, not both.")
    if args.profile:
        config = get_profile_model_config(args.profile)
        label = args.profile
    elif args.model_family:
        config = get_model_configs(args.model_family)
        label = args.model_family
    else:
        raise ValueError("Either --profile or --model_family is required.")

    config = config.copy()
    config["label"] = label
    if args.rewrite_model:
        config["model_name"] = args.rewrite_model
    if args.plan_model:
        config["plan_model_name"] = args.plan_model
    if args.rewrite_prompt_tokenizer_name:
        config["prompt_tokenizer_name"] = args.rewrite_prompt_tokenizer_name
        config["prompt_renderer"] = "chat_template"
    if args.plan_prompt_tokenizer_name:
        config["plan_prompt_tokenizer_name"] = args.plan_prompt_tokenizer_name
        config["plan_prompt_renderer"] = "chat_template"
    return config


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


def render_chat_template(tokenizer, messages, add_generation_prompt):
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


def render_messages_as_plain_text(messages, add_generation_prompt):
    sections = []
    for message in messages:
        role = str(message.get("role", "")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    if add_generation_prompt:
        sections.append("Assistant:\n")
    return "\n\n".join(sections)


def render_model_messages(tokenizer, messages, add_generation_prompt, chat_template_fallback):
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer=tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "only for models that were trained with simple fallback."
        )
    return render_messages_as_plain_text(messages, add_generation_prompt=add_generation_prompt)


def load_prompt_tokenizer(tokenizer_name):
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def build_api_str(example, apis):
    api_lines = []
    for plan in ast.literal_eval(example["candidates"]):
        api_data = apis[plan].copy()
        api_lines.append(f"{plan}: {api_data}")
    return "\n".join(api_lines) + "\n"


def build_rewrite_prompt(example, model_config, rewrite_tokenizer, chat_template_fallback):
    data = {
        "conversation_history": example["conversation_history"],
        "query": example["query"],
    }
    data_json = json.dumps(data, ensure_ascii=False, indent=2)

    if model_config.get("prompt_renderer") == "chat_template":
        return render_model_messages(
            tokenizer=rewrite_tokenizer,
            messages=[
                {"role": "system", "content": RMA_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": data_json},
            ],
            add_generation_prompt=True,
            chat_template_fallback=chat_template_fallback,
        )

    return model_config["prompt_template"].format(data=data_json)


def build_plan_prompt(example, apis, model_config, rewritten_query, plan_tokenizer, chat_template_fallback):
    api_str = build_api_str(example, apis)

    if model_config.get("plan_prompt_renderer") == "chat_template":
        system_msg = PLANNING_SYSTEM_PROMPT.format(tools=api_str)
        return render_model_messages(
            tokenizer=plan_tokenizer,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"User Query: {rewritten_query}"},
            ],
            add_generation_prompt=True,
            chat_template_fallback=chat_template_fallback,
        )

    return model_config["plan_prompt_template"].format(
        tools=api_str,
        data=rewritten_query,
    )


def main():
    args = build_arg_parser().parse_args()

    model_config = resolve_model_config(args)
    sft_apis = read_apis("apis/simple_api.json", simple=True)
    data_files = get_data_files()
    test_keys = parse_test_keys(args.test_key, data_files)
    rewrite_tokenizer = load_prompt_tokenizer(model_config.get("prompt_tokenizer_name"))
    plan_tokenizer = load_prompt_tokenizer(model_config.get("plan_prompt_tokenizer_name"))

    print(f"config: {model_config['label']}")
    print(f"rewrite model: {model_config['model_name']}")
    print(f"planning model: {model_config['plan_model_name']}")
    print(f"rewrite prompt renderer: {model_config.get('prompt_renderer')}")
    print(f"planning prompt renderer: {model_config.get('plan_prompt_renderer')}")
    if model_config.get("prompt_tokenizer_name"):
        print(f"rewrite prompt tokenizer: {model_config['prompt_tokenizer_name']}")
    if model_config.get("plan_prompt_tokenizer_name"):
        print(f"planning prompt tokenizer: {model_config['plan_prompt_tokenizer_name']}")
    if model_config.get("rewrite_stop"):
        print(f"rewrite stop: {model_config['rewrite_stop']}")
    if model_config.get("plan_stop"):
        print(f"planning stop: {model_config['plan_stop']}")
    print(f"chat_template_fallback: {args.chat_template_fallback}")
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

                rewrite_prompt = build_rewrite_prompt(
                    ex,
                    model_config,
                    rewrite_tokenizer,
                    args.chat_template_fallback,
                )
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
                    rewrite_raw = generate_text(
                        rewrite_prompt,
                        model=model_config["model_name"],
                        host=args.rewrite_host,
                        num_predict=args.rewrite_num_predict,
                        stop=model_config.get("rewrite_stop"),
                        temperature=args.temperature,
                    )
                    rewritten_query = parse_rewrite_output_inference_style(
                        rewrite_raw,
                        stop_markers=model_config.get("rewrite_stop"),
                    )
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
                            model_config,
                            rewritten_query,
                            plan_tokenizer,
                            args.chat_template_fallback,
                        )
                        plan_raw = generate_text(
                            plan_prompt,
                            model=model_config["plan_model_name"],
                            host=args.plan_host,
                            num_predict=args.plan_num_predict,
                            stop=model_config.get("plan_stop"),
                            temperature=args.temperature,
                        )
                        plan_result = parse_model_output(
                            plan_raw,
                            stop_markers=model_config.get("plan_stop"),
                        )
                        plan_res = "pass" if plan_result.get("plan") == gt.get("plan") else "fail"
                        arg_res = "pass" if plan_result.get("arguments") == gt.get("arguments") else "fail"
                        all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
                    except Exception as e:
                        plan_result = {"error": str(e)}
                        print(f"Planning error: {e}, raw={plan_raw}")

                row = {
                    "test_key": test_key,
                    "conversation_history": ex.get("conversation_history"),
                    "query": ex.get("query"),
                    "gt_rewrited_query": ex.get("rewrited_query"),
                    "generated_rewrited_query": rewritten_query,
                    "rewrite_prompt": rewrite_prompt,
                    "rewrite_raw_generation": rewrite_raw,
                    "rewrite_generation": rewrite_result if rewrite_result is not None else {"error": rewrite_error, "raw": rewrite_raw},
                    "rewrite_error": rewrite_error,
                    "candidates": ex.get("candidates"),
                    "planning_prompt": plan_prompt,
                    "planning_raw_generation": plan_raw,
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
                title=f"{test_key}/{os.path.basename(file_path)} | model={model_config['label']}",
                test_type=f"{model_config['model_name']} -> {model_config['plan_model_name']}",
            )
            all_results.extend(file_results)

    result = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(
            df_split,
            title=f"split={test_key} | model={model_config['label']}",
            test_type=f"{model_config['model_name']} -> {model_config['plan_model_name']}",
        )
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(
        result,
        title=f"combined={combined_title} | model={model_config['label']}",
        test_type=f"{model_config['model_name']} -> {model_config['plan_model_name']}",
    )
    print_turn_macro_summary(result, title=f"combined={combined_title}", metric="all")
    result.to_csv(args.o, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
