import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


GLM_STOP_SEQUENCES = [
    "<|observation|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
]

BASELINE_SYSTEM_HISTORY_PROMPT = (
    "You are a planner for a mobile assistant. Predict the next action plan and "
    "its arguments from the conversation history and the current user query. Use "
    "the conversation history only when it is relevant for resolving references. "
    "Return exactly one JSON object with keys \"plan\" and \"arguments\". "
    "\"plan\" is the action/tool name to execute, and \"arguments\" is an object. "
    "Only use argument values that are explicitly stated or can be reasonably "
    "inferred from the query or conversation history. If no action is appropriate, "
    "return {\"plan\":\"None\",\"arguments\":{}}."
)

BASELINE_SYSTEM_REWRITE_PROMPT = (
    "You are a planner for a mobile assistant. Predict the next action plan and "
    "its arguments from the user query. Return exactly one JSON object with keys "
    "\"plan\" and \"arguments\". \"plan\" is the action/tool name to execute, "
    "and \"arguments\" is an object. Only use argument values that are explicitly "
    "stated or can be reasonably inferred from the query. If no action is "
    "appropriate, return {\"plan\":\"None\",\"arguments\":{}}."
)


@dataclass(frozen=True)
class BaselinePromptProfile:
    prompt_model_name: str
    prefix: str
    stop: list[str] | None = None


BASELINE_PROMPT_PROFILES = {
    "qwen": BaselinePromptProfile("Qwen/Qwen3-4B", "baseline_all_linear"),
    "phi": BaselinePromptProfile("microsoft/Phi-4-mini-instruct", "baseline_1st"),
    "llama": BaselinePromptProfile("meta-llama/Llama-3.2-3B-Instruct", "baseline_all_linear"),
    "generic": BaselinePromptProfile("", "baseline_all_linear"),
    "glm-edge-1.5b": BaselinePromptProfile("zai-org/glm-edge-1.5b-chat", "baseline_all_linear"),
    "glm-edge-4b": BaselinePromptProfile(
        "zai-org/glm-edge-4b-chat",
        "baseline_all_linear",
        stop=GLM_STOP_SEQUENCES,
    ),
    "smollm2-1.7b": BaselinePromptProfile("HuggingFaceTB/SmolLM2-1.7B", "baseline_simple_template"),
    "smollm2-1.7b-instruct": BaselinePromptProfile(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "baseline_all_linear",
    ),
    "smollm3-3b": BaselinePromptProfile("HuggingFaceTB/SmolLM3-3B", "baseline_all_linear"),
    "falcon3-1b": BaselinePromptProfile("tiiuae/Falcon3-1B-Instruct", "baseline_all_linear"),
    "falcon3-1b-base": BaselinePromptProfile("tiiuae/Falcon3-1B-Base", "baseline_simple_template"),
    "falcon3-3b": BaselinePromptProfile("tiiuae/Falcon3-3B-Instruct", "baseline_all_linear"),
    "falcon3-3b-base": BaselinePromptProfile("tiiuae/Falcon3-3B-Base", "baseline_simple_template"),
    "exaone4-1.2b": BaselinePromptProfile("LGAI-EXAONE/EXAONE-4.0-1.2B", "baseline_all_linear"),
    "olmo2-1b": BaselinePromptProfile("allenai/OLMo-2-0425-1B", "baseline_simple_template"),
    "olmo2-1b-instruct": BaselinePromptProfile("allenai/OLMo-2-0425-1B-Instruct", "baseline_all_linear"),
    "granite3.3-2b": BaselinePromptProfile("ibm-granite/granite-3.3-2b-instruct", "baseline_all_linear"),
    "lfm2.5-1.2b": BaselinePromptProfile("LiquidAI/LFM2.5-1.2B-Instruct", "baseline_all_linear"),
}


DATA_FILES = {
    # "base": [
    #     "datasets/tc/scale/it5_complex_1_tc.tsv",
    #     "datasets/tc/scale/it3_complex_1_tc.tsv",
    #     "datasets/tc/scale/it4_complex_1_tc.tsv",
    #     "datasets/tc/scale/it4_complex_2_tc.tsv",
    #     "datasets/tc/scale/it2_nonNR_tc.tsv",
    #     "datasets/tc/scale/it5_complex_2_tc.tsv",
    #     "datasets/tc/scale/it5_complex_3_tc.tsv",        
    #     "datasets/tc/scale/it3_nonNR_tc.tsv",
    #     "datasets/tc/scale/it4_nonNR_tc.tsv",
    #     "datasets/tc/scale/it5_nonNR_tc.tsv",
    # ],
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


def get_arg_parse():
    parser = argparse.ArgumentParser(description="Ollama inference for baseline planner models")
    parser.add_argument("--t", type=str, default=None, help="Test type, e.g. history-glm-edge-4b")
    parser.add_argument("--profile", choices=sorted(BASELINE_PROMPT_PROFILES.keys()), default=None)
    parser.add_argument("--train_type", choices=["history", "rewrite"], default="history")
    parser.add_argument("--prompt_model_name", default=None, help="Override HF tokenizer model for prompt rendering")
    parser.add_argument("--prefix", default=None, help="Override baseline output prefix used in Ollama model naming")
    parser.add_argument("--model", type=str, default=None, help="Override Ollama model name")
    parser.add_argument("--o", type=str, required=True, help="Output TSV path")
    parser.add_argument("--test_key", type=str, default="", help="Dataset split key(s), comma separated")
    parser.add_argument("--data_files", type=str, default=None, help="Comma-separated TSV files. Overrides --test_key files.")
    parser.add_argument("--host", type=str, default="http://localhost:11436", help="Ollama host URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_predict", type=int, default=512)
    parser.add_argument("--chat_template_fallback", choices=["simple", "error"], default="simple")
    parser.add_argument("--d", action="store_true", help="Print debug output")
    return parser.parse_args()


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


def infer_baseline_ollama_model_name(profile_name: str, prompt_model_name: str, train_type: str, prefix: str) -> str:
    model_slug = sanitize_model_slug(prompt_model_name)
    return f"{model_slug}-{profile_name}-{train_type}-{prefix}:latest"


def build_config_from_profile(profile_name: str, train_type: str, args):
    profile = BASELINE_PROMPT_PROFILES[profile_name]
    prompt_model_name = args.prompt_model_name or profile.prompt_model_name
    if not prompt_model_name:
        raise ValueError("--prompt_model_name is required when --profile generic is used.")

    prefix = args.prefix or profile.prefix
    config = {
        "profile": profile_name,
        "model_name": infer_baseline_ollama_model_name(
            profile_name,
            prompt_model_name,
            train_type,
            prefix,
        ),
        "prompt_model_name": prompt_model_name,
        "prompt_mode": train_type,
        "chat_template_fallback": args.chat_template_fallback,
    }
    if profile.stop:
        config["stop"] = profile.stop
    return config


def build_test_type_configs(args):
    configs = {}
    for profile_name in BASELINE_PROMPT_PROFILES:
        if profile_name == "generic":
            continue
        for train_type in ("history", "rewrite"):
            config = build_config_from_profile(profile_name, train_type, args)
            configs[f"{train_type}-{profile_name}"] = config
            if train_type == "history":
                configs[f"base-{profile_name}"] = config
    return configs


def resolve_config(args):
    if args.t:
        configs = build_test_type_configs(args)
        config = configs.get(args.t)
        if config is None:
            valid_test_types = ", ".join(sorted(configs.keys()))
            raise ValueError(f"Invalid test type: {args.t}. Available test types: {valid_test_types}")
        return config

    if not args.profile:
        raise ValueError("Use either --profile or --t. Example: --profile glm-edge-4b --train_type history")
    return build_config_from_profile(args.profile, args.train_type, args)


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


def render_simple_inference_prompt(system_msg, user_content):
    return (
        f"System:\n{system_msg}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:\n"
    )


def load_prompt_tokenizer(config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config["prompt_model_name"],
        trust_remote_code=True,
    )


def build_baseline_prompt_fields(example, prompt_mode):
    if prompt_mode in ("base", "history"):
        system_msg = BASELINE_SYSTEM_HISTORY_PROMPT
        user_content = (
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    elif prompt_mode == "rewrite":
        system_msg = BASELINE_SYSTEM_REWRITE_PROMPT
        user_content = f"User Query: {example['rewrited_query']}"
    else:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    return system_msg, user_content


def render_baseline_inference_prompt(example, tokenizer, config):
    system_msg, user_content = build_baseline_prompt_fields(
        example=example,
        prompt_mode=config["prompt_mode"],
    )
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer,
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ],
            add_generation_prompt=True,
        )

    if config.get("chat_template_fallback", "simple") == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "only for models trained with simple fallback."
        )
    return render_simple_inference_prompt(system_msg, user_content)


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


def generate_text(prompt, model, host, num_predict=512, stop=None, temperature=0.0):
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "stream": False,
    }
    if stop:
        payload["options"]["stop"] = stop

    response = requests.post(
        f"{host}/api/generate",
        json=payload,
        timeout=300,
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
    except Exception as exc:
        print("JSON extraction/parsing failed:", exc)
        return None


def parse_model_output(raw, stop_markers=None):
    parse_source = truncate_at_stop_markers(raw, stop_markers)
    try:
        result = ast.literal_eval(parse_source)
    except Exception as parse_exc:
        result = extract_json_from_markdown(parse_source)
        if result is None:
            raise ValueError(f"Failed to parse model output: {parse_exc}")

    if not isinstance(result, dict):
        raise ValueError(f"Parsed model output is not a dict: {type(result).__name__}")
    return result


def parse_answer_value(value):
    if isinstance(value, dict):
        return value

    parsed = value
    for _ in range(2):
        if not isinstance(parsed, str):
            break
        try:
            parsed = ast.literal_eval(parsed)
        except Exception:
            try:
                parsed = json.loads(parsed)
            except Exception:
                break
    return parsed if isinstance(parsed, dict) else {}


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
    if title:
        print(f"\n## Performance for {title}\n")

    metric_rows = [
        ("Plan       Accuracy", df["plan"].eq("pass").mean()),
        ("Arguments  Accuracy", df["arguments"].eq("pass").mean()),
        ("All        Accuracy", df["all"].eq("pass").mean()),
        ("All        Plan Macro Accuracy", compute_plan_macro(df, metric="all")),
    ]
    for label, acc in metric_rows:
        print(f"{label} : {acc * 100:.2f}%")

    print("-" * 40)
    Path("logs").mkdir(parents=True, exist_ok=True)
    with open("logs/ollama_inference_log.txt", "a", encoding="utf-8") as file:
        if title:
            file.write(f"\n## Performance for {title}, {test_type}\n")
        for label, acc in metric_rows:
            file.write(f"{label} : {acc * 100:.2f}%\n")
        file.write("-" * 40 + "\n")

    if detail:
        df["gt_plan"] = df["gt"].apply(lambda x: x.get("plan"))
        detail_df = (
            df.groupby("gt_plan")[["plan", "arguments", "all"]]
            .apply(lambda sub: sub.eq("pass").mean().round(2))
            .reset_index()
        )
        print(detail_df.to_string(index=False))


def parse_test_keys(test_key_arg, data_files):
    if not test_key_arg:
        raise ValueError("`--test_key` is required. Example: --test_key base or --test_key base,complex")

    test_keys = [key.strip() for key in test_key_arg.split(",") if key.strip()]
    deduped_test_keys = []
    for key in test_keys:
        if key not in data_files:
            raise ValueError(f"Invalid test_key: {key}. Available keys: {', '.join(sorted(data_files.keys()))}")
        if key not in deduped_test_keys:
            deduped_test_keys.append(key)
    return deduped_test_keys


def resolve_data_files(args):
    if args.data_files:
        return {"custom": [path.strip() for path in args.data_files.split(",") if path.strip()]}, ["custom"]
    test_keys = parse_test_keys(args.test_key, DATA_FILES)
    return DATA_FILES, test_keys


def preprocess_example(example, config, prompt_tokenizer):
    prompt = render_baseline_inference_prompt(
        example=example,
        tokenizer=prompt_tokenizer,
        config=config,
    )
    return {
        "strprompt": prompt,
        "stranswer": json.dumps(parse_answer_value(example["answer"]), ensure_ascii=False),
        "rewrited_query": example.get("rewrited_query"),
        "query": example.get("query"),
        "conversation_history": example.get("conversation_history"),
        "source_candidates": example.get("candidates"),
    }


def main(args):
    from datasets import load_dataset

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config = resolve_config(args)
    model_name = args.model or config["model_name"]
    prompt_tokenizer = load_prompt_tokenizer(config)
    data_files, test_keys = resolve_data_files(args)

    print(model_name)
    print(args.host)
    print(f"prompt_model_name: {config['prompt_model_name']}")
    print(f"prompt_mode: {config['prompt_mode']}")
    print(f"Selected test_keys: {test_keys}")

    all_results = []
    split_results = {key: [] for key in test_keys}
    for test_key in test_keys:
        print(f"\n# Running split: {test_key}")
        print(data_files[test_key])
        for file_path in data_files[test_key]:
            ds = load_dataset("csv", data_files={"tc": [file_path]}, delimiter="\t")["tc"]
            proc = ds.map(
                lambda example: preprocess_example(
                    example,
                    config=config,
                    prompt_tokenizer=prompt_tokenizer,
                )
            )

            print(proc[0]["strprompt"])
            file_results = []
            for ex in tqdm(proc, desc=f"Processing {test_key}/{os.path.basename(file_path)}"):
                prompt = ex["strprompt"]
                raw = ""
                gt = {}
                parse_error = ""

                try:
                    raw = generate_text(
                        prompt,
                        model=model_name,
                        host=args.host,
                        num_predict=args.num_predict,
                        stop=config.get("stop"),
                        temperature=args.temperature,
                    )
                    result = parse_model_output(raw, stop_markers=config.get("stop"))
                    gt = parse_answer_value(ex["stranswer"])

                    plan_res = "pass" if result.get("plan") == gt.get("plan") else "fail"
                    arg_res = "pass" if result.get("arguments") == gt.get("arguments") else "fail"
                    all_res = "pass" if plan_res == "pass" and arg_res == "pass" else "fail"
                except Exception as exc:
                    result = {"error": str(exc)}
                    parse_error = str(exc)
                    print(f"Error: {exc}, {raw}")
                    plan_res = "fail"
                    arg_res = "fail"
                    all_res = "fail"

                row = {
                    "test_key": test_key,
                    "conversation_history": ex.get("conversation_history"),
                    "query": ex.get("query"),
                    "rewrited_query": ex.get("rewrited_query"),
                    "source_candidates": ex.get("source_candidates"),
                    "raw_generation": raw,
                    "generation": result,
                    "parse_error": parse_error,
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
            print_eval(df_file, title=f"{test_key}/{os.path.basename(file_path)}", test_type=model_name)
            all_results.extend(file_results)

    result = pd.DataFrame(all_results)
    for test_key in test_keys:
        df_split = pd.DataFrame(split_results[test_key])
        print_eval(df_split, title=f"split={test_key}", test_type=model_name)
        print_turn_macro_summary(df_split, title=f"split={test_key}", metric="all")

    combined_title = ",".join(test_keys)
    print_eval(result, title=f"combined={combined_title}", test_type=model_name)
    print_turn_macro_summary(result, title=f"combined={combined_title}", metric="all")
    result.to_csv(out_path, sep="\t", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main(get_arg_parse())
