# -*- coding: utf-8 -*-
"""
Vanilla baseline planner trainer.

This entrypoint intentionally removes candidate tools/tool schemas from the
training prompt. The model sees only the conversation/query and learns to emit
the gold {"plan": ..., "arguments": ...} object directly.

Examples:
  python train_baseline_integrated.py --profile qwen
  python train_baseline_integrated.py --profile phi
  python train_baseline_integrated.py --profile glm-edge-4b
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from train_sentence_rewriter import (
    DEFAULT_LLAMA_CPP_DIR,
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    DEFAULT_OUTPUT_ROOT,
    infer_model_slug,
    postprocess_trained_model,
    resolve_model_storage_paths,
)


QWEN_DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
PHI_DEFAULT_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
LLAMA_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

GENERIC_PROFILE_NAMES = {
    "generic",
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

KNOWN_UNSUPPORTED_GGUF_ARCHITECTURES = {
    "Exaone4ForCausalLM",
    "Lfm2ForCausalLM",
    "SmolLM3ForCausalLM",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LEGACY_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
LEGACY_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

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
class ProfileDefaults:
    model_name: str
    train_type: str
    prefix: str
    max_len: int


PROFILE_DEFAULTS = {
    "qwen": ProfileDefaults(
        model_name=QWEN_DEFAULT_MODEL_NAME,
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "phi": ProfileDefaults(
        model_name=PHI_DEFAULT_MODEL_NAME,
        train_type="history",
        prefix="baseline_1st",
        max_len=1024,
    ),
    "llama": ProfileDefaults(
        model_name=LLAMA_DEFAULT_MODEL_NAME,
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "generic": ProfileDefaults(
        model_name="",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "glm-edge-1.5b": ProfileDefaults(
        model_name="zai-org/glm-edge-1.5b-chat",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "glm-edge-4b": ProfileDefaults(
        model_name="zai-org/glm-edge-4b-chat",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "smollm2-1.7b": ProfileDefaults(
        model_name="HuggingFaceTB/SmolLM2-1.7B",
        train_type="history",
        prefix="baseline_simple_template",
        max_len=1536,
    ),
    "smollm2-1.7b-instruct": ProfileDefaults(
        model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "smollm3-3b": ProfileDefaults(
        model_name="HuggingFaceTB/SmolLM3-3B",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "falcon3-1b": ProfileDefaults(
        model_name="tiiuae/Falcon3-1B-Instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "falcon3-1b-base": ProfileDefaults(
        model_name="tiiuae/Falcon3-1B-Base",
        train_type="history",
        prefix="baseline_simple_template",
        max_len=1536,
    ),
    "falcon3-3b": ProfileDefaults(
        model_name="tiiuae/Falcon3-3B-Instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "falcon3-3b-base": ProfileDefaults(
        model_name="tiiuae/Falcon3-3B-Base",
        train_type="history",
        prefix="baseline_simple_template",
        max_len=1536,
    ),
    "exaone4-1.2b": ProfileDefaults(
        model_name="LGAI-EXAONE/EXAONE-4.0-1.2B",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "olmo2-1b": ProfileDefaults(
        model_name="allenai/OLMo-2-0425-1B",
        train_type="history",
        prefix="baseline_simple_template",
        max_len=1536,
    ),
    "olmo2-1b-instruct": ProfileDefaults(
        model_name="allenai/OLMo-2-0425-1B-Instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "granite3.3-2b": ProfileDefaults(
        model_name="ibm-granite/granite-3.3-2b-instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
    "lfm2.5-1.2b": ProfileDefaults(
        model_name="LiquidAI/LFM2.5-1.2B-Instruct",
        train_type="history",
        prefix="baseline_all_linear",
        max_len=1536,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vanilla baseline trainer for planner models without tool/candidate prompt input."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Which model profile to run.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override the profile's legacy default model_name.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(LEGACY_TOOLS_PATH),
        help="Kept for CLI compatibility; baseline prompts do not load tools.",
    )
    parser.add_argument(
        "--train_dir",
        default=str(LEGACY_TRAIN_DIR),
        help="Directory containing training TSV files.",
    )
    parser.add_argument(
        "--train_type",
        default=None,
        choices=["history", "rewrite"],
        help="Override the profile's legacy train_type.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Override the profile's legacy output prefix.",
    )
    parser.add_argument(
        "--max_len",
        default=None,
        type=int,
        help="Override the profile's legacy token truncation length.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Explicit training output dir. If omitted, a run dir is created under output_root/model_name.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for HF cache, checkpoints, and final adapter.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=float,
        help="Training epochs. Legacy default is 3.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=2,
        type=int,
        help="Legacy default is 2.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=4,
        type=int,
        help="Legacy default is 4.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="Legacy default is 5e-5.",
    )
    parser.add_argument(
        "--lora_r",
        default=None,
        type=int,
        help="Override LoRA rank. Generic/all-linear default is 16; use 4 for about one quarter of the LoRA params.",
    )
    parser.add_argument(
        "--lora_alpha",
        default=None,
        type=int,
        help="Override LoRA alpha. If using --lora_r 4, alpha 8 keeps the existing alpha/r ratio.",
    )
    parser.add_argument(
        "--lora_dropout",
        default=None,
        type=float,
        help="Override LoRA dropout.",
    )
    parser.add_argument(
        "--lora_target_modules",
        default=None,
        help="Override LoRA targets. Use 'all-linear' or a comma-separated list like q_proj,v_proj.",
    )
    parser.add_argument(
        "--logging_steps",
        default=50,
        type=int,
        help="Legacy default is 50.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=8,
        type=int,
        help="Legacy default is 8.",
    )
    parser.add_argument(
        "--dry_run_preprocess",
        action="store_true",
        help="Load tokenizer/config, preprocess one example, and exit before model loading/training.",
    )
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for tokenizers without chat_template. `simple` enables base-model SFT.",
    )
    parser.add_argument(
        "--strict_postprocess",
        action="store_true",
        help="Attempt GGUF/Ollama postprocess even for model architectures known to be unsupported locally.",
    )
    parser.add_argument(
        "--epoch_eval_path",
        default=None,
        help="Optional comma-separated TSV path(s) to evaluate with generation at each epoch end.",
    )
    parser.add_argument(
        "--epoch_eval_sample_size",
        default=50,
        type=int,
        help="Number of epoch-eval examples to use. Use <=0 for all rows.",
    )
    parser.add_argument(
        "--epoch_eval_max_new_tokens",
        default=128,
        type=int,
        help="Max new tokens for epoch-end generation evaluation.",
    )
    parser.add_argument(
        "--skip_postprocess",
        action="store_true",
        help="Skip merge, GGUF conversion, and Ollama registration after training.",
    )
    parser.add_argument(
        "--postprocess_only",
        "--postprocessing_only",
        dest="postprocess_only",
        action="store_true",
        help="Skip training and run only merge, GGUF conversion, and Ollama registration from existing outputs.",
    )
    parser.add_argument(
        "--export_checkpoint_epoch",
        type=int,
        default=None,
        help="Export the Nth epoch checkpoint instead of the final checkpoint.",
    )
    parser.add_argument(
        "--export_source",
        choices=["final_checkpoint", "final_adapter"],
        default="final_checkpoint",
        help="Use the final epoch checkpoint or the final saved adapter directory for export.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Override base model ID instead of reading adapter_config.json.",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Local directory containing the base model. If missing, a default path is inferred.",
    )
    parser.add_argument(
        "--merged_dir",
        default=None,
        help="Output directory for the merged HF model.",
    )
    parser.add_argument(
        "--gguf_path",
        default=None,
        help="Output path for the merged GGUF file.",
    )
    parser.add_argument(
        "--modelfile",
        default=None,
        help="Path to write the Ollama Modelfile.",
    )
    parser.add_argument(
        "--ollama_model_name",
        default=None,
        help="Model name to register in Ollama.",
    )
    parser.add_argument(
        "--ollama_host",
        default=DEFAULT_OLLAMA_HOST,
        help="OLLAMA_HOST value for ollama create.",
    )
    parser.add_argument(
        "--ollama_models_dir",
        default=DEFAULT_OLLAMA_MODELS_DIR,
        help="OLLAMA_MODELS directory for ollama create.",
    )
    parser.add_argument(
        "--ollama_bin",
        default=DEFAULT_OLLAMA_BIN,
        help="Path to the ollama binary.",
    )
    parser.add_argument(
        "--ollama_lib_path",
        default=DEFAULT_OLLAMA_LIB_PATH,
        help="LD_LIBRARY_PATH to use for ollama create.",
    )
    parser.add_argument(
        "--llama_cpp_dir",
        default=DEFAULT_LLAMA_CPP_DIR,
        help="Path to llama.cpp containing convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--merge_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype to use when merging the LoRA adapter.",
    )
    parser.add_argument(
        "--gguf_outtype",
        default="bf16",
        help="GGUF output type for llama.cpp convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--trust_remote_code",
        choices=["auto", "true", "false"],
        default="auto",
        help="Whether to trust remote model/tokenizer code while merging.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download of the base model when download is needed.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token for private or gated repos.",
    )
    return parser.parse_args()


def resolve_profile_value(args, name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return getattr(PROFILE_DEFAULTS[args.profile], name)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            path = cwd_path
        else:
            path = PROJECT_ROOT / path
    return path.resolve()


def resolve_output_dir(output_dir: str | None, default_name: str) -> str:
    target = default_name if output_dir is None else output_dir
    return str(resolve_path(target))


def resolve_storage_paths(args, model_name: str, default_output_name: str):
    output_root = Path(args.output_root).expanduser().resolve()
    model_root_dir, default_output_dir, cache_dir = resolve_model_storage_paths(
        output_root,
        model_name,
        default_output_name,
    )
    output_dir = (
        Path(resolve_output_dir(args.output_dir, default_output_name))
        if args.output_dir is not None
        else default_output_dir
    )
    return output_root, model_root_dir, output_dir, cache_dir


def apply_postprocess_defaults(args, output_root: Path, model_name: str, profile: str, train_type: str, prefix: str):
    model_slug = infer_model_slug(model_name)
    artifact_root = output_root.parent
    artifact_stem = f"{model_slug}-{profile}-{train_type}-{prefix}"

    merged_dir = (
        Path(args.merged_dir).expanduser().resolve()
        if args.merged_dir
        else (artifact_root / f"{artifact_stem}-merged")
    )
    gguf_path = (
        Path(args.gguf_path).expanduser().resolve()
        if args.gguf_path
        else Path(f"{merged_dir}.gguf")
    )
    modelfile_path = (
        Path(args.modelfile).expanduser().resolve()
        if args.modelfile
        else (Path.home() / f"Modelfile.{gguf_path.name}")
    )

    args.merged_dir = str(merged_dir)
    args.gguf_path = str(gguf_path)
    args.modelfile = str(modelfile_path)
    if not args.ollama_model_name:
        args.ollama_model_name = artifact_stem


def maybe_run_postprocess_only(
    args,
    output_dir: Path,
    output_root: Path,
    cache_dir: Path,
    model_name: str,
    preexisting_checkpoint_names: set[str],
):
    if not args.postprocess_only:
        return False

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Training output directory not found for --postprocess_only: {output_dir}"
        )

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=args.num_train_epochs,
        final_global_step=None,
        cache_dir=cache_dir,
    )
    return True


def read_simple_apis(api_file: str | Path):
    with open(resolve_path(api_file), "r", encoding="utf-8") as file:
        return json.load(file)


def collect_legacy_train_files(train_dir: str | Path):
    train_dir_path = resolve_path(train_dir)
    train_files = sorted(
        str(path)
        for path in train_dir_path.glob("*.tsv")
    )

    additional_dir = train_dir_path / "additional"
    if additional_dir.is_dir():
        train_files.extend(
            sorted(
                str(path)
                for path in additional_dir.glob("*.tsv")
            )
        )

    return train_files


def print_max_total_length(processed_train):
    max_total_length = 0
    for example in processed_train:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_total_length:
            max_total_length = total_length
    print(f"input_ids + labels 최대 길이: {max_total_length}")


def resolve_trust_remote_code_arg(args, default: bool = True) -> bool:
    if args.trust_remote_code == "true":
        return True
    if args.trust_remote_code == "false":
        return False
    return default


def get_config_architecture(config) -> str | None:
    architectures = getattr(config, "architectures", None)
    if architectures:
        return architectures[0]
    return None


def converter_supports_architecture(llama_cpp_dir: str | Path, architecture: str) -> bool:
    llama_cpp_path = Path(llama_cpp_dir).expanduser().resolve()
    converter_path = llama_cpp_path / "convert_hf_to_gguf.py"
    if not converter_path.exists():
        return False

    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(llama_cpp_path)!r})\n"
        "import convert_hf_to_gguf as c\n"
        f"c.ModelBase.from_model_architecture({architecture!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(llama_cpp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.returncode == 0


def maybe_auto_skip_unsupported_postprocess(args, config):
    architecture = get_config_architecture(config)
    if (
        architecture in KNOWN_UNSUPPORTED_GGUF_ARCHITECTURES
        and not args.skip_postprocess
        and not args.strict_postprocess
    ):
        if converter_supports_architecture(args.llama_cpp_dir, architecture):
            print(
                "[postprocess] selected llama.cpp converter supports "
                f"{architecture}; postprocess will proceed."
            )
            return
        args.skip_postprocess = True
        print(
            "[postprocess] auto-skipped because selected llama.cpp converter does not "
            f"register {architecture}: {args.llama_cpp_dir}. Use a newer "
            "--llama_cpp_dir or --strict_postprocess to attempt it anyway."
        )


def configure_tokenizer_padding(tokenizer, max_len: int) -> int:
    added_tokens = 0
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_len
    return added_tokens


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


def render_simple_prompt(
    tokenizer,
    system_msg: str,
    user_content: str,
    assistant_content: str,
) -> tuple[str, str]:
    prompt_prefix = (
        f"System:\n{system_msg}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:\n"
    )
    eos = tokenizer.eos_token or ""
    return prompt_prefix + assistant_content + eos, prompt_prefix


def build_prompt_fields(example, apis, train_type: str) -> tuple[str, str, str]:
    if train_type == "history":
        system_msg = BASELINE_SYSTEM_HISTORY_PROMPT
        user_content = (
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    else:
        system_msg = BASELINE_SYSTEM_REWRITE_PROMPT
        user_content = f"User Query: {example['rewrited_query']}"

    assistant_content = format_assistant_answer(example["answer"])
    return system_msg, user_content, assistant_content


def render_chat_prompt_pair(
    tokenizer,
    system_msg: str,
    user_content: str,
    assistant_content: str,
    chat_template_fallback: str,
) -> tuple[str, str]:
    if getattr(tokenizer, "chat_template", None):
        prefix_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        full_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        prompt_prefix = render_chat_template(
            tokenizer,
            prefix_messages,
            add_generation_prompt=True,
        )
        prompt_with_answer = render_chat_template(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )
        return prompt_with_answer, prompt_prefix

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "for base-model SFT."
        )

    return render_simple_prompt(tokenizer, system_msg, user_content, assistant_content)


def render_inference_prompt(
    tokenizer,
    system_msg: str,
    user_content: str,
    chat_template_fallback: str,
) -> str:
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
            "for base-model SFT."
        )

    return (
        f"System:\n{system_msg}\n\n"
        f"User:\n{user_content}\n\n"
        "Assistant:\n"
    )


def tokenize_prompt_pair(tokenizer, prompt: str, prompt_prefix: str, max_len: int):
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    input_ids = tokenized["input_ids"]
    label_start = len(
        tokenizer(
            prompt_prefix,
            add_special_tokens=False,
        )["input_ids"]
    )
    label_start = min(label_start, len(input_ids))
    labels = [-100] * label_start + input_ids[label_start:]
    return input_ids, labels


def make_generic_chat_preprocess_fn(
    tokenizer,
    apis,
    train_type: str,
    max_len: int,
    chat_template_fallback: str,
):
    def preprocess_example(example):
        system_msg, user_content, assistant_content = build_prompt_fields(
            example,
            apis,
            train_type,
        )

        prompt_with_answer, prompt_prefix = render_chat_prompt_pair(
            tokenizer,
            system_msg,
            user_content,
            assistant_content,
            chat_template_fallback,
        )
        input_ids, labels = tokenize_prompt_pair(
            tokenizer,
            prompt_with_answer,
            prompt_prefix,
            max_len,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt_with_answer,
            "stranswer": assistant_content,
        }

    return preprocess_example


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


def format_assistant_answer(answer) -> str:
    parsed = parse_answer_value(answer)
    if isinstance(parsed, dict) and ("plan" in parsed or "arguments" in parsed):
        return json.dumps(parsed, ensure_ascii=False)
    if isinstance(answer, dict):
        return json.dumps(answer, ensure_ascii=False)
    return str(answer)


def parse_generated_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        text = text.replace("```json", "```")
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1].strip()

    match = None
    depth = 0
    start = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                match = text[start:index + 1]
                break

    candidate = match or text
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        parsed = json.loads(candidate)
    return parsed if isinstance(parsed, dict) else {}


def resolve_epoch_eval_paths(path_arg: str | None) -> list[str]:
    if not path_arg:
        return []
    return [
        str(resolve_project_path(path))
        for path in path_arg.split(",")
        if path.strip()
    ]


def load_epoch_eval_dataset(path_arg: str | None, sample_size: int):
    data_files = resolve_epoch_eval_paths(path_arg)
    if not data_files:
        return None

    dataset = load_dataset("csv", data_files={"eval": data_files}, delimiter="\t")["eval"]
    if sample_size > 0:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
    return dataset


class EpochGenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset,
        apis,
        train_type: str,
        max_len: int,
        chat_template_fallback: str,
        max_new_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.apis = apis
        self.train_type = train_type
        self.max_len = max_len
        self.chat_template_fallback = chat_template_fallback
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or not state.is_world_process_zero:
            return control

        model_was_training = model.training
        model.eval()

        plan_pass = 0
        arg_pass = 0
        all_pass = 0
        total = 0
        parse_fail = 0
        device = next(model.parameters()).device

        with torch.no_grad():
            for example in self.eval_dataset:
                total += 1
                system_msg, user_content, _ = build_prompt_fields(
                    example,
                    self.apis,
                    self.train_type,
                )
                prompt = render_inference_prompt(
                    self.tokenizer,
                    system_msg,
                    user_content,
                    self.chat_template_fallback,
                )
                model_inputs = self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                model_inputs = {
                    key: value.to(device)
                    for key, value in model_inputs.items()
                }
                input_length = model_inputs["input_ids"].shape[-1]
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated_text = self.tokenizer.decode(
                    generated_ids[0][input_length:],
                    skip_special_tokens=False,
                )

                try:
                    prediction = parse_generated_json(generated_text)
                except Exception:
                    prediction = {}
                    parse_fail += 1

                gold = parse_answer_value(example["answer"])
                plan_ok = prediction.get("plan") == gold.get("plan")
                args_ok = prediction.get("arguments") == gold.get("arguments")
                plan_pass += int(plan_ok)
                arg_pass += int(args_ok)
                all_pass += int(plan_ok and args_ok)

        if model_was_training:
            model.train()

        if total:
            epoch_value = state.epoch if state.epoch is not None else -1
            print(
                "[epoch_eval] "
                f"epoch={epoch_value:.2f} samples={total} "
                f"plan={plan_pass / total * 100:.2f}% "
                f"arguments={arg_pass / total * 100:.2f}% "
                f"all={all_pass / total * 100:.2f}% "
                f"parse_fail={parse_fail}"
            )
        return control


def build_epoch_eval_callbacks(
    args,
    tokenizer,
    apis,
    train_type: str,
    max_len: int,
    chat_template_fallback: str,
):
    eval_dataset = load_epoch_eval_dataset(args.epoch_eval_path, args.epoch_eval_sample_size)
    if eval_dataset is None:
        return []
    print(
        "[epoch_eval] loaded "
        f"{len(eval_dataset)} examples from {resolve_epoch_eval_paths(args.epoch_eval_path)}"
    )
    return [
        EpochGenerationEvalCallback(
            tokenizer=tokenizer,
            eval_dataset=eval_dataset,
            apis=apis,
            train_type=train_type,
            max_len=max_len,
            chat_template_fallback=chat_template_fallback,
            max_new_tokens=args.epoch_eval_max_new_tokens,
        )
    ]


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        label_pad_token_id = -100
        max_label_length = max(len(feature["labels"]) for feature in features)
        for feature in features:
            padding_length = max_label_length - len(feature["labels"])
            feature["labels"] = feature["labels"] + [label_pad_token_id] * padding_length
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")


def make_qwen_preprocess_fn(tokenizer, apis, train_type: str, max_len: int):
    return make_generic_chat_preprocess_fn(
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback="error",
    )


def make_phi_preprocess_fn(tokenizer, apis, train_type: str, max_len: int):
    return make_generic_chat_preprocess_fn(
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback="error",
    )


def make_llama_history_preprocess_fn(tokenizer, apis, prompt_template: str, model_name: str, max_len: int):
    return make_generic_chat_preprocess_fn(
        tokenizer,
        apis,
        "history",
        max_len,
        chat_template_fallback="simple",
    )


def make_llama_rewrite_preprocess_fn(tokenizer, apis, prompt_template: str, model_name: str, max_len: int):
    return make_generic_chat_preprocess_fn(
        tokenizer,
        apis,
        "rewrite",
        max_len,
        chat_template_fallback="simple",
    )


def resolve_lora_target_modules(args, default_target_modules):
    if not args.lora_target_modules:
        return default_target_modules

    value = args.lora_target_modules.strip()
    if value == "all-linear":
        return "all-linear"

    modules = [module.strip() for module in value.split(",") if module.strip()]
    if not modules:
        raise ValueError("--lora_target_modules must be 'all-linear' or a non-empty comma-separated list.")
    return modules


def build_lora_config(args, default_r, default_alpha, default_dropout, default_target_modules):
    r = args.lora_r if args.lora_r is not None else default_r
    alpha = args.lora_alpha if args.lora_alpha is not None else default_alpha
    dropout = args.lora_dropout if args.lora_dropout is not None else default_dropout
    target_modules = resolve_lora_target_modules(args, default_target_modules)

    if r <= 0:
        raise ValueError("--lora_r must be positive.")
    if alpha <= 0:
        raise ValueError("--lora_alpha must be positive.")
    if not 0 <= dropout < 1:
        raise ValueError("--lora_dropout must be in [0, 1).")

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )


def print_lora_config(lora_config):
    print(
        "LoRA config: "
        f"r={lora_config.r}, "
        f"alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, "
        f"target_modules={lora_config.target_modules}"
    )


def build_qwen_lora_config(args):
    return build_lora_config(
        args,
        default_r=16,
        default_alpha=32,
        default_dropout=0.05,
        default_target_modules="all-linear",
    )


def build_phi_lora_config(args):
    return build_lora_config(
        args,
        default_r=16,
        default_alpha=32,
        default_dropout=0.05,
        default_target_modules="all-linear",
    )


def build_llama_lora_config(args):
    return build_lora_config(
        args,
        default_r=8,
        default_alpha=16,
        default_dropout=0.1,
        default_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
        ],
    )


def build_generic_lora_config(args):
    return build_lora_config(
        args,
        default_r=16,
        default_alpha=32,
        default_dropout=0.05,
        default_target_modules="all-linear",
    )


def build_training_args(output_dir: str, args, include_fp16: bool):
    kwargs = {
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "logging_dir": f"{output_dir}/logs",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
    }
    if include_fp16:
        kwargs["fp16"] = False
    return TrainingArguments(**kwargs)


def run_qwen_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    output_root, model_root_dir, output_dir, cache_dir = resolve_storage_paths(
        args,
        model_name,
        f"{model_name.split('/')[-1]}-{train_type}-{prefix}",
    )
    apply_postprocess_defaults(args, output_root, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    if args.postprocess_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Training output directory not found for --postprocess_only: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    print(f"model_name: {model_name}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")

    if maybe_run_postprocess_only(
        args,
        output_dir,
        output_root,
        cache_dir,
        model_name,
        preexisting_checkpoint_names,
    ):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1538

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    apis = {}
    train_files = collect_legacy_train_files(args.train_dir)
    print(train_files)
    raw_ds = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

    preprocess_example = make_qwen_preprocess_fn(tokenizer, apis, train_type, max_len)
    processed_train = raw_ds["train"].map(
        preprocess_example,
        desc="Applying Qwen chat template",
    )
    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    lora_config = build_qwen_lora_config(args)
    print_lora_config(lora_config)
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M ({100 * trainable / total:.2f}%)")

    training_args = build_training_args(str(output_dir), args, include_fp16=False)
    callbacks = build_epoch_eval_callbacks(
        args,
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback="error",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


def run_phi_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    output_root, model_root_dir, output_dir, cache_dir = resolve_storage_paths(
        args,
        model_name,
        f"{model_name.split('/')[-1]}-{train_type}-{prefix}",
    )
    apply_postprocess_defaults(args, output_root, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    if args.postprocess_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Training output directory not found for --postprocess_only: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    print(f"model_name: {model_name}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")

    if maybe_run_postprocess_only(
        args,
        output_dir,
        output_root,
        cache_dir,
        model_name,
        preexisting_checkpoint_names,
    ):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    train_files = collect_legacy_train_files(args.train_dir)
    raw_ds = load_dataset(
        "csv",
        data_files={"train": train_files},
        delimiter="\t",
    )
    apis = {}

    preprocess_example = make_phi_preprocess_fn(tokenizer, apis, train_type, max_len)
    processed_train = raw_ds["train"].map(
        preprocess_example,
        desc="Applying Phi-4 chat template",
    )

    print(train_files)
    print()
    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    lora_config = build_phi_lora_config(args)
    print_lora_config(lora_config)
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M ({100 * trainable / total:.2f}%)")

    training_args = build_training_args(str(output_dir), args, include_fp16=False)
    callbacks = build_epoch_eval_callbacks(
        args,
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback="error",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


def run_llama_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    output_root, model_root_dir, output_dir, cache_dir = resolve_storage_paths(
        args,
        model_name,
        f"{model_name.split('/')[0]}-{train_type}-{prefix}",
    )
    apply_postprocess_defaults(args, output_root, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    if args.postprocess_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Training output directory not found for --postprocess_only: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    print(f"model_name: {model_name}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")

    if maybe_run_postprocess_only(
        args,
        output_dir,
        output_root,
        cache_dir,
        model_name,
        preexisting_checkpoint_names,
    ):
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(cache_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    apis = {}
    train_files = collect_legacy_train_files(args.train_dir)
    print(train_files)
    raw_datasets = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

    print(train_type, output_dir)

    if model_name not in {
        "meta-llama/Llama-3.2-3B-Instruct",
        "google/gemma-3-4b-it",
    }:
        raise ValueError(
            "llama profile supports only meta-llama/Llama-3.2-3B-Instruct "
            "and google/gemma-3-4b-it."
        )

    preprocess_fn = make_generic_chat_preprocess_fn(
        tokenizer=tokenizer,
        apis=apis,
        train_type=train_type,
        max_len=max_len,
        chat_template_fallback=args.chat_template_fallback,
    )
    processed_train = raw_datasets["train"].map(
        preprocess_fn,
        desc="Applying baseline chat template for llama profile",
    )

    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    lora_config = build_llama_lora_config(args)
    print_lora_config(lora_config)
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        "학습 가능 파라미터: "
        f"{trainable}, 전체 파라미터: {total}, "
        f"학습 비율: {100 * trainable / total:.2f}%"
    )

    training_args = build_training_args(str(output_dir), args, include_fp16=True)
    callbacks = build_epoch_eval_callbacks(
        args,
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback=args.chat_template_fallback,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


def run_generic_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    if not model_name:
        raise ValueError("--model_name is required when --profile generic is used.")

    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    default_output_name = f"{model_name.split('/')[-1]}-{train_type}-{prefix}"
    output_root, model_root_dir, output_dir, cache_dir = resolve_storage_paths(
        args,
        model_name,
        default_output_name,
    )
    apply_postprocess_defaults(args, output_root, model_name, args.profile, train_type, prefix)
    cache_dir.mkdir(parents=True, exist_ok=True)

    trust_remote_code = resolve_trust_remote_code_arg(args, default=True)
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )
    maybe_auto_skip_unsupported_postprocess(args, config)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    if args.postprocess_only:
        if not output_dir.exists():
            raise FileNotFoundError(
                f"Training output directory not found for --postprocess_only: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    print(f"profile: {args.profile}")
    print(f"model_name: {model_name}")
    print(f"model_architecture: {get_config_architecture(config)}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")
    print(f"chat_template_fallback: {args.chat_template_fallback}")

    if maybe_run_postprocess_only(
        args,
        output_dir,
        output_root,
        cache_dir,
        model_name,
        preexisting_checkpoint_names,
    ):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        trust_remote_code=trust_remote_code,
    )
    added_tokens = configure_tokenizer_padding(tokenizer, max_len)

    apis = {}
    train_files = collect_legacy_train_files(args.train_dir)
    print(train_files)
    raw_ds = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

    preprocess_example = make_generic_chat_preprocess_fn(
        tokenizer=tokenizer,
        apis=apis,
        train_type=train_type,
        max_len=max_len,
        chat_template_fallback=args.chat_template_fallback,
    )

    if args.dry_run_preprocess:
        sample = preprocess_example(raw_ds["train"][0])
        target_tokens = sum(label != -100 for label in sample["labels"])
        if target_tokens <= 0:
            raise ValueError(
                "Dry-run produced no trainable target tokens. Increase --max_len or inspect the prompt."
            )
        print(sample["strprompt"])
        print(f"dry_run_input_tokens: {len(sample['input_ids'])}")
        print(f"dry_run_target_tokens: {target_tokens}")
        print("[dry_run] skipped model loading/training")
        return

    processed_train = raw_ds["train"].map(
        preprocess_example,
        desc=f"Applying chat template for {args.profile}",
    )
    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
    )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    lora_config = build_generic_lora_config(args)
    print_lora_config(lora_config)
    model = get_peft_model(model, lora_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M ({100 * trainable / total:.2f}%)")

    training_args = build_training_args(str(output_dir), args, include_fp16=False)
    callbacks = build_epoch_eval_callbacks(
        args,
        tokenizer,
        apis,
        train_type,
        max_len,
        chat_template_fallback=args.chat_template_fallback,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
        args=args,
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


def main():
    args = parse_args()
    if args.profile == "qwen":
        run_qwen_profile(args)
    elif args.profile == "phi":
        run_phi_profile(args)
    elif args.profile in GENERIC_PROFILE_NAMES:
        run_generic_profile(args)
    else:
        run_llama_profile(args)


if __name__ == "__main__":
    main()
