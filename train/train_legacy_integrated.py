# -*- coding: utf-8 -*-
"""
Unified entrypoint that preserves the legacy behavior of:
  - train_qwen.py
  - train_phi.py
  - train_llama.py

Examples:
  python train_legacy_integrated.py --profile qwen
  python train_legacy_integrated.py --profile phi
  python train_legacy_integrated.py --profile llama
"""

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from gemma_prompts import SFT_HISTORY_TRAIN_GEMMA, SFT_REWRITE_TRAIN_GEMMA
from llama_prompts import SFT_HISTORY_TRAIN_LLAMA, SFT_REWRITE_TRAIN_LLAMA
from train_sentence_rewriter import (
    DEFAULT_LLAMA_CPP_DIR,
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    infer_model_slug,
    postprocess_trained_model,
)


QWEN_DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
PHI_DEFAULT_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
LLAMA_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LEGACY_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
LEGACY_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

LEGACY_SYSTEM_HISTORY_PROMPT = (
    "You are a helpful assistant capable of selecting appropriate tools based on "
    "user queries and generating corresponding parameters. Use information from "
    "the conversation history when relevant. Only use parameter values that are "
    "explicitly stated or can be reasonably inferred from the query. If no tool "
    "matches the query, set the tool to 'None'.\n <|tool|>{tools}<|/tool|>"
)

LEGACY_SYSTEM_REWRITE_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query.\n "
    "<|tool|>{tools}<|/tool|>"
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
        prefix="all_linear",
        max_len=1536,
    ),
    "phi": ProfileDefaults(
        model_name=PHI_DEFAULT_MODEL_NAME,
        train_type="history",
        prefix="1st",
        max_len=1024,
    ),
    "llama": ProfileDefaults(
        model_name=LLAMA_DEFAULT_MODEL_NAME,
        train_type="history",
        prefix="all_linear",
        max_len=1536,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Integrated legacy trainer for qwen/phi/llama planner scripts."
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=["qwen", "phi", "llama"],
        help="Which legacy training script behavior to run.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Override the profile's legacy default model_name.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(LEGACY_TOOLS_PATH),
        help="Path to simple_api.json.",
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
        help="Explicit output_dir. If omitted, the legacy naming rule is used.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5,
        type=float,
        help="Training epochs. Legacy default is 4.",
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


def resolve_output_dir(output_dir: str | None, default_name: str) -> str:
    target = default_name if output_dir is None else output_dir
    return str(resolve_path(target))


def apply_postprocess_defaults(args, output_dir: Path, model_name: str, profile: str, train_type: str, prefix: str):
    model_slug = infer_model_slug(model_name)
    artifact_root = output_dir.parent
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
        output_root=output_dir.parent,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=args.num_train_epochs,
        final_global_step=None,
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


def build_api_str(example, apis):
    api_str = ""
    candidates = ast.literal_eval(example["candidates"])
    for plan in candidates:
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
    return api_str


def print_max_total_length(processed_train):
    max_total_length = 0
    for example in processed_train:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_total_length:
            max_total_length = total_length
    print(f"input_ids + labels 최대 길이: {max_total_length}")


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
    def preprocess_example(example):
        api_str = build_api_str(example, apis)

        if train_type == "history":
            system_msg = LEGACY_SYSTEM_HISTORY_PROMPT.format(tools=api_str)
            user_content = (
                f"Conversation History: {example['conversation_history']}\n"
                f"User Query: {example['query']}"
            )
        else:
            system_msg = LEGACY_SYSTEM_REWRITE_PROMPT.format(tools=api_str)
            user_content = f"User Query: {example['rewrited_query']}"

        assistant_content = (
            json.dumps(example["answer"], ensure_ascii=False)
            if isinstance(example["answer"], dict)
            else example["answer"]
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        prompt_with_answer = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        tokenized = tokenizer(
            prompt_with_answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        input_ids = tokenized["input_ids"]

        assistant_start = prompt_with_answer.rfind("<|assistant|>") + len("<|assistant|>")
        label_start = len(
            tokenizer(
                prompt_with_answer[:assistant_start],
                add_special_tokens=False,
            )["input_ids"]
        )
        labels = [-100] * label_start + input_ids[label_start:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt_with_answer,
            "stranswer": assistant_content,
        }

    return preprocess_example


def make_phi_preprocess_fn(tokenizer, apis, train_type: str, max_len: int):
    def preprocess_example(example):
        api_str = build_api_str(example, apis)

        if train_type == "history":
            system_msg = LEGACY_SYSTEM_HISTORY_PROMPT.format(tools=api_str)
            user_content = (
                f"Conversation History: {example['conversation_history']}\n"
                f"User Query: {example['query']}"
            )
        else:
            system_msg = LEGACY_SYSTEM_REWRITE_PROMPT.format(tools=api_str)
            user_content = f"User Query: {example['rewrited_query']}"

        assistant_content = (
            json.dumps(example["answer"], ensure_ascii=False)
            if isinstance(example["answer"], dict)
            else example["answer"]
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        prompt_with_answer = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenized = tokenizer(
            prompt_with_answer,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        input_ids = tokenized["input_ids"]

        assistant_start = prompt_with_answer.rfind("<|assistant|>") + len("<|assistant|>")
        label_start = len(
            tokenizer(
                prompt_with_answer[:assistant_start],
                add_special_tokens=False,
            )["input_ids"]
        )
        labels = [-100] * label_start + input_ids[label_start:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt_with_answer,
            "stranswer": assistant_content,
        }

    return preprocess_example


def make_llama_history_preprocess_fn(tokenizer, apis, prompt_template: str, model_name: str, max_len: int):
    def preprocess_example_history(example):
        api_str = build_api_str(example, apis)
        prompt = prompt_template.format(
            tools=api_str,
            conversation_history=example["conversation_history"],
            data=example["query"],
            answer=example["answer"],
        )

        tokenized = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]

        if model_name == "meta-llama/Llama-3.2-3B-Instruct":
            model_start = prompt.find("assistant<|end_header_id|>")
        elif model_name == "google/gemma-3-4b-it":
            model_start = prompt.find("<start_of_turn>model")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if model_start == -1:
            raise ValueError("Prompt does not contain 'assistant<|end_header_id|>'")

        model_token_start = len(
            tokenizer(prompt[:model_start], add_special_tokens=False)["input_ids"]
        )
        labels = [-100] * model_token_start + input_ids[model_token_start:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": (
                json.dumps(example["answer"], indent=2, ensure_ascii=False)
                if isinstance(example["answer"], dict)
                else example["answer"]
            ),
        }

    return preprocess_example_history


def make_llama_rewrite_preprocess_fn(tokenizer, apis, prompt_template: str, model_name: str, max_len: int):
    def preprocess_example_rewrite(example):
        api_str = build_api_str(example, apis)
        prompt = prompt_template.format(
            tools=api_str,
            data=example["rewrited_query"],
            answer=example["answer"],
        )

        tokenized = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=max_len,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]

        if model_name == "meta-llama/Llama-3.2-3B-Instruct":
            model_start = prompt.find("assistant<|end_header_id|>")
        elif model_name == "google/gemma-3-4b-it":
            model_start = prompt.find("<start_of_turn>model")
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        if model_start == -1:
            raise ValueError("Prompt does not contain 'assistant<|end_header_id|>'")

        model_token_start = len(
            tokenizer(prompt[:model_start], add_special_tokens=False)["input_ids"]
        )
        labels = [-100] * model_token_start + input_ids[model_token_start:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": (
                json.dumps(example["answer"], indent=2, ensure_ascii=False)
                if isinstance(example["answer"], dict)
                else example["answer"]
            ),
        }

    return preprocess_example_rewrite


def build_qwen_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )


def build_phi_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )


def build_llama_lora_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "o_proj",
        ],
        lora_dropout=0.1,
        bias="none",
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
    output_dir = Path(
        resolve_output_dir(
            args.output_dir,
            f"{model_name.split('/')[-1]}-{train_type}-{prefix}",
        )
    )
    apply_postprocess_defaults(args, output_dir, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    if maybe_run_postprocess_only(args, output_dir, model_name, preexisting_checkpoint_names):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    apis = read_simple_apis(args.tools_path)
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

    model = get_peft_model(model, build_qwen_lora_config())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M")

    training_args = build_training_args(str(output_dir), args, include_fp16=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
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
        output_root=output_dir.parent,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
    )


def run_phi_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    output_dir = Path(
        resolve_output_dir(
            args.output_dir,
            f"{model_name.split('/')[-1]}-{train_type}-{prefix}",
        )
    )
    apply_postprocess_defaults(args, output_dir, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    if maybe_run_postprocess_only(args, output_dir, model_name, preexisting_checkpoint_names):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = 1024

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    train_files = collect_legacy_train_files(args.train_dir)
    raw_ds = load_dataset(
        "csv",
        data_files={"train": train_files},
        delimiter="\t",
    )
    apis = read_simple_apis(args.tools_path)

    preprocess_example = make_phi_preprocess_fn(tokenizer, apis, train_type, max_len)
    processed_train = raw_ds["train"].map(
        preprocess_example,
        desc="Applying Phi-4 chat template",
    )

    print(train_files)
    print()
    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    model = get_peft_model(model, build_phi_lora_config())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M")

    training_args = build_training_args(str(output_dir), args, include_fp16=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer),
        tokenizer=tokenizer,
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
        output_root=output_dir.parent,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
    )


def run_llama_profile(args):
    model_name = resolve_profile_value(args, "model_name")
    train_type = resolve_profile_value(args, "train_type")
    prefix = resolve_profile_value(args, "prefix")
    max_len = resolve_profile_value(args, "max_len")
    output_dir = Path(
        resolve_output_dir(
            args.output_dir,
            f"{model_name.split('/')[0]}-{train_type}-{prefix}",
        )
    )
    apply_postprocess_defaults(args, output_dir, model_name, args.profile, train_type, prefix)

    if args.postprocess_only and args.skip_postprocess:
        raise ValueError("--postprocess_only and --skip_postprocess cannot be used together.")

    preexisting_checkpoint_names = {
        path.name for path in output_dir.glob("checkpoint-*") if path.is_dir()
    }
    if maybe_run_postprocess_only(args, output_dir, model_name, preexisting_checkpoint_names):
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    apis = read_simple_apis(args.tools_path)
    train_files = collect_legacy_train_files(args.train_dir)
    print(train_files)
    raw_datasets = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

    print(train_type, output_dir)

    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        if train_type == "history":
            prompt_template = SFT_HISTORY_TRAIN_LLAMA
            preprocess_fn = make_llama_history_preprocess_fn(
                tokenizer,
                apis,
                prompt_template,
                model_name,
                max_len,
            )
        else:
            prompt_template = SFT_REWRITE_TRAIN_LLAMA
            preprocess_fn = make_llama_rewrite_preprocess_fn(
                tokenizer,
                apis,
                prompt_template,
                model_name,
                max_len,
            )
    elif model_name == "google/gemma-3-4b-it":
        if train_type == "history":
            prompt_template = SFT_HISTORY_TRAIN_GEMMA
            preprocess_fn = make_llama_history_preprocess_fn(
                tokenizer,
                apis,
                prompt_template,
                model_name,
                max_len,
            )
        else:
            prompt_template = SFT_REWRITE_TRAIN_GEMMA
            preprocess_fn = make_llama_rewrite_preprocess_fn(
                tokenizer,
                apis,
                prompt_template,
                model_name,
                max_len,
            )
    else:
        raise ValueError(
            "llama profile supports only meta-llama/Llama-3.2-3B-Instruct "
            "and google/gemma-3-4b-it."
        )

    processed_train = raw_datasets["train"].map(preprocess_fn)

    print(processed_train[0]["strprompt"])
    print_max_total_length(processed_train)

    model = get_peft_model(model, build_llama_lora_config())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        "학습 가능 파라미터: "
        f"{trainable}, 전체 파라미터: {total}, "
        f"학습 비율: {100 * trainable / total:.2f}%"
    )

    training_args = build_training_args(str(output_dir), args, include_fp16=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        tokenizer=tokenizer,
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
        output_root=output_dir.parent,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
    )


def main():
    args = parse_args()
    if args.profile == "qwen":
        run_qwen_profile(args)
    elif args.profile == "phi":
        run_phi_profile(args)
    else:
        run_llama_profile(args)


if __name__ == "__main__":
    main()
