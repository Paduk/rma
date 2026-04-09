#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import get_peft_model
from transformers import AutoTokenizer

from postprocess_gemma4 import (
    DEFAULT_GEMMA4_LLAMA_CPP_DIR,
    POSTPROCESS_STAGE_CHOICES,
    postprocess_gemma4_trained_model,
)
from train_gemma4 import (
    apply_chat_template,
    build_lora_config,
    build_trainer,
    build_training_args,
    ensure_gemma4_support,
    freeze_non_text_parameters,
    load_model,
    print_max_total_length,
)
from train_sentence_rewriter import (
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    DEFAULT_OUTPUT_ROOT,
    collect_training_files as collect_rewrite_training_files,
    infer_model_slug,
    resolve_model_storage_paths,
)


DEFAULT_MODEL_NAME = "google/gemma-4-E2B-it"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
DEFAULT_ADDITIONAL_DIR = DEFAULT_TRAIN_DIR / "additional"
PLANNING_EXTRA_FILE = "it2_NR_train.tsv"
DEFAULT_OUTPUT_TAG = "multitask-rma-gemma4"

REWRITE_SYSTEM_PROMPT = (
    "Rewrite the query clearly by replacing ambiguous pronouns (like \"it\", "
    "\"that\") with explicit information from the conversation history. Keep "
    "exactly the same sentence structure. Do NOT generate or include any "
    "information, words, or values outside of the provided conversation_history "
    "and query."
)

PLANNING_SYSTEM_PROMPT = (
    "Given a user query and a list of available tools, select the most "
    "appropriate tool and generate the corresponding parameters. If no tool "
    "matches the query, set the tool to 'None'. Only use parameter values that "
    "are explicitly stated or can be reasonably inferred from the query.\n"
    "<|tool|>{tools}<|/tool|>"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Gemma 4 multitask LoRA adapter for rewrite + planning."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--additional_dir", default=str(DEFAULT_ADDITIONAL_DIR))
    parser.add_argument("--tools_path", default=str(DEFAULT_TOOLS_PATH))
    parser.add_argument(
        "--mix_strategy",
        choices=["balanced", "concat"],
        default="balanced",
    )
    parser.add_argument("--rewrite_sampling_prob", type=float, default=0.5)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--num_train_epochs", type=float, default=DEFAULT_NUM_TRAIN_EPOCHS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--skip_postprocess", action="store_true")
    parser.add_argument("--postprocess_only", action="store_true")
    parser.add_argument("--postprocess_stage", choices=POSTPROCESS_STAGE_CHOICES, default="all")
    parser.add_argument("--export_checkpoint_epoch", type=int, default=None)
    parser.add_argument(
        "--export_source",
        choices=["final_checkpoint", "final_adapter"],
        default="final_adapter",
    )
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("--merged_dir", default=None)
    parser.add_argument("--gguf_path", default=None)
    parser.add_argument("--modelfile", default=None)
    parser.add_argument("--ollama_model_name", default=None)
    parser.add_argument("--ollama_host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--ollama_models_dir", default=DEFAULT_OLLAMA_MODELS_DIR)
    parser.add_argument("--ollama_bin", default=DEFAULT_OLLAMA_BIN)
    parser.add_argument("--ollama_lib_path", default=DEFAULT_OLLAMA_LIB_PATH)
    parser.add_argument("--llama_cpp_dir", default=DEFAULT_GEMMA4_LLAMA_CPP_DIR)
    parser.add_argument("--merge_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--gguf_outtype", default="bf16")
    parser.add_argument(
        "--trust_remote_code",
        choices=["auto", "true", "false"],
        default="auto",
    )
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def read_simple_apis(api_path: Path) -> Dict:
    with open(api_path, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_planning_training_files(train_dir: Path) -> List[Path]:
    files = sorted(path for path in train_dir.glob("*.tsv") if "nonNR" in path.name)
    extra_file = train_dir / PLANNING_EXTRA_FILE
    if extra_file.exists():
        files.append(extra_file)
    return files


def dataframe_to_dataset(files: List[Path], task_name: str) -> Dataset:
    if not files:
        raise ValueError(f"No TSV files found for task={task_name}.")

    dataframes = []
    for file in files:
        dataframe = pd.read_csv(file, sep="\t", dtype=str)
        dataframe["task_name"] = task_name
        dataframe["source_file"] = file.name
        dataframes.append(dataframe)

    return Dataset.from_pandas(
        pd.concat(dataframes, ignore_index=True, sort=False),
        preserve_index=False,
    )


def build_api_str(example, apis: Dict) -> str:
    api_str = ""
    candidates = ast.literal_eval(example["candidates"])
    for plan in candidates:
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
    return api_str


def tokenize_with_labels(tokenizer, prompt: str, prompt_prefix: str, max_length: int):
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]
    label_start = len(tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"])
    label_start = min(label_start, len(input_ids))
    labels = [-100] * label_start + input_ids[label_start:]
    return input_ids, labels


def build_rewrite_preprocess_fn(tokenizer, max_length: int):
    def preprocess(example):
        data = {
            "conversation_history": example["conversation_history"],
            "query": example["query"],
        }
        user_content = json.dumps(data, ensure_ascii=False, indent=2)
        assistant_content = json.dumps(
            {"rewrited_query": example["rewrited_query"]},
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt_prefix = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        prompt = apply_chat_template(
            tokenizer,
            messages + [{"role": "assistant", "content": assistant_content}],
            add_generation_prompt=False,
        )

        input_ids, labels = tokenize_with_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            max_length=max_length,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": assistant_content,
        }

    return preprocess


def build_planning_preprocess_fn(tokenizer, apis: Dict, max_length: int):
    def preprocess(example):
        api_str = build_api_str(example, apis)
        system_msg = PLANNING_SYSTEM_PROMPT.format(tools=api_str)
        user_content = f"User Query: {example['rewrited_query']}"
        assistant_content = (
            json.dumps(example["answer"], ensure_ascii=False)
            if isinstance(example["answer"], dict)
            else example["answer"]
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        prompt_prefix = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        prompt = apply_chat_template(
            tokenizer,
            messages + [{"role": "assistant", "content": assistant_content}],
            add_generation_prompt=False,
        )

        input_ids, labels = tokenize_with_labels(
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_prefix=prompt_prefix,
            max_length=max_length,
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "strprompt": prompt,
            "stranswer": assistant_content,
        }

    return preprocess


def mix_processed_datasets(rewrite_ds: Dataset, planning_ds: Dataset, args) -> Dataset:
    if args.mix_strategy == "concat":
        return concatenate_datasets([rewrite_ds, planning_ds]).shuffle(seed=args.shuffle_seed)

    rewrite_prob = args.rewrite_sampling_prob
    if not 0.0 < rewrite_prob < 1.0:
        raise ValueError("--rewrite_sampling_prob must be between 0 and 1.")

    return interleave_datasets(
        [rewrite_ds, planning_ds],
        probabilities=[rewrite_prob, 1.0 - rewrite_prob],
        seed=args.shuffle_seed,
        stopping_strategy="all_exhausted",
    )


def apply_postprocess_defaults(args, model_name: str, output_root: Path):
    model_slug = infer_model_slug(model_name)
    artifact_root = output_root.parent
    artifact_stem = f"{model_slug}-{args.output_tag}"

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


def main():
    args = parse_args()
    if "gemma-4" not in args.model_name.lower():
        raise ValueError("train_multitask_rma_gemma4.py supports Gemma 4 checkpoints only.")
    ensure_gemma4_support()

    train_dir = resolve_path(args.train_dir)
    additional_dir = resolve_path(args.additional_dir)
    tools_path = resolve_path(args.tools_path)
    output_root = Path(args.output_root).expanduser().resolve()
    model_root_dir, output_dir, cache_dir = resolve_model_storage_paths(
        output_root,
        args.model_name,
        args.output_tag,
    )
    apply_postprocess_defaults(args, args.model_name, output_root)

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

    print(f"model_name: {args.model_name}")
    print(f"train_dir: {train_dir}")
    print(f"additional_dir: {additional_dir}")
    print(f"tools_path: {tools_path}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"training_output_dir: {output_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"mix_strategy: {args.mix_strategy}")

    if args.postprocess_only:
        postprocess_gemma4_trained_model(
            args=args,
            model_name=args.model_name,
            output_dir=output_dir,
            output_root=output_root,
            preexisting_checkpoint_names=preexisting_checkpoint_names,
            num_train_epochs=args.num_train_epochs,
            final_global_step=None,
            cache_dir=cache_dir,
        )
        return

    tokenizer_kwargs = {
        "cache_dir": str(cache_dir),
        "force_download": args.force_download,
    }
    if args.hf_token:
        tokenizer_kwargs["token"] = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_length

    model = load_model(
        args.model_name,
        cache_dir,
        args.hf_token,
        args.force_download,
        args.attn_implementation,
    )
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    freeze_non_text_parameters(model)

    rewrite_files = collect_rewrite_training_files(train_dir, additional_dir)
    planning_files = collect_planning_training_files(train_dir)
    print("rewrite_files:", [str(path) for path in rewrite_files])
    print("planning_files:", [str(path) for path in planning_files])

    rewrite_raw = dataframe_to_dataset(rewrite_files, "rewrite")
    planning_raw = dataframe_to_dataset(planning_files, "planning")
    apis = read_simple_apis(tools_path)

    rewrite_processed = rewrite_raw.map(
        build_rewrite_preprocess_fn(tokenizer, args.max_length),
        remove_columns=rewrite_raw.column_names,
        desc="Preprocessing rewrite task",
    )
    planning_processed = planning_raw.map(
        build_planning_preprocess_fn(tokenizer, apis, args.max_length),
        remove_columns=planning_raw.column_names,
        desc="Preprocessing planning task",
    )

    print(f"rewrite_examples: {len(rewrite_processed)}")
    print(f"planning_examples: {len(planning_processed)}")
    print("rewrite sample prompt:")
    print(rewrite_processed[0]["strprompt"])
    print()
    print("planning sample prompt:")
    print(planning_processed[0]["strprompt"])
    print()
    print("rewrite max total length:")
    print_max_total_length(rewrite_processed)
    print("planning max total length:")
    print_max_total_length(planning_processed)

    processed_train = mix_processed_datasets(rewrite_processed, planning_processed, args)
    print(f"mixed_examples: {len(processed_train)}")
    print("mixed max total length:")
    print_max_total_length(processed_train)

    model = get_peft_model(model, build_lora_config(model, args))
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Trainable Parameters: "
        f"{trainable_params}, Total Parameters: {total_params}, "
        f"Training Rate: {100 * trainable_params / total_params:.2f}%"
    )

    training_args = build_training_args(str(output_dir), args)
    trainer = build_trainer(model, training_args, processed_train, tokenizer)

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_gemma4_trained_model(
        args=args,
        model_name=args.model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
