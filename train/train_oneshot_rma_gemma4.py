#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from datasets import Dataset
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
DEFAULT_OUTPUT_TAG = "oneshot-rma-gemma4"
ONESHOT_SYSTEM_PROMPT = (
    "Given a conversation history, a user query, and a list of available tools, "
    "first rewrite the query by resolving ambiguous references using the "
    "conversation history. Then select the most appropriate tool and generate "
    "its arguments. Return compact JSON only with keys "
    "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
    "keys. The value of \"arguments\" must always be an object. If no tool "
    "matches the request, set \"plan\" to \"None\" and \"arguments\" to {{}}.\n"
    "<|tool|>{tools}<|/tool|>"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Gemma 4 one-shot RMA LoRA adapter."
    )
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output_tag", default=DEFAULT_OUTPUT_TAG)
    parser.add_argument("--train_dir", default=str(DEFAULT_TRAIN_DIR))
    parser.add_argument("--additional_dir", default=str(DEFAULT_ADDITIONAL_DIR))
    parser.add_argument("--tools_path", default=str(DEFAULT_TOOLS_PATH))
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


def collect_training_files(train_dir: Path, additional_dir: Path) -> List[Path]:
    planning_files = sorted(path for path in train_dir.glob("*.tsv") if "nonNR" in path.name)
    extra_file = train_dir / PLANNING_EXTRA_FILE
    if extra_file.exists():
        planning_files.append(extra_file)
    additional_files = sorted(additional_dir.glob("*.tsv"))
    return [*planning_files, *additional_files]


def dataframe_to_dataset(files: List[Path]) -> Dataset:
    if not files:
        raise ValueError("No TSV files found for one-shot training.")

    dataframes = []
    for file in files:
        dataframe = pd.read_csv(file, sep="\t", dtype=str)
        dataframe["source_file"] = file.name
        dataframes.append(dataframe)

    return Dataset.from_pandas(
        pd.concat(dataframes, ignore_index=True, sort=False),
        preserve_index=False,
    )


def parse_literal(raw_value, field_name: str, source_file: str | None = None):
    if isinstance(raw_value, (dict, list)):
        return raw_value
    if not isinstance(raw_value, str):
        raise ValueError(f"Unsupported type for {field_name}: {type(raw_value).__name__}")
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        location = f" source_file={source_file}" if source_file else ""
        raise ValueError(f"Failed to parse {field_name}.{location}") from exc


def build_api_str_from_candidates(candidates, apis) -> str:
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list.")

    lines = []
    for plan_name in candidates:
        api_data = apis[plan_name].copy()
        lines.append(f"{plan_name}: {api_data}")
    return "\n".join(lines)


def build_oneshot_target(example) -> str:
    answer = parse_literal(
        example["answer"],
        field_name="answer",
        source_file=example.get("source_file"),
    )
    if not isinstance(answer, dict):
        raise ValueError("answer must parse to a dict.")

    arguments = answer.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("answer['arguments'] must be a dict.")

    target = {
        "rewrited_query": example["rewrited_query"],
        "plan": answer.get("plan", "None"),
        "arguments": arguments,
    }
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


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


def build_preprocess_fn(tokenizer, apis: Dict, max_length: int):
    def preprocess(example):
        candidates = parse_literal(
            example["candidates"],
            field_name="candidates",
            source_file=example.get("source_file"),
        )
        api_str = build_api_str_from_candidates(candidates, apis)
        target_json = build_oneshot_target(example)
        system_msg = ONESHOT_SYSTEM_PROMPT.format(tools=api_str)
        user_content = (
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        prompt_prefix = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        prompt = apply_chat_template(
            tokenizer,
            messages + [{"role": "assistant", "content": target_json}],
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
            "stranswer": target_json,
        }

    return preprocess


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
        raise ValueError("train_oneshot_rma_gemma4.py supports Gemma 4 checkpoints only.")
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
    print(f"default_merged_dir: {args.merged_dir}")
    print(f"default_gguf_path: {args.gguf_path}")
    print(f"default_ollama_model_name: {args.ollama_model_name}")

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

    files = collect_training_files(train_dir, additional_dir)
    print("training_files:", [str(file) for file in files])
    raw_dataset = dataframe_to_dataset(files)
    apis = read_simple_apis(tools_path)

    processed_train = raw_dataset.map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            apis=apis,
            max_length=args.max_length,
        ),
        remove_columns=raw_dataset.column_names,
        desc="Preprocessing one-shot task",
    )

    print(f"training_examples: {len(processed_train)}")
    print("first_training_example_source_file:")
    print(raw_dataset[0]["source_file"])
    print()
    print("first_training_prompt:")
    print(processed_train[0]["strprompt"])
    print()
    print("first_training_target:")
    print(processed_train[0]["stranswer"])
    print()
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
