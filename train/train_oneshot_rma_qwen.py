import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
UTILS_DIR = PROJECT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from oneshot_qwen_prompt import (
    build_api_str_from_candidates,
    build_oneshot_messages,
    render_chat_template,
)
from train_sentence_rewriter import (
    DEFAULT_LLAMA_CPP_DIR,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    DEFAULT_OUTPUT_ROOT,
    build_lora_config,
    infer_model_slug,
    postprocess_trained_model,
    resolve_model_storage_paths,
)

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
PHI_MODEL_NAME = "microsoft/Phi-4-mini-instruct"
LLAMA_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"
DEFAULT_ADDITIONAL_DIR = DEFAULT_TRAIN_DIR / "additional"
PLANNING_EXTRA_FILE = "it2_NR_train.tsv"
DEFAULT_OUTPUT_TAG = "oneshot-rma"
ONESHOT_SYSTEM_PROMPT = (
    "Given a conversation history, a user query, and a list of available tools, "
    "first rewrite the query by resolving ambiguous references using the "
    "conversation history. Then select the most appropriate tool and generate "
    "its arguments. Return compact JSON only with keys "
    "\"rewrited_query\", \"plan\", and \"arguments\". Always include all three "
    "keys. The value of \"arguments\" must always be an object. If no tool "
    "matches the request, set \"plan\" to \"None\" and \"arguments\" to {}."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a one-shot RMA LoRA adapter for Qwen, Phi-4, or Llama-3."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Base model name to load from Hugging Face.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for HF cache, checkpoints, and final adapter.",
    )
    parser.add_argument(
        "--output_tag",
        default=DEFAULT_OUTPUT_TAG,
        help="Suffix used to distinguish this run under output_root.",
    )
    parser.add_argument(
        "--train_dir",
        default=str(DEFAULT_TRAIN_DIR),
        help="Directory containing planning TSV files.",
    )
    parser.add_argument(
        "--additional_dir",
        default=str(DEFAULT_ADDITIONAL_DIR),
        help="Directory containing additional planning TSV files.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1536,
        help="Token truncation length.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging interval in steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=8,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--skip_postprocess",
        action="store_true",
        help="Skip merge, GGUF conversion, and Ollama registration after training.",
    )
    parser.add_argument(
        "--postprocess_only",
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
        help="Optional explicit local directory containing the base model snapshot.",
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


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def is_qwen_model(model_name: str) -> bool:
    return "Qwen/" in model_name


def is_phi_model(model_name: str) -> bool:
    return model_name == PHI_MODEL_NAME


def is_llama_model(model_name: str) -> bool:
    return model_name == LLAMA_MODEL_NAME


def ensure_supported_model(model_name: str):
    if not (is_qwen_model(model_name) or is_phi_model(model_name) or is_llama_model(model_name)):
        raise ValueError(
            "train_oneshot_rma_qwen.py supports only Qwen-family models, "
            "microsoft/Phi-4-mini-instruct, and meta-llama/Llama-3.2-3B-Instruct."
        )


def read_simple_apis(api_path: Path) -> Dict:
    with open(api_path, "r", encoding="utf-8") as file:
        return json.load(file)


def collect_training_files(train_dir: Path, additional_dir: Path) -> List[Path]:
    planning_files = sorted(
        path for path in train_dir.glob("*.tsv") if "nonNR" in path.name
    )
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


def build_llama_prompts(api_str: str, conversation_history: str, query: str, target_json: str):
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


def build_preprocess_fn(tokenizer, model_name: str, apis: Dict, max_length: int):
    def preprocess(example):
        candidates = parse_literal(
            example["candidates"],
            field_name="candidates",
            source_file=example.get("source_file"),
        )
        api_str = build_api_str_from_candidates(candidates, apis)
        target_json = build_oneshot_target(example)
        if is_llama_model(model_name):
            prompt_prefix, prompt = build_llama_prompts(
                api_str=api_str,
                conversation_history=example["conversation_history"],
                query=example["query"],
                target_json=target_json,
            )
        else:
            messages = build_oneshot_messages(
                conversation_history=example["conversation_history"],
                query=example["query"],
                candidates=candidates,
                apis=apis,
            )

            prompt_prefix = render_chat_template(
                tokenizer=tokenizer,
                messages=messages,
                add_generation_prompt=True,
                model_name=model_name,
            )
            prompt = render_chat_template(
                tokenizer=tokenizer,
                messages=messages + [{"role": "assistant", "content": target_json}],
                add_generation_prompt=False,
                model_name=model_name,
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


def max_total_length(dataset: Dataset) -> int:
    max_len = 0
    for example in dataset:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_len:
            max_len = total_length
    return max_len


def apply_postprocess_defaults(args, model_name: str, output_root: Path):
    model_slug = infer_model_slug(model_name)
    artifact_root = output_root.parent
    artifact_stem = f"{model_slug}-rma-oneshot"

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
        args.ollama_model_name = f"{model_slug}-oneshot"


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


def main():
    args = parse_args()
    ensure_supported_model(args.model_name)

    train_dir = resolve_path(args.train_dir)
    additional_dir = resolve_path(args.additional_dir)
    tools_path = resolve_path(args.tools_path)
    output_root = Path(args.output_root).expanduser()
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
        postprocess_trained_model(
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    files = collect_training_files(train_dir, additional_dir)
    print("training_files:", [str(file) for file in files])
    raw_dataset = dataframe_to_dataset(files)
    apis = read_simple_apis(tools_path)

    processed_train = raw_dataset.map(
        build_preprocess_fn(
            tokenizer=tokenizer,
            model_name=args.model_name,
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
    print(f"max total length: {max_total_length(processed_train)}")

    model = get_peft_model(model, build_lora_config(args.model_name))
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Trainable Parameters: "
        f"{trainable_params}, Total Parameters: {total_params}, "
        f"Training Rate: {100 * trainable_params / total_params:.2f}%"
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        logging_dir=str(output_dir / "logs"),
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        data_collator=DataCollatorForCausalLM(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    trainer.train()
    final_global_step = trainer.state.global_step
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if args.skip_postprocess:
        print("[postprocess] skipped by --skip_postprocess")
        return

    postprocess_trained_model(
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
