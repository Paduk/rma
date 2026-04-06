import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from download_and_merge_adapter import (
    download_base_model,
    merge_adapter,
    resolve_base_model_id,
)
from llama_prompts import SFT_RMA_TRAIN_GEMMA
from llama_prompts import SFT_RMA_TRAIN_LLAMA
from llama_prompts import SFT_RMA_TRAIN_PHI4
from llama_prompts import SFT_RMA_TRAIN_QWEN3

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct" #"Qwen/Qwen3-4B"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/hj153lee/rma_train")
DEFAULT_OUTPUT_TAG = "sentence-rewriter"
DEFAULT_NUM_TRAIN_EPOCHS = 5
TRAIN_EXCLUDE_PATTERNS = ("_NR_",)
ADDITION_INCLUDE_PATTERNS = ("complex", "various_nonNR")
EXTRA_TRAIN_FILES = ("it2_NR_train.tsv",)

DEFAULT_OLLAMA_BIN = "/home/hj153lee/.local/ollama-v0.17.7/bin/ollama"
DEFAULT_OLLAMA_LIB_PATH = (
    "/home/hj153lee/.local/ollama-v0.17.7/lib/ollama:"
    "/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v13:"
    "/home/hj153lee/.local/ollama-v0.17.7/lib/ollama/cuda_v12"
)
DEFAULT_OLLAMA_MODELS_DIR = "/home/hj153lee/.ollama-qwen3-test"
DEFAULT_OLLAMA_HOST = "127.0.0.1:11435"
DEFAULT_LLAMA_CPP_DIR = "/home/hj153lee/llama.cpp"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentence rewriter LoRA adapter.")
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Base model name to load from Hugging Face.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for HF cache, epoch checkpoints, and final adapter.",
    )
    parser.add_argument(
        "--output_tag",
        default=DEFAULT_OUTPUT_TAG,
        help="Run-specific subdirectory under output_root/model_name for checkpoints and adapters.",
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


def has_any_pattern(filename: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in filename for pattern in patterns)


def collect_training_files(train_path: Path, addition_path: Path) -> List[Path]:
    base_train_files = sorted(
        file
        for file in train_path.glob("*.tsv")
        if not has_any_pattern(file.name, TRAIN_EXCLUDE_PATTERNS)
    )
    extra_train_files = [train_path / filename for filename in EXTRA_TRAIN_FILES]
    addition_files = sorted(
        file
        for file in addition_path.glob("*.tsv")
        if has_any_pattern(file.name, ADDITION_INCLUDE_PATTERNS)
    )
    return [*base_train_files, *extra_train_files, *addition_files]


def select_prompt_template(model_name: str) -> str:
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_RMA_TRAIN_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_RMA_TRAIN_GEMMA
    if model_name == "microsoft/Phi-4-mini-instruct":
        return SFT_RMA_TRAIN_PHI4
    if "Qwen/" in model_name:
        return SFT_RMA_TRAIN_QWEN3
    raise ValueError(f"Unsupported model_name for prompt template: {model_name}")


def build_lora_config(model_name: str) -> LoraConfig:
    if "Phi-4" in model_name or "Qwen" in model_name:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
        )
    if "Llama" in model_name:
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
    if "gemma" in model_name:
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
        )
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )


def infer_model_slug(model_name: str) -> str:
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


def sanitize_model_dirname(model_name: str) -> str:
    return model_name.replace("/", "__")


def resolve_model_storage_paths(
    output_root: Path,
    model_name: str,
    output_tag: str | None = None,
) -> tuple[Path, Path, Path]:
    model_root_dir = output_root / sanitize_model_dirname(model_name)
    output_dir = model_root_dir / output_tag if output_tag else model_root_dir
    cache_dir = model_root_dir / "hf_cache"
    return model_root_dir, output_dir, cache_dir


def checkpoint_sort_key(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def normalize_ollama_host(host: str) -> str:
    if "://" not in host:
        return host

    parsed = urlparse(host)
    if not parsed.hostname:
        raise ValueError(f"Invalid ollama_host: {host}")
    if parsed.port:
        return f"{parsed.hostname}:{parsed.port}"
    return parsed.hostname


def run_command(cmd: List[str], cwd: str | None = None, env: Dict[str, str] | None = None):
    print(f"[postprocess] running: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def ensure_ollama_server(ollama_bin: str, env: Dict[str, str], ollama_host: str):
    try:
        subprocess.run(
            [ollama_bin, "list"],
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(
            f"Ollama server is not reachable at {ollama_host}. "
            f"Start `ollama serve` first. {stderr}"
        ) from exc


def resolve_epoch_label(export_checkpoint_epoch: int | None, num_train_epochs: float) -> str:
    if export_checkpoint_epoch is not None:
        return str(export_checkpoint_epoch)
    if float(num_train_epochs).is_integer():
        return str(int(num_train_epochs))
    return str(num_train_epochs).replace(".", "_")


def collect_export_checkpoints(output_dir: Path, preexisting_checkpoint_names: set[str]) -> List[Path]:
    checkpoints = sorted(
        [
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and (path / "adapter_model.safetensors").exists()
        ],
        key=checkpoint_sort_key,
    )
    new_checkpoints = [path for path in checkpoints if path.name not in preexisting_checkpoint_names]
    return new_checkpoints or checkpoints


def select_export_adapter_dir(
    output_dir: Path,
    preexisting_checkpoint_names: set[str],
    export_checkpoint_epoch: int | None,
    export_source: str,
    final_global_step: int | None,
) -> Path:
    final_adapter_weights = output_dir / "adapter_model.safetensors"
    if export_source == "final_adapter":
        if not final_adapter_weights.exists():
            raise FileNotFoundError(f"Final adapter not found: {final_adapter_weights}")
        print(f"[postprocess] selected final adapter dir: {output_dir}")
        return output_dir

    if export_checkpoint_epoch is None and final_global_step is not None:
        final_checkpoint = output_dir / f"checkpoint-{final_global_step}"
        if (final_checkpoint / "adapter_model.safetensors").exists():
            print(f"[postprocess] selected final checkpoint: {final_checkpoint}")
            return final_checkpoint

    checkpoints = collect_export_checkpoints(output_dir, preexisting_checkpoint_names)
    if not checkpoints:
        if final_adapter_weights.exists():
            print(f"[postprocess] checkpoint not found, falling back to final adapter dir: {output_dir}")
            return output_dir
        raise FileNotFoundError(f"No exportable checkpoint found under {output_dir}")

    if export_checkpoint_epoch is None:
        selected = checkpoints[-1]
    else:
        if export_checkpoint_epoch < 1 or export_checkpoint_epoch > len(checkpoints):
            raise ValueError(
                f"export_checkpoint_epoch={export_checkpoint_epoch} is out of range for "
                f"{len(checkpoints)} available checkpoint(s)"
            )
        selected = checkpoints[export_checkpoint_epoch - 1]

    print(f"[postprocess] selected checkpoint: {selected}")
    return selected


def find_cached_base_model_path(
    model_name: str,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
) -> str | None:
    if cache_dir is None:
        if output_dir is None:
            return None
        cache_root = output_dir / "hf_cache"
    else:
        cache_root = cache_dir

    if not cache_root.exists():
        return None

    repo_cache_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshot_root = repo_cache_dir / "snapshots"
    if not snapshot_root.exists():
        return None

    snapshots = sorted(
        [
            path
            for path in snapshot_root.iterdir()
            if path.is_dir() and (path / "config.json").exists()
        ]
    )
    if not snapshots:
        return None
    return str(snapshots[-1])


def resolve_base_model_path(
    model_name: str,
    adapter_dir: Path,
    args,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> str:
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else None
    if base_dir and (base_dir / "config.json").exists():
        print(f"[postprocess] using explicit base model dir: {base_dir}")
        return str(base_dir)

    cached_base_model_path = find_cached_base_model_path(
        model_name=model_name,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    if cached_base_model_path:
        print(f"[postprocess] using cached base model dir: {cached_base_model_path}")
        return cached_base_model_path

    base_model_id = resolve_base_model_id(adapter_dir, args.base_model)
    print(f"[postprocess] base model id: {base_model_id}")
    target_cache_dir = cache_dir or (output_dir / "hf_cache")
    print(f"[postprocess] downloading base model into hf cache: {target_cache_dir}")
    return download_base_model(
        model_id=base_model_id,
        base_dir=base_dir,
        hf_cache_dir=target_cache_dir if base_dir is None else None,
        force_download=args.force_download,
        hf_token=args.hf_token,
    )


def infer_export_targets(model_name: str, output_root: Path, args, epoch_label: str) -> Dict[str, Path | str]:
    model_slug = infer_model_slug(model_name)
    artifact_root = output_root.parent
    artifact_stem = f"{model_slug}-rma-new"

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
    ollama_model_name = args.ollama_model_name or f"{model_slug}-new"

    return {
        "merged_dir": merged_dir,
        "gguf_path": gguf_path,
        "modelfile_path": modelfile_path,
        "ollama_model_name": ollama_model_name,
    }


def postprocess_trained_model(
    args,
    model_name: str,
    output_dir: Path,
    output_root: Path,
    preexisting_checkpoint_names: set[str],
    num_train_epochs: float,
    final_global_step: int | None,
    cache_dir: Path | None = None,
):
    epoch_label = resolve_epoch_label(args.export_checkpoint_epoch, num_train_epochs)
    targets = infer_export_targets(model_name, output_root, args, epoch_label)

    adapter_dir = select_export_adapter_dir(
        output_dir=output_dir,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        export_checkpoint_epoch=args.export_checkpoint_epoch,
        export_source=args.export_source,
        final_global_step=final_global_step,
    )
    base_model_path = resolve_base_model_path(
        model_name,
        adapter_dir,
        args,
        output_dir,
        cache_dir=cache_dir,
    )

    merged_dir = targets["merged_dir"]
    gguf_path = targets["gguf_path"]
    modelfile_path = targets["modelfile_path"]
    ollama_model_name = targets["ollama_model_name"]
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()

    merged_dir.mkdir(parents=True, exist_ok=True)
    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    modelfile_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[postprocess] merge output dir: {merged_dir}")
    merge_adapter(
        base_model_path,
        adapter_dir,
        merged_dir,
        args.merge_dtype,
        args.trust_remote_code,
    )

    run_command(
        [
            sys.executable,
            "convert_hf_to_gguf.py",
            str(merged_dir),
            "--outfile",
            str(gguf_path),
            "--outtype",
            args.gguf_outtype,
        ],
        cwd=str(llama_cpp_dir),
    )

    modelfile_path.write_text(
        f"FROM {gguf_path}\nPARAMETER temperature 0\n",
        encoding="utf-8",
    )
    print(f"[postprocess] wrote Modelfile: {modelfile_path}")

    ollama_env = os.environ.copy()
    ollama_env["OLLAMA_HOST"] = normalize_ollama_host(args.ollama_host)
    ollama_env["OLLAMA_MODELS"] = args.ollama_models_dir
    if args.ollama_lib_path:
        ollama_env["LD_LIBRARY_PATH"] = args.ollama_lib_path

    ensure_ollama_server(args.ollama_bin, ollama_env, args.ollama_host)
    run_command(
        [
            args.ollama_bin,
            "create",
            str(ollama_model_name),
            "-f",
            str(modelfile_path),
        ],
        env=ollama_env,
    )

    print(f"[postprocess] adapter dir : {adapter_dir}")
    print(f"[postprocess] merged dir  : {merged_dir}")
    print(f"[postprocess] gguf path   : {gguf_path}")
    print(f"[postprocess] modelfile   : {modelfile_path}")
    print(f"[postprocess] ollama model: {ollama_model_name}")


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
    model_name = args.model_name

    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir.parent / "datasets"
    train_dir = dataset_dir / "train"
    additional_dir = dataset_dir / "train" / "additional"

    output_root = Path(args.output_root).expanduser()
    model_root_dir, output_dir, cache_dir = resolve_model_storage_paths(
        output_root,
        model_name,
        args.output_tag,
    )

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
    print(f"dataset_dir: {dataset_dir}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")

    if args.postprocess_only:
        postprocess_trained_model(
            args=args,
            model_name=model_name,
            output_dir=output_dir,
            output_root=output_root,
            preexisting_checkpoint_names=preexisting_checkpoint_names,
            num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
            final_global_step=None,
            cache_dir=cache_dir,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if "google/gemma" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.gradient_checkpointing_enable()

    files = collect_training_files(train_dir, additional_dir)
    print([str(file) for file in files])
    dataframes = [pd.read_csv(file, sep="\t", dtype=str) for file in files]
    df_all = pd.concat(dataframes, ignore_index=True, sort=False)
    raw_datasets = Dataset.from_pandas(df_all)

    prompt_template = select_prompt_template(model_name)
    max_length = 1536

    def preprocess_example_it(example):
        data = {
            "conversation_history": example["conversation_history"],
            "query": example["query"],
        }
        prompt = prompt_template.format(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            answer={"rewrited_query": example["rewrited_query"]},
        )

        tokenized = tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        input_ids = tokenized["input_ids"]

        if model_name == "meta-llama/Llama-3.2-3B-Instruct":
            model_start = prompt.find("assistant<|end_header_id|>")
        elif model_name == "google/gemma-3-4b-it":
            model_start = prompt.find("<start_of_turn>model")
        elif model_name == "microsoft/Phi-4-mini-instruct":
            model_start = prompt.find("<|end|><|assistant|>")
        elif "Qwen/" in model_name:
            model_start = prompt.find("<|im_end|><|im_start|>assistant")
        else:
            raise ValueError(f"Unsupported model_name for label masking: {model_name}")

        if model_start == -1:
            raise ValueError("Prompt does not contain finish prompt")

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

    processed_train = raw_datasets.map(preprocess_example_it)

    print(processed_train[0]["strprompt"])
    max_total_length = 0
    for example in processed_train:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_total_length:
            max_total_length = total_length
    print(f"input_ids + labels max length: {max_total_length}")

    model = get_peft_model(model, build_lora_config(model_name))

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
        num_train_epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        logging_steps=50,
        logging_dir=str(output_dir / "logs"),
        save_strategy="epoch",
        save_total_limit=8,
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
        model_name=model_name,
        output_dir=output_dir,
        output_root=output_root,
        preexisting_checkpoint_names=preexisting_checkpoint_names,
        num_train_epochs=training_args.num_train_epochs,
        final_global_step=final_global_step,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    main()
