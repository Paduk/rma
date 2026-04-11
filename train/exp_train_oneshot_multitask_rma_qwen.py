import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from rma_model_profiles import (
    RMA_MODEL_PROFILES,
    resolve_profile_max_length,
    resolve_profile_model_name,
)
from llama_prompts import (
    SFT_RMA_INFERENCE_LLAMA,
    SFT_RMA_INFERENCE_PHI4,
    SFT_RMA_INFERENCE_QWEN3,
    SFT_RMA_TRAIN_LLAMA,
    SFT_RMA_TRAIN_PHI4,
    SFT_RMA_TRAIN_QWEN3,
)
from train_multitask_rma import (
    build_planning_preprocess_fn,
    build_rewrite_preprocess_fn as build_generic_rewrite_preprocess_fn,
    collect_planning_training_files,
)
from train_oneshot_rma_qwen import (
    DEFAULT_ADDITIONAL_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAIN_DIR,
    DEFAULT_TOOLS_PATH,
    DataCollatorForCausalLM,
    build_preprocess_fn as build_oneshot_preprocess_fn,
    configure_tokenizer_padding,
    collect_training_files as collect_oneshot_training_files,
    ensure_supported_model,
    get_config_architecture,
    max_total_length,
    maybe_auto_skip_unsupported_postprocess,
    read_simple_apis,
    resolve_path,
    resolve_trust_remote_code_arg,
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
    collect_training_files as collect_rewrite_training_files,
    infer_model_slug,
    postprocess_trained_model,
    resolve_model_storage_paths,
)


DEFAULT_OUTPUT_TAG = "exp-oneshot-rewrite-multitask-rma"
DEFAULT_TASK_COMBO = "oneshot,rewrite"
TASK_COMBO_CHOICES = [
    "oneshot",
    "oneshot,rewrite",
    "oneshot,planning",
    "oneshot,rewrite,planning",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a configurable one-shot/rewrite/planning multitask LoRA adapter."
    )
    parser.add_argument(
        "--profile",
        choices=sorted(RMA_MODEL_PROFILES.keys()),
        default=None,
        help="Named model profile. --model_name overrides the profile model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
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
        help="Directory containing additional TSV files.",
    )
    parser.add_argument(
        "--tools_path",
        default=str(DEFAULT_TOOLS_PATH),
        help="Path to simple_api.json.",
    )
    parser.add_argument(
        "--mix_strategy",
        choices=["balanced", "concat"],
        default="balanced",
        help="How to mix the enabled task datasets.",
    )
    parser.add_argument(
        "--task_combo",
        choices=TASK_COMBO_CHOICES,
        default=DEFAULT_TASK_COMBO,
        help=(
            "Which tasks to train together. Supported values: "
            "oneshot, oneshot,rewrite, oneshot,planning, "
            "oneshot,rewrite,planning."
        ),
    )
    parser.add_argument(
        "--rewrite_sampling_prob",
        type=float,
        default=0.5,
        help=(
            "Rewrite sampling probability when mix_strategy=balanced and rewrite is "
            "included in --task_combo. If planning is also included, the remaining "
            "probability is split equally between one-shot and planning."
        ),
    )
    parser.add_argument(
        "--include_planning",
        action="store_true",
        help=(
            "Deprecated compatibility flag. Equivalent to adding planning to "
            "--task_combo."
        ),
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Shuffle/interleave seed.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Token truncation length.",
    )
    parser.add_argument(
        "--chat_template_fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback for tokenizers without chat_template.",
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
        "--strict_postprocess",
        action="store_true",
        help="Attempt GGUF/Ollama postprocess even for model architectures known to be unsupported locally.",
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


def dataframe_to_dataset(files: List[Path], task_name: str) -> Dataset:
    if not files:
        raise ValueError(f"No TSV files found for task={task_name}")

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


def select_rewrite_templates(model_name: str) -> Tuple[str, str]:
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_RMA_TRAIN_LLAMA, SFT_RMA_INFERENCE_LLAMA
    if model_name == "microsoft/Phi-4-mini-instruct":
        return SFT_RMA_TRAIN_PHI4, SFT_RMA_INFERENCE_PHI4
    if "Qwen/" in model_name:
        return SFT_RMA_TRAIN_QWEN3, SFT_RMA_INFERENCE_QWEN3
    raise ValueError(f"Unsupported model_name for rewrite task: {model_name}")


def build_rewrite_preprocess_fn(tokenizer, model_name: str, max_length: int):
    train_template, inference_template = select_rewrite_templates(model_name)

    def preprocess(example):
        data = {
            "conversation_history": example["conversation_history"],
            "query": example["query"],
        }
        prompt_prefix = inference_template.format(
            data=json.dumps(data, ensure_ascii=False, indent=2),
        )
        prompt = train_template.format(
            data=json.dumps(data, ensure_ascii=False, indent=2),
            answer={"rewrited_query": example["rewrited_query"]},
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
            "stranswer": example["rewrited_query"],
        }

    return preprocess


def resolve_task_combo(args) -> List[str]:
    if args.include_planning:
        if args.task_combo == DEFAULT_TASK_COMBO:
            args.task_combo = "oneshot,rewrite,planning"
        elif "planning" not in args.task_combo:
            raise ValueError(
                "--include_planning conflicts with --task_combo values that exclude planning."
            )
    return args.task_combo.split(",")


def mix_processed_datasets(task_datasets: Dict[str, Dataset], args) -> Dataset:
    task_order = ["oneshot", "rewrite", "planning"]
    datasets = [task_datasets[task] for task in task_order if task in task_datasets]

    if not datasets:
        raise ValueError("At least one task dataset must be provided.")

    if len(datasets) == 1:
        return datasets[0]

    if args.mix_strategy == "concat":
        return concatenate_datasets(datasets).shuffle(seed=args.shuffle_seed)

    if "rewrite" not in task_datasets:
        probabilities = [0.5, 0.5]
    else:
        rewrite_prob = args.rewrite_sampling_prob
        if not 0.0 < rewrite_prob < 1.0:
            raise ValueError("--rewrite_sampling_prob must be between 0 and 1.")

        if "planning" not in task_datasets:
            probabilities = [1.0 - rewrite_prob, rewrite_prob]
        else:
            remaining_prob = 1.0 - rewrite_prob
            probabilities = [
                remaining_prob / 2.0,
                rewrite_prob,
                remaining_prob / 2.0,
            ]

    return interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=args.shuffle_seed,
        stopping_strategy="all_exhausted",
    )


def apply_postprocess_defaults(args, model_name: str, output_root: Path):
    model_slug = infer_model_slug(model_name)
    artifact_root = output_root.parent
    task_combo_slug = args.task_combo.replace(",", "-")
    artifact_stem = f"{model_slug}-rma-{task_combo_slug}-multitask"

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
        args.ollama_model_name = f"{model_slug}-{task_combo_slug}-multitask"


def main():
    args = parse_args()
    args.model_name = resolve_profile_model_name(
        profile=args.profile,
        model_name=args.model_name,
        default_model_name=DEFAULT_MODEL_NAME,
    )
    args.max_length = resolve_profile_max_length(
        profile=args.profile,
        max_length=args.max_length,
        default_max_length=1536,
    )
    ensure_supported_model(args.model_name)
    selected_tasks = resolve_task_combo(args)

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
    trust_remote_code = resolve_trust_remote_code_arg(args, default=True)
    config = AutoConfig.from_pretrained(
        args.model_name,
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
    print(f"model_name: {args.model_name}")
    print(f"model_architecture: {get_config_architecture(config)}")
    print(f"train_dir: {train_dir}")
    print(f"additional_dir: {additional_dir}")
    print(f"tools_path: {tools_path}")
    print(f"model_root_dir: {model_root_dir}")
    print(f"training_output_dir: {output_dir}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"mix_strategy: {args.mix_strategy}")
    print(f"task_combo: {args.task_combo}")
    print(f"rewrite_sampling_prob: {args.rewrite_sampling_prob}")
    print(f"include_planning: {args.include_planning}")
    print(f"default_merged_dir: {args.merged_dir}")
    print(f"default_gguf_path: {args.gguf_path}")
    print(f"default_ollama_model_name: {args.ollama_model_name}")
    print(f"chat_template_fallback: {args.chat_template_fallback}")

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
        trust_remote_code=trust_remote_code,
    )
    added_tokens = configure_tokenizer_padding(tokenizer)

    if "google/gemma" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=str(cache_dir),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            device_map="auto",
        )
    if added_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    oneshot_files = collect_oneshot_training_files(train_dir, additional_dir)
    rewrite_files = (
        collect_rewrite_training_files(train_dir, additional_dir)
        if "rewrite" in selected_tasks
        else []
    )
    planning_files = (
        collect_planning_training_files(train_dir, additional_dir)
        if "planning" in selected_tasks
        else []
    )
    print("oneshot_files:", [str(file) for file in oneshot_files])
    if "rewrite" in selected_tasks:
        print("rewrite_files:", [str(file) for file in rewrite_files])
    if "planning" in selected_tasks:
        print("planning_files:", [str(file) for file in planning_files])

    oneshot_raw = dataframe_to_dataset(oneshot_files, "oneshot")
    rewrite_raw = (
        dataframe_to_dataset(rewrite_files, "rewrite")
        if "rewrite" in selected_tasks
        else None
    )
    planning_raw = (
        dataframe_to_dataset(planning_files, "planning")
        if "planning" in selected_tasks
        else None
    )
    apis = read_simple_apis(tools_path)

    oneshot_processed = oneshot_raw.map(
        build_oneshot_preprocess_fn(
            tokenizer=tokenizer,
            model_name=args.model_name,
            apis=apis,
            max_length=args.max_length,
            chat_template_fallback=args.chat_template_fallback,
        ),
        remove_columns=oneshot_raw.column_names,
        desc="Preprocessing one-shot task",
    )
    rewrite_processed = (
        rewrite_raw.map(
            build_generic_rewrite_preprocess_fn(
                tokenizer=tokenizer,
                model_name=args.model_name,
                max_length=args.max_length,
                chat_template_fallback=args.chat_template_fallback,
            ),
            remove_columns=rewrite_raw.column_names,
            desc="Preprocessing rewrite task",
        )
        if rewrite_raw is not None
        else None
    )
    planning_processed = None
    if planning_raw is not None:
        planning_processed = planning_raw.map(
            build_planning_preprocess_fn(
                tokenizer=tokenizer,
                model_name=args.model_name,
                apis=apis,
                max_length=args.max_length,
                chat_template_fallback=args.chat_template_fallback,
            ),
            remove_columns=planning_raw.column_names,
            desc="Preprocessing planning task",
        )

    print(f"oneshot_examples: {len(oneshot_processed)}")
    if rewrite_processed is not None:
        print(f"rewrite_examples: {len(rewrite_processed)}")
    if planning_processed is not None:
        print(f"planning_examples: {len(planning_processed)}")
    print("oneshot sample prompt:")
    print(oneshot_processed[0]["strprompt"])
    print()
    if rewrite_processed is not None:
        print("rewrite sample prompt:")
        print(rewrite_processed[0]["strprompt"])
        print()
    if planning_processed is not None:
        print("planning sample prompt:")
        print(planning_processed[0]["strprompt"])
        print()
    print(f"oneshot max total length: {max_total_length(oneshot_processed)}")
    if rewrite_processed is not None:
        print(f"rewrite max total length: {max_total_length(rewrite_processed)}")
    if planning_processed is not None:
        print(f"planning max total length: {max_total_length(planning_processed)}")

    task_datasets = {"oneshot": oneshot_processed}
    if rewrite_processed is not None:
        task_datasets["rewrite"] = rewrite_processed
    if planning_processed is not None:
        task_datasets["planning"] = planning_processed

    processed_train = mix_processed_datasets(task_datasets=task_datasets, args=args)
    print(f"mixed_examples: {len(processed_train)}")
    print(f"mixed max total length: {max_total_length(processed_train)}")

    model = get_peft_model(model, build_lora_config(args.model_name))
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
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
