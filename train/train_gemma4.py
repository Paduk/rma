#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone Gemma 4 trainer for the legacy planner dataset.

Recommended first run:
  python3 train_gemma4.py --skip_postprocess

Gemma 4 support requires a recent Transformers build. The official Hugging Face
examples use:
  transformers[audio,chat_template,kernels,video,vision]>=5.5.0
"""

import argparse
import ast
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from gemma4_legacy_prompting import (
    LEGACY_SYSTEM_HISTORY_PROMPT,
    LEGACY_SYSTEM_REWRITE_PROMPT,
    apply_chat_template,
)
from planner_json_utils import normalize_answer_to_dict
from postprocess_gemma4 import (
    DEFAULT_GEMMA4_LLAMA_CPP_DIR,
    POSTPROCESS_STAGE_CHOICES,
    postprocess_gemma4_trained_model,
)
from train_sentence_rewriter import (
    DEFAULT_OLLAMA_BIN,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_LIB_PATH,
    DEFAULT_OLLAMA_MODELS_DIR,
    DEFAULT_OUTPUT_ROOT,
    resolve_model_storage_paths,
)


DEFAULT_MODEL_NAME = "google/gemma-4-E2B-it"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

LEGACY_TOOLS_PATH = PROJECT_ROOT / "apis" / "simple_api.json"
LEGACY_TRAIN_DIR = PROJECT_ROOT / "datasets" / "train"

LORA_MODULE_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Gemma 4 LoRA trainer for legacy planner data."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Gemma 4 base model ID, e.g. google/gemma-4-E2B-it or google/gemma-4-E4B-it.",
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
        default="history",
        choices=["history", "rewrite"],
        help="Whether to train on history-aware or rewrite-only examples.",
    )
    parser.add_argument(
        "--prefix",
        default="lm_only_lora",
        help="Suffix used in the run directory name.",
    )
    parser.add_argument(
        "--max_len",
        default=1536,
        type=int,
        help="Maximum sequence length for truncation.",
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
        default=5,
        type=float,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=2,
        type=int,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=4,
        type=int,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.05,
        type=float,
        help="Warmup ratio for the LR scheduler.",
    )
    parser.add_argument(
        "--logging_steps",
        default=50,
        type=int,
        help="Logging interval.",
    )
    parser.add_argument(
        "--save_total_limit",
        default=8,
        type=int,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--lora_r",
        default=8,
        type=int,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora_alpha",
        default=16,
        type=int,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora_dropout",
        default=0.1,
        type=float,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        help="Attention backend passed to from_pretrained.",
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
        "--postprocess_stage",
        choices=POSTPROCESS_STAGE_CHOICES,
        default="all",
        help="Run only a specific postprocess stage or the full chain.",
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
        default="final_adapter",
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
        default=DEFAULT_GEMMA4_LLAMA_CPP_DIR,
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
        help="Force a fresh download from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token for gated model access.",
    )
    parser.add_argument(
        "--debug_print_samples",
        default=1,
        type=int,
        help="Number of preprocessed training samples to print for prompt/answer debugging. Use 0 to disable.",
    )
    parser.add_argument(
        "--probe_every_steps",
        default=0,
        type=int,
        help="Run a single-sample generation probe every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--probe_at_steps",
        default="",
        help="Comma-separated optimizer steps at which to run the single-sample generation probe.",
    )
    parser.add_argument(
        "--probe_sample_idx",
        default=0,
        type=int,
        help="Start index for contiguous generation probe samples.",
    )
    parser.add_argument(
        "--probe_num_samples",
        default=5,
        type=int,
        help="Number of contiguous processed training samples to probe from --probe_sample_idx.",
    )
    parser.add_argument(
        "--probe_sample_indices",
        default="",
        help="Optional comma-separated processed training sample indices to probe. Overrides --probe_sample_idx/--probe_num_samples.",
    )
    parser.add_argument(
        "--probe_max_new_tokens",
        default=128,
        type=int,
        help="Maximum number of new tokens to generate for the single-sample generation probe.",
    )
    parser.add_argument(
        "--probe_log_file",
        default=None,
        help="Optional JSONL file for generation probe logs. Defaults to <output_dir>/generation_probe.jsonl when probing is enabled.",
    )
    return parser.parse_args()


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_output_dir(output_dir: str | None, default_name: str) -> Path:
    target = default_name if output_dir is None else output_dir
    return resolve_path(target)


def make_run_name(model_name: str, train_type: str, prefix: str) -> str:
    model_tag = re.sub(r"[^a-z0-9]+", "-", model_name.split("/")[-1].lower()).strip("-")
    return f"{model_tag}-{train_type}-{prefix}"


def resolve_storage_paths(args):
    output_root = Path(args.output_root).expanduser().resolve()
    default_run_name = make_run_name(args.model_name, args.train_type, args.prefix)
    _, default_output_dir, cache_dir = resolve_model_storage_paths(
        output_root,
        args.model_name,
        default_run_name,
    )
    output_dir = (
        resolve_output_dir(args.output_dir, default_run_name)
        if args.output_dir is not None
        else default_output_dir
    )
    return output_root, output_dir, cache_dir, default_run_name


def apply_postprocess_defaults(args, output_root: Path):
    artifact_root = output_root.parent
    artifact_stem = make_run_name(args.model_name, args.train_type, args.prefix)

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
    preexisting_checkpoint_names: set[str],
):
    if not args.postprocess_only:
        return False

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Training output directory not found for --postprocess_only: {output_dir}"
        )

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
    return True


def read_simple_apis(api_file: str | Path):
    with open(resolve_path(api_file), "r", encoding="utf-8") as file:
        return json.load(file)


def collect_legacy_train_files(train_dir: str | Path):
    train_dir_path = resolve_path(train_dir)
    train_files = sorted(str(path) for path in train_dir_path.glob("*.tsv"))

    additional_dir = train_dir_path / "additional"
    if additional_dir.is_dir():
        train_files.extend(sorted(str(path) for path in additional_dir.glob("*.tsv")))

    return train_files


def build_api_str(example, apis):
    api_str = ""
    candidates = ast.literal_eval(example["candidates"])
    for plan in candidates:
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
    return api_str


def build_planner_target_text(answer) -> str:
    normalized = normalize_answer_to_dict(answer)

    plan = normalized.get("plan", "None")
    arguments = normalized.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("answer['arguments'] must be a dict.")

    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            "<plan>",
            plan,
            "</plan>",
            "<arguments>",
            arguments_json,
            "</arguments>",
        ]
    )


def extract_tagged_section(text: str, tag: str, following_tags: tuple[str, ...]) -> str | None:
    start_match = re.search(rf"<{tag}>", text, re.IGNORECASE)
    if not start_match:
        return None

    start = start_match.end()
    end_match = re.search(rf"</{tag}>", text[start:], re.IGNORECASE | re.DOTALL)
    if end_match:
        end = start + end_match.start()
        return text[start:end].strip()

    next_positions = []
    for next_tag in following_tags:
        next_match = re.search(rf"<{next_tag}>", text[start:], re.IGNORECASE)
        if next_match:
            next_positions.append(start + next_match.start())
    end = min(next_positions) if next_positions else len(text)
    return text[start:end].strip()


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    if depth > 0 and text[start:].strip().startswith("{"):
        return text[start:] + ("}" * depth)
    return None


def parse_arguments_block(arguments_text: str) -> dict:
    candidate = arguments_text.strip()
    if not candidate:
        raise ValueError("arguments block is empty.")

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate = extract_first_json_object(candidate)
        if not candidate:
            raise ValueError("No JSON object found in arguments block.")
        parsed = json.loads(candidate)

    if not isinstance(parsed, dict):
        raise ValueError("arguments block did not parse to a dict.")
    return parsed


def parse_planner_target_text(raw_text: str) -> dict:
    plan = extract_tagged_section(raw_text, "plan", ("arguments",))
    arguments_text = extract_tagged_section(raw_text, "arguments", ())

    if plan is None or arguments_text is None:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Planner target text did not parse to a dict.")
        if "plan" not in parsed or "arguments" not in parsed:
            raise ValueError("Planner target text must contain plan and arguments.")
        if not isinstance(parsed["arguments"], dict):
            raise ValueError("Planner target arguments must be a dict.")
        return parsed

    return {
        "plan": plan,
        "arguments": parse_arguments_block(arguments_text),
    }


def print_max_total_length(processed_train):
    max_total_length = 0
    for example in processed_train:
        total_length = len(example["input_ids"]) + len(example["labels"])
        if total_length > max_total_length:
            max_total_length = total_length
    print(f"input_ids + labels 최대 길이: {max_total_length}")


def print_debug_examples(processed_train, limit: int):
    if limit <= 0:
        return

    sample_count = min(limit, len(processed_train))
    for idx in range(sample_count):
        sample = processed_train[idx]
        print(f"\n=== Debug Sample {idx} Prompt Only ===")
        print(sample["debug_prompt_only"])
        print(f"\n=== Debug Sample {idx} Prompt + Answer ===")
        print(sample["strprompt"])
        print(f"\n=== Debug Sample {idx} Answer Only ===")
        print(sample["stranswer"])


def print_preprocess_diagnostics(processed_train):
    sample_count = len(processed_train)
    if sample_count == 0:
        return

    truncated_count = sum(1 for sample in processed_train if sample["was_truncated"])
    answer_fully_truncated = sum(
        1 for sample in processed_train if sample["answer_fully_truncated"]
    )
    supervised_token_counts = [sample["supervised_token_count"] for sample in processed_train]
    prompt_token_counts = [sample["prompt_token_count"] for sample in processed_train]
    full_token_counts = [sample["full_token_count"] for sample in processed_train]

    print("\n=== Preprocess Diagnostics ===")
    print(f"samples: {sample_count}")
    print(
        f"truncated_examples: {truncated_count} "
        f"({truncated_count / sample_count:.2%})"
    )
    print(
        f"answer_fully_truncated_examples: {answer_fully_truncated} "
        f"({answer_fully_truncated / sample_count:.2%})"
    )
    print(
        "supervised_tokens "
        f"min/avg/max: {min(supervised_token_counts)}/"
        f"{sum(supervised_token_counts) / sample_count:.1f}/"
        f"{max(supervised_token_counts)}"
    )
    print(
        "prompt_tokens "
        f"min/avg/max: {min(prompt_token_counts)}/"
        f"{sum(prompt_token_counts) / sample_count:.1f}/"
        f"{max(prompt_token_counts)}"
    )
    print(
        "full_tokens "
        f"min/avg/max: {min(full_token_counts)}/"
        f"{sum(full_token_counts) / sample_count:.1f}/"
        f"{max(full_token_counts)}"
    )


def parse_step_list(raw_value: str) -> set[int]:
    if not raw_value.strip():
        return set()

    steps = set()
    for piece in raw_value.split(","):
        token = piece.strip()
        if not token:
            continue
        step = int(token)
        if step < 0:
            raise ValueError(f"Probe steps must be non-negative integers: {raw_value!r}")
        steps.add(step)
    return steps


def resolve_probe_log_path(output_dir: Path, probe_log_file: str | None) -> Path:
    if probe_log_file:
        path = Path(probe_log_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return (output_dir / "generation_probe.jsonl").resolve()


def parse_index_list(raw_value: str) -> list[int]:
    if not raw_value.strip():
        return []

    indices = []
    seen = set()
    for piece in raw_value.split(","):
        token = piece.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0:
            raise ValueError(f"Probe sample indices must be non-negative: {raw_value!r}")
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


class MultiSampleGenerationProbeCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        samples,
        max_len: int,
        every_steps: int,
        at_steps: set[int],
        max_new_tokens: int,
        log_path: Path | None,
    ):
        self.tokenizer = tokenizer
        self.samples = samples
        self.max_len = max_len
        self.every_steps = every_steps
        self.at_steps = at_steps
        self.max_new_tokens = max_new_tokens
        self.log_path = log_path
        self.seen_train_steps = set()
        self.seen_probe_keys = set()

        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def describe(self):
        print("\n=== Generation Probe Config ===")
        print(f"sample_indices: {[sample['sample_idx'] for sample in self.samples]}")
        print(f"every_steps: {self.every_steps}")
        print(f"at_steps: {sorted(self.at_steps)}")
        print(f"max_new_tokens: {self.max_new_tokens}")
        if self.log_path is not None:
            print(f"log_file: {self.log_path}")
        for sample in self.samples:
            print(
                f"sample_idx={sample['sample_idx']} | query={sample.get('debug_query', '')}"
            )

    def _should_run(self, step: int) -> bool:
        if step < 0 or step in self.seen_train_steps:
            return False
        if step in self.at_steps:
            return True
        if self.every_steps > 0 and step > 0 and step % self.every_steps == 0:
            return True
        return False

    def _get_model_input_device(self, model):
        try:
            return model.get_input_embeddings().weight.device
        except Exception:
            return next(model.parameters()).device

    def _append_log_record(self, record: Dict):
        if self.log_path is None:
            return
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_probe(self, model, step: int, stage: str):
        probe_key = (stage, step)
        if probe_key in self.seen_probe_keys:
            return

        core_model = getattr(model, "module", model)
        was_training = core_model.training
        previous_use_cache = getattr(core_model.config, "use_cache", None)

        try:
            core_model.eval()
            if previous_use_cache is not None:
                core_model.config.use_cache = True

            print(f"\n=== Generation Probe @ step {step} ({stage}) ===")
            device = self._get_model_input_device(core_model)
            for sample in self.samples:
                encoded = self.tokenizer(
                    sample["prompt_text"],
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_len,
                )
                encoded = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in encoded.items()
                }

                with torch.inference_mode():
                    generated = core_model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_token_count = encoded["input_ids"].shape[-1]
                generated_ids = generated[0][prompt_token_count:]
                raw_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                ).strip()

                parse_ok = True
                parsed_json = None
                parse_error = ""
                try:
                    parsed_json = parse_planner_target_text(raw_text)
                except Exception as exc:
                    parse_ok = False
                    parse_error = str(exc)

                matches_gt = (
                    parse_ok
                    and sample["expected_json"] is not None
                    and parsed_json == sample["expected_json"]
                )

                print(
                    f"sample_idx={sample['sample_idx']} | "
                    f"parse_ok={parse_ok} | matches_gt={matches_gt}"
                )
                print(raw_text)
                if parse_error:
                    print(f"[probe parse_error] {parse_error}")

                self._append_log_record(
                    {
                        "stage": stage,
                        "step": step,
                        "sample_idx": sample["sample_idx"],
                        "query": sample.get("debug_query", ""),
                        "prompt": sample["prompt_text"],
                        "expected": sample["expected_text"],
                        "raw": raw_text,
                        "parse_ok": parse_ok,
                        "parse_error": parse_error,
                        "parsed": parsed_json,
                        "matches_gt": matches_gt,
                    }
                )
            self.seen_probe_keys.add(probe_key)
            if stage == "train":
                self.seen_train_steps.add(step)
        finally:
            if previous_use_cache is not None:
                core_model.config.use_cache = previous_use_cache
            if was_training:
                core_model.train()

    def on_step_end(self, args, state, control, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return control

        step = int(state.global_step)
        if self._should_run(step):
            model = kwargs.get("model")
            if model is not None:
                self.run_probe(model, step=step, stage="train")
        return control


def maybe_build_generation_probe_callback(args, tokenizer, processed_train, output_dir: Path):
    probe_steps = parse_step_list(args.probe_at_steps)
    if args.probe_every_steps <= 0 and not probe_steps:
        return None

    if args.probe_every_steps < 0:
        raise ValueError("--probe_every_steps must be >= 0.")
    if args.probe_max_new_tokens <= 0:
        raise ValueError("--probe_max_new_tokens must be > 0.")
    if args.probe_num_samples <= 0:
        raise ValueError("--probe_num_samples must be > 0.")
    if not 0 <= args.probe_sample_idx < len(processed_train):
        raise IndexError(
            f"--probe_sample_idx must be between 0 and {len(processed_train) - 1}, "
            f"got {args.probe_sample_idx}."
        )

    sample_indices = parse_index_list(args.probe_sample_indices)
    if not sample_indices:
        end_idx = min(
            len(processed_train),
            args.probe_sample_idx + args.probe_num_samples,
        )
        sample_indices = list(range(args.probe_sample_idx, end_idx))
    if not sample_indices:
        raise ValueError("No valid probe sample indices were resolved.")
    for sample_idx in sample_indices:
        if not 0 <= sample_idx < len(processed_train):
            raise IndexError(
                f"Probe sample index must be between 0 and {len(processed_train) - 1}, "
                f"got {sample_idx}."
            )

    samples = []
    for sample_idx in sample_indices:
        sample = processed_train[sample_idx]
        try:
            expected_json = parse_planner_target_text(sample["stranswer"])
        except Exception:
            expected_json = None
        samples.append(
            {
                "sample_idx": sample_idx,
                "prompt_text": sample["debug_prompt_only"],
                "expected_text": sample["stranswer"],
                "expected_json": expected_json,
                "debug_query": sample.get("debug_query", ""),
            }
        )

    log_path = resolve_probe_log_path(output_dir, args.probe_log_file)
    return MultiSampleGenerationProbeCallback(
        tokenizer=tokenizer,
        samples=samples,
        max_len=args.max_len,
        every_steps=args.probe_every_steps,
        at_steps=probe_steps,
        max_new_tokens=args.probe_max_new_tokens,
        log_path=log_path,
    )


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


def make_preprocess_fn(tokenizer, apis, train_type: str, max_len: int):
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

        assistant_content = build_planner_target_text(example["answer"])

        prompt_messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": assistant_content}
        ]

        prompt_text = apply_chat_template(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_text = apply_chat_template(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )

        prompt_ids_full = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
        full_ids_full = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        overlap = min(len(prompt_ids_full), len(full_ids_full))
        if full_ids_full[:overlap] != prompt_ids_full[:overlap]:
            raise ValueError(
                "Prompt/answer token alignment mismatch before truncation. "
                f"query={example.get('query')!r}"
            )

        prompt_ids = prompt_ids_full[:max_len]
        full_ids = full_ids_full[:max_len]
        overlap = min(len(prompt_ids), len(full_ids))
        if full_ids[:overlap] != prompt_ids[:overlap]:
            raise ValueError(
                "Prompt/answer token alignment mismatch after truncation. "
                f"query={example.get('query')!r}"
            )

        label_start = min(len(prompt_ids), len(full_ids))
        labels = [-100] * label_start + full_ids[label_start:]
        supervised_token_count = max(0, len(full_ids) - label_start)

        return {
            "input_ids": full_ids,
            "labels": labels,
            "debug_prompt_only": prompt_text,
            "strprompt": full_text,
            "stranswer": assistant_content,
            "debug_query": example.get("query", example.get("rewrited_query", "")),
            "prompt_token_count": len(prompt_ids_full),
            "full_token_count": len(full_ids_full),
            "supervised_token_count": supervised_token_count,
            "was_truncated": len(full_ids_full) > max_len,
            "answer_fully_truncated": supervised_token_count == 0,
        }

    return preprocess_example


def load_model(
    model_name: str,
    cache_dir: Path,
    hf_token: str | None,
    force_download: bool,
    attn_implementation: str | None,
):
    kwargs = {
        "cache_dir": str(cache_dir),
        "device_map": "auto",
        "force_download": force_download,
    }
    if hf_token:
        kwargs["token"] = hf_token
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            **kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the Gemma 4 model. "
            "Check that Transformers is new enough for Gemma 4 and that the "
            "requested checkpoint is accessible from your Hugging Face account."
        ) from exc


def freeze_non_text_parameters(model):
    text_prefixes = ("model.language_model", "language_model")
    has_text_prefix = any(
        name.startswith(text_prefixes) for name, _ in model.named_parameters()
    )

    if has_text_prefix:
        for name, param in model.named_parameters():
            if not name.startswith(text_prefixes):
                param.requires_grad = False


def collect_lora_target_modules(model):
    preferred = []
    fallback = []

    for name, _ in model.named_modules():
        if not name:
            continue
        terminal = name.rsplit(".", 1)[-1]
        if terminal not in LORA_MODULE_SUFFIXES:
            continue

        if "language_model" in name:
            preferred.append(name)
            continue

        if ".layers." in name and "vision" not in name and "audio" not in name:
            fallback.append(name)

    target_modules = sorted(set(preferred or fallback))
    if not target_modules:
        raise RuntimeError(
            "Could not find Gemma 4 text attention/MLP modules for LoRA injection."
        )

    print(f"LoRA target modules: {len(target_modules)}")
    return target_modules


def build_lora_config(model, args):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=collect_lora_target_modules(model),
        lora_dropout=args.lora_dropout,
        bias="none",
    )


def build_training_args(output_dir: str, args):
    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "logging_dir": f"{output_dir}/logs",
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "report_to": "none",
    }
    if "overwrite_output_dir" in TrainingArguments.__dataclass_fields__:
        kwargs["overwrite_output_dir"] = True
    return TrainingArguments(**kwargs)


def ensure_gemma4_support():
    if hasattr(transformers, "Gemma4ForConditionalGeneration") or hasattr(
        transformers, "Gemma4ForCausalLM"
    ):
        return

    raise RuntimeError(
        "This environment does not have Gemma 4 model support in transformers. "
        f"Current version: {transformers.__version__}. "
        "Gemma 4 support was added later; upgrade transformers to a recent build "
        "(the official Gemma 4 examples use transformers[audio,chat_template,kernels,video,vision]>=5.5.0)."
    )


def build_trainer(model, training_args, processed_train, tokenizer, callbacks=None):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": processed_train,
        "data_collator": DataCollatorForCausalLM(tokenizer),
    }
    trainer_fields = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_fields:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_fields:
        kwargs["tokenizer"] = tokenizer
    if callbacks:
        kwargs["callbacks"] = callbacks
    return Trainer(**kwargs)


def main():
    args = parse_args()
    if "gemma-4" not in args.model_name.lower():
        raise ValueError(
            "train_gemma4.py is intended for Gemma 4 checkpoints only. "
            "Use a model name such as google/gemma-4-E2B-it or google/gemma-4-E4B-it."
        )
    ensure_gemma4_support()

    output_root, output_dir, cache_dir, run_name = resolve_storage_paths(args)
    apply_postprocess_defaults(args, output_root)

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
    print(f"run_name: {run_name}")
    print(f"hf_cache_dir: {cache_dir}")
    print(f"training_output_dir: {output_dir}")

    if maybe_run_postprocess_only(
        args,
        output_dir,
        output_root,
        cache_dir,
        preexisting_checkpoint_names,
    ):
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=str(cache_dir),
        force_download=args.force_download,
        token=args.hf_token,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.max_len

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

    apis = read_simple_apis(args.tools_path)
    train_files = collect_legacy_train_files(args.train_dir)
    print(train_files)
    raw_ds = load_dataset("csv", data_files={"train": train_files}, delimiter="\t")

    preprocess_fn = make_preprocess_fn(
        tokenizer,
        apis,
        args.train_type,
        args.max_len,
    )
    processed_train = raw_ds["train"].map(
        preprocess_fn,
        desc="Applying Gemma 4 chat template",
    )

    print_debug_examples(processed_train, args.debug_print_samples)
    print_max_total_length(processed_train)
    print_preprocess_diagnostics(processed_train)

    model = get_peft_model(model, build_lora_config(model, args))
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M")

    training_args = build_training_args(str(output_dir), args)
    probe_callback = maybe_build_generation_probe_callback(
        args,
        tokenizer,
        processed_train,
        output_dir,
    )
    callbacks = [probe_callback] if probe_callback is not None else None
    trainer = build_trainer(
        model,
        training_args,
        processed_train,
        tokenizer,
        callbacks=callbacks,
    )

    if probe_callback is not None:
        probe_callback.describe()
        probe_callback.run_probe(model, step=0, stage="before_train")

    trainer.train()
    final_global_step = trainer.state.global_step

    if probe_callback is not None:
        probe_callback.run_probe(model, step=int(final_global_step), stage="after_train")

    trainer.save_model()
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
