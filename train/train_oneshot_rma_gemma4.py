#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoTokenizer, TrainerCallback

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
DEFAULT_OUTPUT_TAG = "oneshot-rma-rewrite-first-block-gemma4"
PRIMARY_QUERY_FIELD = "rewritten_query"
TAGGED_ONESHOT_SYSTEM_PROMPT = (
    "Given a conversation history, a user query, and a list of available tools, "
    "first rewrite the query by resolving only ambiguous pronouns or omitted "
    "references using the conversation history. If there are no ambiguous "
    "pronouns or omitted references, rewritten_query may be identical to the "
    "user query. Then, based on the rewritten_query, select the most appropriate "
    "tool and generate its arguments. Return the answer using exactly these "
    "three sections in this order:\n"
    "<rewritten_query>\n"
    "...rewritten query...\n"
    "</rewritten_query>\n"
    "<plan>\n"
    "...tool name or None...\n"
    "</plan>\n"
    "<arguments>\n"
    "...compact JSON object...\n"
    "</arguments>\n"
    "Always include all three sections. The content inside <arguments> must be "
    "a valid compact JSON object. If no tool matches the request, set the plan "
    "to None and the arguments object to {}."
)
TAG_SEQUENCE = (PRIMARY_QUERY_FIELD, "plan", "arguments")


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
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
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
    parser.add_argument(
        "--probe_every_steps",
        default=100,
        type=int,
        help="Run tagged one-shot generation probes every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--probe_at_steps",
        default="",
        help="Comma-separated optimizer steps at which to run generation probes.",
    )
    parser.add_argument(
        "--probe_sample_idx",
        default=0,
        type=int,
        help="Start index for contiguous generation probe samples.",
    )
    parser.add_argument(
        "--probe_num_samples",
        default=3,
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
        help="Maximum number of new tokens to generate for each probe sample.",
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


def build_tagged_system_message(api_str: str | None = None) -> str:
    if api_str:
        return f"{TAGGED_ONESHOT_SYSTEM_PROMPT}\n<|tool|>{api_str}<|/tool|>"
    return TAGGED_ONESHOT_SYSTEM_PROMPT


def get_example_rewritten_query(example) -> str:
    if isinstance(example.get(PRIMARY_QUERY_FIELD), str):
        return example[PRIMARY_QUERY_FIELD]
    if isinstance(example.get("rewrited_query"), str):
        return example["rewrited_query"]
    raise KeyError(
        f"Example is missing both {PRIMARY_QUERY_FIELD!r} and 'rewrited_query'."
    )


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

    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            "<rewritten_query>",
            get_example_rewritten_query(example),
            "</rewritten_query>",
            "<plan>",
            answer.get("plan", "None"),
            "</plan>",
            "<arguments>",
            arguments_json,
            "</arguments>",
        ]
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


def resolve_probe_log_path(output_dir: Path, probe_log_file: str | None) -> Path:
    if probe_log_file:
        path = Path(probe_log_file).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return (output_dir / "generation_probe.jsonl").resolve()


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


def extract_rewritten_query_section(raw_text: str) -> str:
    section = extract_tagged_section(raw_text, PRIMARY_QUERY_FIELD, TAG_SEQUENCE[1:])
    if section is not None:
        return section
    raise ValueError(f"Missing <{PRIMARY_QUERY_FIELD}> section.")


def parse_oneshot_response(raw_text: str) -> dict:
    rewritten_query = extract_rewritten_query_section(raw_text)
    sections = {PRIMARY_QUERY_FIELD: rewritten_query}
    for idx, tag in enumerate(TAG_SEQUENCE[1:], start=1):
        section = extract_tagged_section(raw_text, tag, TAG_SEQUENCE[idx + 1 :])
        if section is None:
            raise ValueError(f"Missing <{tag}> section.")
        sections[tag] = section

    return {
        PRIMARY_QUERY_FIELD: sections[PRIMARY_QUERY_FIELD],
        "plan": sections["plan"],
        "arguments": parse_arguments_block(sections["arguments"]),
    }


class TaggedGenerationProbeCallback(TrainerCallback):
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
                parsed_result = None
                parse_error = ""
                try:
                    parsed_result = parse_oneshot_response(raw_text)
                except Exception as exc:
                    parse_ok = False
                    parse_error = str(exc)

                matches_gt = (
                    parse_ok
                    and sample["expected_result"] is not None
                    and parsed_result == sample["expected_result"]
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
                        "parsed": parsed_result,
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
            expected_result = parse_oneshot_response(sample["stranswer"])
        except Exception:
            expected_result = None
        samples.append(
            {
                "sample_idx": sample_idx,
                "prompt_text": sample["debug_prompt_only"],
                "expected_text": sample["stranswer"],
                "expected_result": expected_result,
                "debug_query": sample.get("debug_query", ""),
            }
        )

    log_path = resolve_probe_log_path(output_dir, args.probe_log_file)
    return TaggedGenerationProbeCallback(
        tokenizer=tokenizer,
        samples=samples,
        max_len=args.max_length,
        every_steps=args.probe_every_steps,
        at_steps=probe_steps,
        max_new_tokens=args.probe_max_new_tokens,
        log_path=log_path,
    )


def build_preprocess_fn(tokenizer, apis: Dict, max_length: int):
    def preprocess(example):
        candidates = parse_literal(
            example["candidates"],
            field_name="candidates",
            source_file=example.get("source_file"),
        )
        api_str = build_api_str_from_candidates(candidates, apis)
        target_text = build_oneshot_target(example)
        system_msg = build_tagged_system_message(api_str)
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
            messages + [{"role": "assistant", "content": target_text}],
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
            "debug_prompt_only": prompt_prefix,
            "strprompt": prompt,
            "stranswer": target_text,
            "debug_query": example["query"],
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
