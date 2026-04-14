# 필수 라이브러리 임포트
import argparse
import torch
from datasets import load_dataset, Dataset
from typing import List, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from llama_prompts import (
    SFT_RMA_INFERENCE_GEMMA,
    SFT_RMA_INFERENCE_LLAMA,
    SFT_RMA_INFERENCE_PHI4,
    SFT_RMA_INFERENCE_QWEN3,
    SFT_RMA_TRAIN_GEMMA,
    SFT_RMA_TRAIN_LLAMA,
    SFT_RMA_TRAIN_PHI4,
    SFT_RMA_TRAIN_QWEN3,
)
#from train_utils import preprocess_example_history, preprocess_example_rewrite
import pdb
import json
import ast
import copy
import os
import glob
import pandas as pd
from pathlib import Path

try:
    from rma_model_profiles import (
        RMA_MODEL_PROFILES,
        resolve_profile_max_length,
        resolve_profile_model_name,
    )
    from train_sentence_rewriter import (
        DEFAULT_LLAMA_CPP_DIR,
        DEFAULT_OLLAMA_BIN,
        DEFAULT_OLLAMA_HOST,
        DEFAULT_OLLAMA_LIB_PATH,
        DEFAULT_OLLAMA_MODELS_DIR,
        infer_model_slug,
        postprocess_trained_model,
    )
except ImportError:
    from train.rma_model_profiles import (
        RMA_MODEL_PROFILES,
        resolve_profile_max_length,
        resolve_profile_model_name,
    )
    from train.train_sentence_rewriter import (
        DEFAULT_LLAMA_CPP_DIR,
        DEFAULT_OLLAMA_BIN,
        DEFAULT_OLLAMA_HOST,
        DEFAULT_OLLAMA_LIB_PATH,
        DEFAULT_OLLAMA_MODELS_DIR,
        infer_model_slug,
        postprocess_trained_model,
    )

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/hj153lee/rma_train")
DEFAULT_MAX_LENGTH = 1536
DEFAULT_NUM_TRAIN_EPOCHS = 3
RMA_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the query clearly by replacing ambiguous pronouns (like \"it\", "
    "\"that\") with explicit information from the conversation history. Keep "
    "exactly the same sentence structure. Do NOT generate or include any "
    "information, words, or values outside of the provided conversation_history "
    "and query."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RMA LoRA adapter.")
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
        help="Root directory for HF cache, epoch checkpoints, and final adapter.",
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
        "--trust_remote_code",
        choices=["auto", "true", "false"],
        default="auto",
        help="Pass trust_remote_code to Hugging Face loaders. `auto` defaults to false for Phi and true otherwise.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=DEFAULT_NUM_TRAIN_EPOCHS,
        type=float,
        help="Training epochs.",
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


def resolve_trust_remote_code_arg(args, default: bool = True) -> bool:
    if args.trust_remote_code == "true":
        return True
    if args.trust_remote_code == "false":
        return False
    return default


def default_trust_remote_code_for_model(model_name: str) -> bool:
    lower_name = model_name.lower()
    if "phi-4" in lower_name or "phi4" in lower_name:
        return False
    return True


def configure_tokenizer_padding(tokenizer, max_length: int) -> int:
    added_tokens = 0
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            added_tokens = tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length
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


def render_messages_as_plain_text(messages, add_generation_prompt: bool) -> str:
    sections = []
    for message in messages:
        role = str(message.get("role", "")).strip().capitalize() or "User"
        content = str(message.get("content", "")).strip()
        sections.append(f"{role}:\n{content}")
    if add_generation_prompt:
        sections.append("Assistant:\n")
    return "\n\n".join(sections)


def render_model_messages(tokenizer, messages, add_generation_prompt: bool, chat_template_fallback: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return render_chat_template(
            tokenizer=tokenizer,
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    if chat_template_fallback == "error":
        raise ValueError(
            "Tokenizer does not define chat_template. Use --chat_template_fallback simple "
            "for base-model RMA SFT."
        )
    return render_messages_as_plain_text(messages, add_generation_prompt=add_generation_prompt)


def select_legacy_rma_templates(model_name: str):
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return SFT_RMA_TRAIN_LLAMA, SFT_RMA_INFERENCE_LLAMA
    if model_name == "google/gemma-3-4b-it":
        return SFT_RMA_TRAIN_GEMMA, SFT_RMA_INFERENCE_GEMMA
    if model_name == "microsoft/Phi-4-mini-instruct":
        return SFT_RMA_TRAIN_PHI4, SFT_RMA_INFERENCE_PHI4
    if "Qwen/" in model_name:
        return SFT_RMA_TRAIN_QWEN3, SFT_RMA_INFERENCE_QWEN3
    return None, None


def build_rma_prompt_pair(example, tokenizer, model_name: str, max_length: int, chat_template_fallback: str):
    data = {
        "conversation_history": example["conversation_history"],
        "query": example["query"],
    }
    data_json = json.dumps(data, ensure_ascii=False, indent=2)
    assistant_content = {"rewrited_query": example["rewrited_query"]}
    train_template, inference_template = select_legacy_rma_templates(model_name)

    if train_template is not None and inference_template is not None:
        prompt = train_template.format(data=data_json, answer=assistant_content)
        prompt_prefix = inference_template.format(data=data_json)
        return prompt, prompt_prefix, assistant_content

    messages = [
        {"role": "system", "content": RMA_REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": data_json},
    ]
    prompt_prefix = render_model_messages(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=True,
        chat_template_fallback=chat_template_fallback,
    )
    prompt = render_model_messages(
        tokenizer=tokenizer,
        messages=messages + [{"role": "assistant", "content": str(assistant_content)}],
        add_generation_prompt=False,
        chat_template_fallback=chat_template_fallback,
    )
    return prompt, prompt_prefix, assistant_content


def tokenize_prompt_pair(tokenizer, prompt: str, prompt_prefix: str, max_length: int):
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


def uses_all_linear_lora(model_name: str) -> bool:
    lower_name = model_name.lower()
    all_linear_markers = (
        "phi-4",
        "phi4",
        "qwen",
        "glm-edge",
        "smollm",
        "falcon3",
        "exaone",
        "olmo",
        "granite",
        "lfm",
    )
    return any(marker in lower_name for marker in all_linear_markers)


def build_lora_config(model_name: str) -> LoraConfig:
    if uses_all_linear_lora(model_name):
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


def get_lora_prefix(args, model_name: str) -> str:
    if args.profile:
        return RMA_MODEL_PROFILES[args.profile].prefix
    if uses_all_linear_lora(model_name):
        return "all_linear"
    return "lora"


def apply_postprocess_defaults(args, output_root: Path, model_name: str):
    model_slug = infer_model_slug(model_name)
    profile_slug = args.profile or model_slug
    prefix = get_lora_prefix(args, model_name)
    artifact_root = output_root.parent
    artifact_stem = f"{model_slug}-{profile_slug}-rma-{prefix}"

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


args = parse_args()
model_name = resolve_profile_model_name(
    profile=args.profile,
    model_name=args.model_name,
    default_model_name=DEFAULT_MODEL_NAME,
)
max_length = resolve_profile_max_length(
    profile=args.profile,
    max_length=args.max_length,
    default_max_length=DEFAULT_MAX_LENGTH,
)
default_trust_remote_code = default_trust_remote_code_for_model(model_name)
if args.trust_remote_code == "auto" and not default_trust_remote_code:
    args.trust_remote_code = "false"
trust_remote_code = resolve_trust_remote_code_arg(
    args,
    default=default_trust_remote_code,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "datasets"
TRAIN_DIR = DATASET_DIR / "train"
TC_DIR = TRAIN_DIR / "additional"
#TC_DIR = DATASET_DIR / "tc"

output_root = Path(args.output_root).expanduser()
model_dirname = model_name.replace("/", "__")
output_dir = output_root / model_dirname
cache_dir = output_dir / "hf_cache"

apply_postprocess_defaults(args, output_root, model_name)

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
print(f"profile: {args.profile}")
print(f"dataset_dir: {DATASET_DIR}")
print(f"hf_cache_dir: {cache_dir}")
print(f"training_output_dir: {output_dir}")
print(f"merged_dir: {args.merged_dir}")
print(f"gguf_path: {args.gguf_path}")
print(f"ollama_model_name: {args.ollama_model_name}")
print(f"max_length: {max_length}")
print(f"chat_template_fallback: {args.chat_template_fallback}")
print(f"trust_remote_code: {trust_remote_code}")

if args.postprocess_only:
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
    raise SystemExit(0)

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=str(cache_dir),
    trust_remote_code=trust_remote_code,
)
added_tokens = configure_tokenizer_padding(tokenizer, max_length)

if 'google/gemma' in model_name.lower():
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=str(cache_dir),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
        device_map="auto"
    )

if added_tokens:
    model.resize_token_embeddings(len(tokenizer))

# 메모리 효율성을 위해 gradient_checkpointing 활성화
model.gradient_checkpointing_enable()

# 데이터 로딩 및 전처리
#train_tsv_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.startswith('o4_') and f.endswith('.tsv') and 'train' in f]
train_path = TRAIN_DIR
tc_path = TC_DIR

files = sorted([file for file in train_path.glob("*.tsv") if "_NR_" not in file.name])
#added_files = [file for file in tc_path.glob("*.tsv") if "_NR_" not in file.name]
added_files = sorted([file for file in tc_path.glob("*.tsv") if ("complex" in file.name or "various_nonNR" in file.name)])
files.append(train_path / "it2_NR_train.tsv")
files.extend(added_files)
print([str(file) for file in files])
dfs = [pd.read_csv(f, sep="\t", dtype=str) for f in files]
df_all = pd.concat(dfs, ignore_index=True, sort=False)
raw_datasets = Dataset.from_pandas(df_all)
# data_files = {'train': train_tsv_files}
# raw_datasets = load_dataset('csv', data_files=data_files, delimiter='\t')

def preprocess_example_it(example):
    prompt, prompt_prefix, assistant_content = build_rma_prompt_pair(
        example=example,
        tokenizer=tokenizer,
        model_name=model_name,
        max_length=max_length,
        chat_template_fallback=args.chat_template_fallback,
    )
    input_ids, labels = tokenize_prompt_pair(
        tokenizer=tokenizer,
        prompt=prompt,
        prompt_prefix=prompt_prefix,
        max_length=max_length,
    )
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "strprompt": prompt,        
        "stranswer": json.dumps(assistant_content, indent=2, ensure_ascii=False),
    }

processed_train = raw_datasets.map(
    preprocess_example_it,     
)


# decoded_input = tokenizer.decode(processed_train[0]['input_ids'], skip_special_tokens=False)
# print("=== Decoded Input ===")
# print(decoded_input)

print(processed_train[0]["strprompt"])
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
        max_label_length = max(len(f["labels"]) for f in features)
        for f in features:
            padding_length = max_label_length - len(f["labels"])
            f["labels"] = f["labels"] + [label_pad_token_id] * padding_length
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        return batch

data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

lora_config = build_lora_config(model_name)
# LoRA 적용 모델 생성
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 비율 출력 (옵션)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"학습 가능 파라미터: {trainable_params}, 전체 파라미터: {total_params}, 학습 비율: {100 * trainable_params / total_params:.2f}%")

training_args = TrainingArguments(
    output_dir=str(output_dir),
    overwrite_output_dir=True,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    logging_steps=50,
    logging_dir=str(output_dir / "logs"),
    # PEFT 모델 기준으로 epoch checkpoint가 저장됩니다.
    save_strategy="epoch",
    save_total_limit=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train,
    data_collator=data_collator,
    tokenizer=tokenizer,    
)

trainer.train()
final_global_step = trainer.state.global_step

# 학습 완료된 LoRA 어댑터 모델 저장
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))

if args.skip_postprocess:
    print("[postprocess] skipped by --skip_postprocess")
else:
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
