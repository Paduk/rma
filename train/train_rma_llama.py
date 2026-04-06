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
from llama_prompts import SFT_RMA_TRAIN_LLAMA
from llama_prompts import SFT_RMA_TRAIN_PHI4
from llama_prompts import SFT_RMA_TRAIN_QWEN3, SFT_RMA_TRAIN_GEMMA
#from train_utils import preprocess_example_history, preprocess_example_rewrite
import pdb
import json
import ast
import copy
import os
import glob
import pandas as pd
from pathlib import Path

DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/hj153lee/rma_train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an RMA LoRA adapter.")
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
    return parser.parse_args()


args = parse_args()
model_name = args.model_name

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "datasets"
TRAIN_DIR = DATASET_DIR / "train"
TC_DIR = DATASET_DIR / "tc"

model_dirname = model_name.replace("/", "__")
output_dir = Path(args.output_root).expanduser() / model_dirname
cache_dir = output_dir / "hf_cache"

output_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"model_name: {model_name}")
print(f"dataset_dir: {DATASET_DIR}")
print(f"hf_cache_dir: {cache_dir}")
print(f"training_output_dir: {output_dir}")

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=str(cache_dir),
)
# 만약 pad_token_id가 없으면 eos 토큰으로 설정
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if 'google/gemma' in model_name:
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        cache_dir=str(cache_dir),
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

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

max_length = 1536

def preprocess_example_it(example):
    data = {"conversation_history": example["conversation_history"], "query": example["query"]}
    prompt = prompt_template.format(                
        data=json.dumps(data, ensure_ascii=False, indent=2),
        answer={"rewrited_query": example["rewrited_query"]}
    )
    
    # 전체 prompt+answer를 토크나이즈합니다.
    tokenized = tokenizer(
        prompt,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True
    )
    input_ids = tokenized["input_ids"]

    # model이 정답 토큰을 시작하는 위치를 찾아, 
    # 사용자 입력 부분은 label 값으로 -100을 할당하고,
    # 모델 생성 부분은 실제 토큰 id를 할당합니다.
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        model_start = prompt.find("assistant<|end_header_id|>")
    elif model_name == "google/gemma-3-4b-it":
        model_start = prompt.find("<start_of_turn>model")
    elif model_name == "microsoft/Phi-4-mini-instruct":
        model_start = prompt.find("<|end|><|assistant|>")
    elif "Qwen/" in model_name:
        model_start = prompt.find("<|im_end|><|im_start|>assistant")
        
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
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }

if model_name == "meta-llama/Llama-3.2-3B-Instruct":
    prompt_template = SFT_RMA_TRAIN_LLAMA
elif model_name == "google/gemma-3-4b-it":
    prompt_template = SFT_RMA_TRAIN_GEMMA
elif model_name == "microsoft/Phi-4-mini-instruct":
    prompt_template = SFT_RMA_TRAIN_PHI4
elif "Qwen/" in model_name:
    prompt_template = SFT_RMA_TRAIN_QWEN3

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

if 'Phi-4' in model_name or 'Qwen' in model_name:
    lora_config = LoraConfig(
        task_type       = TaskType.CAUSAL_LM,
        r               = 16,
        lora_alpha      = 32,
        lora_dropout    = 0.05,
        bias            = "none",
        target_modules  = "all-linear",
    )
elif 'Llama' in model_name:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=[
        "q_proj", "k_proj", "v_proj",
        "gate_proj", "up_proj", "down_proj", "o_proj"
        ], # 0.38
        lora_dropout=0.1,
        bias="none"
    )
elif 'gemma' in model_name:
        lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none"
)
else:
    # LoRA 설정 (일반적으로 LLaMA 모델에서도 "q_proj", "v_proj" 모듈을 target으로 사용합니다)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )
# LoRA 적용 모델 생성
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 비율 출력 (옵션)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"학습 가능 파라미터: {trainable_params}, 전체 파라미터: {total_params}, 학습 비율: {100 * trainable_params / total_params:.2f}%")

training_args = TrainingArguments(
    output_dir=str(output_dir),
    overwrite_output_dir=True,
    num_train_epochs=6,
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

# 학습 완료된 LoRA 어댑터 모델 저장
trainer.save_model(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
