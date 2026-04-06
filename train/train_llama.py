# 필수 라이브러리 임포트
import torch
from datasets import load_dataset
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
from llama_prompts import SFT_REWRITE_TRAIN_LLAMA, SFT_HISTORY_TRAIN_LLAMA
from gemma_prompts import SFT_REWRITE_TRAIN_GEMMA, SFT_HISTORY_TRAIN_GEMMA
#from train_utils import preprocess_example_history, preprocess_example_rewrite
import pdb
import json
import ast
import copy
import os

model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name = "google/gemma-3-4b-it"

# tokenizer 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 만약 pad_token_id가 없으면 eos 토큰으로 설정
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 메모리 효율성을 위해 gradient_checkpointing 활성화
model.gradient_checkpointing_enable()

def read_apis(api_file):
    api_dict = {}
    with open(api_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                api_data = json.loads(line)
                api_data.pop("examples", None)
                api_data.pop("returns", None)
                api_data.pop("next_turn_plans", None)
                api_dict[api_data["plan"]] = api_data
    
    return api_dict

def read_simple_apis(api_file):
    with open(api_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    return api_data

#apis = read_apis("../apis/api_v3.0.1.jsonl")
apis = read_simple_apis("../apis/simple_api.json")

# 데이터 로딩 및 전처리

train_path = '../datasets/train/'
train_tsv_files = [os.path.join(train_path, f) for f in os.listdir(train_path) if 'nonNR' in f and f.endswith('.tsv')]
train_tsv_files.append(os.path.join(train_path, 'it2_NR_train.tsv'))
print(train_tsv_files)
data_files = {'train': train_tsv_files} 
raw_datasets = load_dataset('csv', data_files=data_files, delimiter='\t')

max_length = 1536
train_type = 'history'  # 'history' 또는 'rewrite'로 설정
prefix = "all_linear"

output_dir = f"{model_name.split('/')[0]}-{train_type}-{prefix}"
print(train_type, output_dir)

def preprocess_example_history(example):    
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:
        # apis 사전에서 해당 plan에 대응하는 데이터를 복사하여 문자열로 변환합니다.
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
        #api_str += json.dumps(api_data, indent=2, ensure_ascii=False) + "\n"
        
    prompt = prompt_template.format(
        tools=api_str,
        conversation_history=example["conversation_history"],
        data=example["query"],
        answer=example["answer"]
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
        # answer가 딕셔너리인 경우 보기 좋게 json 문자열로 변환합니다.
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }

def preprocess_example_rewrite(example):    
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:
        # apis 사전에서 해당 plan에 대응하는 데이터를 복사하여 문자열로 변환합니다.
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"
        #api_str += json.dumps(api_data, indent=2, ensure_ascii=False) + "\n"
        
    prompt = prompt_template.format(
        tools=api_str,
        data=example["rewrited_query"],
        answer=example["answer"]
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
        # answer가 딕셔너리인 경우 보기 좋게 json 문자열로 변환합니다.
        "stranswer": json.dumps(example["answer"], indent=2, ensure_ascii=False)
                      if isinstance(example["answer"], dict) else example["answer"],
    }

if model_name == "meta-llama/Llama-3.2-3B-Instruct":
    if train_type == 'history': 
        prompt_template = SFT_HISTORY_TRAIN_LLAMA
        processed_train = raw_datasets["train"].map(
            preprocess_example_history,     
        )
    else:
        prompt_template = SFT_REWRITE_TRAIN_LLAMA
        processed_train = raw_datasets["train"].map(
            preprocess_example_rewrite,     
        )
elif model_name == "google/gemma-3-4b-it":    
    if train_type == 'history': 
        prompt_template = SFT_HISTORY_TRAIN_GEMMA
        processed_train = raw_datasets["train"].map(
            preprocess_example_history,     
        )
    else:
        prompt_template = SFT_REWRITE_TRAIN_GEMMA
        processed_train = raw_datasets["train"].map(
            preprocess_example_rewrite,     
        )

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

# lora_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     r=8,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj"],
#     lora_dropout=0.1,
#     bias="none"
# )
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

# LoRA 적용 모델 생성
model = get_peft_model(model, lora_config)

# 학습 가능한 파라미터 비율 출력 (옵션)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"학습 가능 파라미터: {trainable_params}, 전체 파라미터: {total_params}, 학습 비율: {100 * trainable_params / total_params:.2f}%")


training_args = TrainingArguments(
    output_dir=output_dir,
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
    logging_dir=f"{output_dir}/logs",
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
trainer.save_model()
tokenizer.save_pretrained(output_dir)