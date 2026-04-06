# -*- coding: utf-8 -*-

import os, json, ast, glob
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)
from llama_prompts import SFT_RMA_TRAIN_PHI4
from llama_prompts import SFT_RMA_TRAIN_QWEN3

#MODEL_NAME = "microsoft/Phi-4-mini-instruct"        
MODEL_NAME = "Qwen/Qwen3-4B"
TOOLS_PATH = "../apis/simple_api.json"              
TRAIN_DIR  = "../datasets/train/"                   
TRAIN_TYPE = "rewrite"                              
PREFIX     = "rma-rewrite-integrated-half"                                  

# ─── Model & tokenizer ────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.pad_token = tokenizer.unk_token

tokenizer.pad_token_id     = tokenizer.eos_token_id

tokenizer.padding_side  = "right"
tokenizer.model_max_length = 1536                 # keep small for LoRA; raise if needed

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    trust_remote_code=True,
    device_map="auto"
)
model.gradient_checkpointing_enable()

# 2. Load tool specs (used inside <|tool|> … <|/tool|>)
def read_simple_apis(api_file):
    with open(api_file, "r", encoding="utf-8") as f:
        api_data = json.load(f)
    return api_data

# 3. Dataset loading
train_files = [
    os.path.join(TRAIN_DIR, f)
    for f in os.listdir(TRAIN_DIR)
    if ("nonNR" in f and f.endswith(".tsv"))
]
train_files.append(os.path.join(TRAIN_DIR, "it2_NR_train.tsv"))

raw_ds = load_dataset(
    "csv",
    data_files={"train": train_files},
    delimiter="\t",
)

MAX_LEN = 1536     # truncate after chat-template is applied
apis = read_simple_apis("../apis/simple_api.json")

def planning_preprocess_example(example):
    """
    Converts a TSV row to the chat-template string expected by Phi-4:
      - system message holds the tool JSON block
      - user content is either conversation-history + query   (history mode)
        or rewrited_query                                   (rewrite mode)
      - assistant content is the gold-reference answer / JSON
    Produces: input_ids, labels, plus pretty-printed strings for debugging.
    """
    
    api_str = ""    
    candidates = ast.literal_eval(example['candidates'])
    for plan in candidates:        
        api_data = apis[plan].copy()
        api_str += f"{plan}: {api_data}\n"        

    if TRAIN_TYPE == "history":
        system_msg = f"You are a helpful assistant capable of selecting appropriate tools based on user queries and generating corresponding parameters. Use information from the conversation history when relevant. Only use parameter values that are explicitly stated or can be reasonably inferred from the query. If no tool matches the query, set the tool to 'None'.\n <|tool|>{api_str}<|/tool|>"
        
        user_content = (            
            f"Conversation History: {example['conversation_history']}\n"
            f"User Query: {example['query']}"
        )
    elif TRAIN_TYPE == "rewrite":  # rewrite
        system_msg = f"Given a user query and a list of available tools, select the most appropriate tool and generate the corresponding parameters. If no tool matches the query, set the tool to 'None'. Only use parameter values that are explicitly stated or can be reasonably inferred from the query.\n <|tool|>{api_str}<|/tool|>"
        
        user_content = (            
            f"User Query: {example['rewrited_query']}"
        )

    assistant_content = (
        json.dumps(example["answer"], ensure_ascii=False)
        if isinstance(example["answer"], dict)
        else example["answer"]
    )

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    prompt_with_answer = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,   
    )

    tok = tokenizer(
        prompt_with_answer,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
    )
    input_ids = tok["input_ids"]

    assistant_start = prompt_with_answer.rfind("<|assistant|>") + len("<|assistant|>")
    label_start = len(
        tokenizer(prompt_with_answer[:assistant_start],
                  add_special_tokens=False)["input_ids"]
    )

    labels = [-100] * label_start + input_ids[label_start:]

    return {
        "input_ids": input_ids,
        "labels":    labels,
        "strprompt": prompt_with_answer,
        "stranswer": assistant_content,
    }

# ==== 

rma_files = [file for file in glob.glob("../datasets/train/*.tsv", recursive=True) if "_NR_" not in file and file.endswith(".tsv")]
#added_files = [file for file in glob.glob("../datasets/tc/*.tsv", recursive=True) if "_NR_" not in file and file.endswith(".tsv")]
added_files = [file for file in glob.glob("../datasets/tc/*.tsv", recursive=True) if ("complex" in file or "various_nonNR" in file) and file.endswith(".tsv")]
rma_files.append("../datasets/train/it2_NR_train.tsv")
rma_files.extend(added_files)

dfs = [pd.read_csv(f, sep="\t", dtype=str) for f in rma_files]
df_all = pd.concat(dfs, ignore_index=True, sort=False)
rma_raw_datasets = Dataset.from_pandas(df_all)

max_length = 1536

def preprocess_example_rma(example):
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
    if MODEL_NAME == "meta-llama/Llama-3.2-3B-Instruct":
        model_start = prompt.find("assistant<|end_header_id|>")
    elif MODEL_NAME == "google/gemma-3-4b-it":
        model_start = prompt.find("<start_of_turn>model")
    elif MODEL_NAME == "microsoft/Phi-4-mini-instruct":
        model_start = prompt.find("<|end|><|assistant|>")
    elif "Qwen" in MODEL_NAME:
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

if MODEL_NAME == "microsoft/Phi-4-mini-instruct":
    prompt_template = SFT_RMA_TRAIN_PHI4
elif "Qwen/" in MODEL_NAME:
    prompt_template = SFT_RMA_TRAIN_QWEN3

planning_processed_train = raw_ds["train"].map(
    planning_preprocess_example,
    desc="Applying Phi-4 chat template"
)

rma_processed_train = rma_raw_datasets.map(
    preprocess_example_rma,     
)

print(rma_processed_train[0]["strprompt"])
print()
print(planning_processed_train[0]["strprompt"])

processed_train = concatenate_datasets([planning_processed_train, rma_processed_train])
processed_train = processed_train.shuffle(seed=42)
# ====

max_total_length = 0
for example in rma_processed_train:
    total_length = len(example["input_ids"]) + len(example["labels"])
    if total_length > max_total_length:
        max_total_length = total_length
print(f"input_ids + labels 최대 길이: {max_total_length}")
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        label_pad = -100
        max_len   = max(len(f["labels"]) for f in features)
        for f in features:
            f["labels"] += [label_pad] * (max_len - len(f["labels"]))
        return self.tokenizer.pad(features, padding=True, return_tensors="pt")

data_collator = DataCollatorForCausalLM(tokenizer)

lora_cfg = LoraConfig(
    task_type       = TaskType.CAUSAL_LM,
    r               = 16,
    lora_alpha      = 32,
    lora_dropout    = 0.05,
    bias            = "none",
    target_modules  = "all-linear",   
)

model = get_peft_model(model, lora_cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"LoRA params: {trainable/1e6:.1f} M  /  Total: {total/1e6:.1f} M")

output_dir = f"{MODEL_NAME.split('/')[-1]}-{TRAIN_TYPE}-{PREFIX}"

training_args = TrainingArguments(
    output_dir              = output_dir,
    overwrite_output_dir    = True,
    num_train_epochs        = 6,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate           = 5e-5,
    bf16                    = True,
    gradient_checkpointing  = True,
    max_grad_norm           = 1.0,
    logging_steps           = 50,
    logging_dir             = f"{output_dir}/logs",
    save_strategy           = "epoch",
    save_total_limit        = 8,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = processed_train,
    data_collator   = data_collator,
    tokenizer       = tokenizer,
)

trainer.train()

trainer.save_model()
tokenizer.save_pretrained(output_dir)
