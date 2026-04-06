# -*- coding: utf-8 -*-
"""
Fine-tune microsoft/Phi-4-mini-instruct with LoRA (history / rewrite)
--------------------------------------------------------------------
• Keeps your dataset logic intact – just switch `train_type` = "history" or "rewrite".
• Uses Phi-4's chat-and-tool template:
    <|system|><|tool|>{JSON tools}<|/tool|><|end|>
    <|user|>{user text}<|end|>
    <|assistant|>{assistant text}
• Relies on tokenizer.apply_chat_template() so that every special-token edge-case
  (padding-side, EOS spacing, etc.) matches the official Microsoft recipe.
"""


# 1. Imports & model / tokenizer loading

import os, json, ast
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
)
# /mnt/data/singularity-images/pytorch_22.09-py3-cp.sif 
MODEL_NAME = "microsoft/Phi-4-mini-instruct"        
TOOLS_PATH = "../apis/simple_api.json"              
TRAIN_DIR  = "../datasets/train/"                   
TRAIN_TYPE = "history"                              
PREFIX     = "1st"                                  

# ─── Model & tokenizer ────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
)
tokenizer.pad_token     = tokenizer.unk_token      # avoid endless generation
tokenizer.pad_token_id  = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side  = "right"
tokenizer.model_max_length = 1024                  # keep small for LoRA; raise if needed

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
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

MAX_LEN = 1024     # truncate after chat-template is applied
apis = read_simple_apis("../apis/simple_api.json")

def preprocess_example(example):
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

    # -----  assistant gold -------------------------------------------------
    assistant_content = (
        json.dumps(example["answer"], ensure_ascii=False)
        if isinstance(example["answer"], dict)
        else example["answer"]
    )

    # Build list-of-messages for tokenizer.apply_chat_template()
    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # ► 1. Render to raw text with special tokens
    prompt_with_answer = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,   # we already include assistant gold
    )

    # ► 2. Tokenise once to compute input_ids + labels
    tok = tokenizer(
        prompt_with_answer,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LEN,
    )
    input_ids = tok["input_ids"]

    # The assistant segment starts right after the last "<|assistant|>" token
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

processed_train = raw_ds["train"].map(
    preprocess_example,
    desc="Applying Phi-4 chat template"
)

# Simple sanity-check
print(train_files)
print()
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

# lora_cfg = LoraConfig(
#         task_type=TaskType.CAUSAL_LM,
#         r=8,
#         lora_alpha=16,
#         target_modules=["qkv_proj","o_proj"],
#         lora_dropout=0.1,
#         bias="none"
# )

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
