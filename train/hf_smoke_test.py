#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for a merged HF model.")
    parser.add_argument("--model-dir", required=True, help="Merged HF model directory")
    parser.add_argument(
        "--prompt",
        default="한국어로 한 문장 자기소개를 해줘.",
        help="Prompt used for generation",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=96, help="Maximum generated tokens"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for the generation result",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    result = {
        "model_dir": str(model_dir),
        "prompt": args.prompt,
        "response": response,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
