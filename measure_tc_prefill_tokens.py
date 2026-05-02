import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from ollama_inference_baseline import render_baseline_inference_prompt
from ollama_inference_oneshot_baseline import build_oneshot_prompt


QWEN3_TOKENIZER_NAME = "Qwen/Qwen3-4B"
DEFAULT_TC_PATH = Path("/home/hj153lee/ondevice-llm-inference/app/src/main/assets/tc.tsv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build prompts for tc.tsv using the prompt logic from "
            "ollama_inference_oneshot_baseline.py and ollama_inference_baseline.py, "
            "then measure prompt token counts with the Qwen3 tokenizer."
        )
    )
    parser.add_argument(
        "--tc-path",
        type=Path,
        default=DEFAULT_TC_PATH,
        help="Path to tc.tsv",
    )
    parser.add_argument(
        "--tokenizer",
        default=QWEN3_TOKENIZER_NAME,
        help="HF tokenizer name used to render prompts",
    )
    parser.add_argument(
        "--baseline-prompt-mode",
        choices=["history", "rewrite"],
        default="history",
        help="Prompt mode passed to ollama_inference_baseline.py",
    )
    parser.add_argument(
        "--chat-template-fallback",
        choices=["simple", "error"],
        default="simple",
        help="Fallback used when tokenizer has no chat_template",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick checks",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional per-row TSV output path",
    )
    return parser.parse_args()


def build_oneshot_baseline_prompt(example: dict[str, Any], tokenizer, chat_template_fallback: str) -> str:
    return build_oneshot_prompt(
        example=example,
        prompt_tokenizer=tokenizer,
        prompt_model_name=QWEN3_TOKENIZER_NAME,
        chat_template_fallback=chat_template_fallback,
    )


def build_baseline_prompt(
    example: dict[str, Any],
    tokenizer,
    prompt_mode: str,
    chat_template_fallback: str,
) -> str:
    config = {
        "prompt_mode": prompt_mode,
        "chat_template_fallback": chat_template_fallback,
    }
    return render_baseline_inference_prompt(
        example=example,
        tokenizer=tokenizer,
        config=config,
    )


def count_tokens(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def parse_answer_dict(answer_value: Any) -> dict[str, Any]:
    if isinstance(answer_value, dict):
        return answer_value
    if not isinstance(answer_value, str):
        raise ValueError(f"Unsupported answer type: {type(answer_value).__name__}")

    try:
        parsed = ast.literal_eval(answer_value)
    except Exception:
        parsed = json.loads(answer_value)

    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed answer is not a dict: {type(parsed).__name__}")
    return parsed


def build_generation_text(example: dict[str, Any], prompt_mode: str) -> str:
    if prompt_mode == "oneshot_baseline":
        payload = parse_answer_dict(example["answer"]).copy()
        payload["rewrited_query"] = example.get("rewrited_query", "")
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

    if prompt_mode == "baseline_history":
        return str(example["answer"])

    if prompt_mode == "baseline_rewrite":
        return str(example["answer"])

    raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")


def build_example(row: pd.Series) -> dict[str, Any]:
    example = row.to_dict()
    if pd.isna(example.get("rewrited_query")):
        example["rewrited_query"] = ""
    return example


def summarize(df: pd.DataFrame, mode: str):
    sub = df[df["prompt_mode"] == mode].copy()
    avg_prompt_tokens = sub["prompt_tokens"].mean()
    avg_generation_tokens = sub["generation_tokens"].mean()
    min_prompt_tokens = sub["prompt_tokens"].min()
    max_prompt_tokens = sub["prompt_tokens"].max()

    print(f"\n[{mode}]")
    print(f"rows: {len(sub)}")
    print(f"average prompt tokens: {avg_prompt_tokens:.4f}")
    print(f"average generation tokens: {avg_generation_tokens:.4f}")
    print(f"min prompt tokens: {min_prompt_tokens}")
    print(f"max prompt tokens: {max_prompt_tokens}")


def main():
    args = parse_args()

    df = pd.read_csv(args.tc_path, sep="\t")
    if args.limit is not None:
        df = df.head(args.limit).copy()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        example = build_example(row)
        prompt_builders = [
            (
                "oneshot_baseline",
                lambda: build_oneshot_baseline_prompt(
                    example=example,
                    tokenizer=tokenizer,
                    chat_template_fallback=args.chat_template_fallback,
                ),
            ),
            (
                f"baseline_{args.baseline_prompt_mode}",
                lambda: build_baseline_prompt(
                    example=example,
                    tokenizer=tokenizer,
                    prompt_mode=args.baseline_prompt_mode,
                    chat_template_fallback=args.chat_template_fallback,
                ),
            ),
        ]

        for prompt_mode, prompt_builder in prompt_builders:
            prompt = prompt_builder()
            prompt_tokens = count_tokens(tokenizer, prompt)
            generation_text = build_generation_text(example, prompt_mode)
            generation_tokens = count_tokens(tokenizer, generation_text)
            rows.append(
                {
                    "prompt_mode": prompt_mode,
                    "unique_idx": example.get("unique_idx"),
                    "query": example.get("query"),
                    "rewrited_query": example.get("rewrited_query"),
                    "prompt_tokens": prompt_tokens,
                    "generation_tokens": generation_tokens,
                }
            )

    result_df = pd.DataFrame(rows)
    summarize(result_df, "oneshot_baseline")
    summarize(result_df, f"baseline_{args.baseline_prompt_mode}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output, sep="\t", index=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
