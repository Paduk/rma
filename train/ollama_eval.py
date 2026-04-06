#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an Ollama model with prompts.")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument(
        "--prompts",
        default="/home/hj153lee/RMA/train/ollama_eval_prompts.jsonl",
        help="JSONL prompt file",
    )
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama host URL"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for responses",
    )
    return parser.parse_args()


def load_prompts(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def generate(host: str, model: str, prompt: str):
    response = requests.post(
        f"{host}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 256,
            },
        },
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama generate failed ({response.status_code}): {response.text}"
        )
    payload = response.json()
    return payload.get("response", "").strip()


def main():
    args = parse_args()
    prompt_path = Path(args.prompts).expanduser().resolve()
    prompts = load_prompts(prompt_path)

    results = []
    for item in prompts:
        output = generate(args.host, args.model, item["prompt"])
        result = {
            "id": item["id"],
            "prompt": item["prompt"],
            "response": output,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
