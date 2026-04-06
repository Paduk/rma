# Android Test Checklist

Use the same quantized GGUF that passed desktop Ollama validation.

## Artifact

- GGUF: `/home/hj153lee/qwen3-1.7b-artifacts/qwen3-1.7b-rma-q4_k_m.gguf`

## Prompt Set

- Reuse `/home/hj153lee/RMA/train/ollama_eval_prompts.jsonl`

## What to Record

- Model load success or failure
- First token latency
- Total response time
- Device temperature / throttling signs
- Battery drain during a short test run
- Output quality differences versus desktop

## Decision Rule

- Keep `Q4_K_M` if the model loads reliably and the response quality is acceptable.
- If memory or speed is insufficient, generate a smaller quantization from the full GGUF.
- If quality drops too much, generate a higher-quality quantization from the full GGUF.
