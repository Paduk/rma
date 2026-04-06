# RMA Paper153 Code Repository

This repository contains code for dataset generation, model training, inference, and evaluation for the RMA project. The most operationally important part of the repository is the multi-turn dataset generation pipeline built around `o4-mini`.

## Repository Structure

- [`o4_pipeline.sh`](/home/hj153lee/RMA/o4_pipeline.sh): original non-complex multi-turn data generation pipeline.
- [`o4_complex_pipeline.sh`](/home/hj153lee/RMA/o4_complex_pipeline.sh): original complex data generation pipeline.
- [`o4_pipeline_refactored.sh`](/home/hj153lee/RMA/o4_pipeline_refactored.sh): function-based refactor of the original non-complex pipeline.
- [`o4_complex_pipeline_refactored.sh`](/home/hj153lee/RMA/o4_complex_pipeline_refactored.sh): function-based refactor of the original complex pipeline.
- [`datasets/`](/home/hj153lee/RMA/datasets): TSV datasets used for training and evaluation.
- [`train/`](/home/hj153lee/RMA/train): training scripts and model merge utilities.
- [`ollama_inference_multi.py`](/home/hj153lee/RMA/ollama_inference_multi.py): multi-file inference and evaluation script for Ollama-based local models.
- [`cloudllm_inference.py`](/home/hj153lee/RMA/cloudllm_inference.py): inference for closed-source/cloud models.
- [`evaluation_analysis.py`](/home/hj153lee/RMA/evaluation_analysis.py): result analysis utilities.

## Data Generation Overview

The dataset is generated turn by turn. A turn is represented by file names such as `it1_*`, `it2_*`, `it3_*`, and so on.

Important naming conventions:

- `itN_s1.jsonl`: integrated turn-`N` samples before final filtering.
- `itN_s1_filtered.jsonl`: filtered turn-`N` samples. This is the file typically used as input for the next turn.
- `itN_s1_nonnr.jsonl`: integrated turn-`N` samples before balancing/filtering.
- `itN_s2.jsonl`: device responses generated from the previous turn input.
- `itN_s2_idxed.jsonl`: `s2` output reattached to original metadata and `unique_idx`.
- `itN_s3.jsonl`: next-turn user queries.
- `itN_s3_idxed.jsonl`: indexed `s3` output.
- `itN_s3_idxed_dedup.jsonl`: deduplicated `s3` data.
- `itN_s3_idxed_rewrite.jsonl`: rewritten query form used for final answer generation.
- `itN_s4.jsonl`: generated final answers for the next turn.

High-level non-complex flow:

1. Generate or load the previous turn `s1`.
2. Generate `s2` device responses.
3. Generate `s3` next-turn user requests.
4. Deduplicate and rewrite `s3`.
5. Generate `s4` answers.
6. Integrate into `it(N+1)_s1_nonnr.jsonl`.
7. Balance and filter into `it(N+1)_s1.jsonl` and `it(N+1)_s1_filtered.jsonl`.

The practical final output for each turn is usually `itN_s1_filtered.jsonl`, because that is the file consumed by the next iteration.

## Non-Complex Pipeline

The original script is [`o4_pipeline.sh`](/home/hj153lee/RMA/o4_pipeline.sh). It is a linear shell script with turn blocks connected by `&&`.

The refactored version is [`o4_pipeline_refactored.sh`](/home/hj153lee/RMA/o4_pipeline_refactored.sh). It keeps the same pipeline semantics but organizes them into functions:

- `run_turn_1`
- `run_turn_2`
- `run_turn_3`
- `run_turn_4`
- `run_turn_5`

Usage:

Run one turn only:

```bash
TURN=5 bash o4_pipeline_refactored.sh
```

Run all turns sequentially:

```bash
bash o4_pipeline_refactored.sh
```

Run a subset with explicit toggles:

```bash
RUN_TURN_2=1 RUN_TURN_3=1 bash o4_pipeline_refactored.sh
```

Default settings:

- `MODEL=o4-mini`
- `DATADIR=difficult_o4`
- `API_FILE=apis/api_v3.0.1.jsonl`

Example turn transition:

- Turn 1 produces `it1_s1.jsonl`
- Turn 2 consumes `it1_s1.jsonl` and produces `it2_s1_nonnr.jsonl`, `it2_s1.jsonl`, `it2_s1_filtered.jsonl`
- Turn 3 consumes `it2_s1_filtered.jsonl`
- Turn 4 consumes `it3_s1_filtered.jsonl`
- Turn 5 consumes `it4_s1_filtered.jsonl`

## Complex Pipeline

The original script is [`o4_complex_pipeline.sh`](/home/hj153lee/RMA/o4_complex_pipeline.sh). Complex data is different from standard multi-turn data because it refers back to turns older than the immediately previous turn.

Key concept:

- standard non-complex data: uses the most recent conversation context
- complex data: refers to a previous turn from two or more turns ago

This is tracked by `refered_turn`.

Examples:

- `it5_complex_1_tc.tsv`: turn 5 complex data referring back to turn-1-level context
- `it5_complex_2_tc.tsv`: turn 5 complex data referring back to turn-2-level context
- `it5_complex_3_tc.tsv`: turn 5 complex data referring back to turn-3-level context

The refactored version is [`o4_complex_pipeline_refactored.sh`](/home/hj153lee/RMA/o4_complex_pipeline_refactored.sh).

Usage:

Run one complex turn only:

```bash
TURN=4 bash o4_complex_pipeline_refactored.sh
```

Run all complex turns sequentially:

```bash
bash o4_complex_pipeline_refactored.sh
```

Run a subset with explicit toggles:

```bash
RUN_TURN_3_COMPLEX=1 RUN_TURN_5_COMPLEX=1 bash o4_complex_pipeline_refactored.sh
```

Complex turn mapping:

- Complex Turn 3 uses `it3_s1_filtered.jsonl` and produces `it3_s1_complex.jsonl`
- Complex Turn 4 uses `it4_s1_filtered.jsonl` and produces `it4_s1_complex.jsonl`
- Complex Turn 5 uses `it5_s1_filtered.jsonl` and produces `it5_s1_complex.jsonl`

## JSONL to TSV Conversion

Final evaluation files live under [`datasets/tc/`](/home/hj153lee/RMA/datasets/tc).

## Training

Training scripts live under [`train/`](/home/hj153lee/RMA/train).

Planner training entry points:

- [`train/train_llama.py`](/home/hj153lee/RMA/train/train_llama.py)
- [`train/train_qwen.py`](/home/hj153lee/RMA/train/train_qwen.py)
- [`train/train_phi.py`](/home/hj153lee/RMA/train/train_phi.py)
- [`train/train_gemma.py`](/home/hj153lee/RMA/train/train_gemma.py)

RMA-oriented training entry points:

- [`train/train_rma_llama.py`](/home/hj153lee/RMA/train/train_rma_llama.py)
- [`train/train_integrated_phi.py`](/home/hj153lee/RMA/train/train_integrated_phi.py)

Typical planner training example:

```bash
cd train
python3 train_llama.py
```

Operational notes:

- several training scripts contain model names and dataset paths directly in the file
- for example, [`train/train_llama.py`](/home/hj153lee/RMA/train/train_llama.py) currently loads TSVs from `../datasets/train/`
- the training scripts distinguish between `history` and `rewrite` style prompting
- LoRA-based fine-tuning is used in the planner training scripts

## Model Merge

Merge adapter weights with a pretrained base model using [`train/model_merge.py`](/home/hj153lee/RMA/train/model_merge.py).

Example:

```bash
cd train
python3 model_merge.py --model qwen3 --t1 /path/to/lora/checkpoint --t2 /path/to/merged_model
```

Operational notes:

- `--model` selects a hardcoded base-model path inside the script
- `--t1` is the LoRA checkpoint path
- `--t2` is the output directory for the merged model
- training and inference may use different environments after merge

## Inference

### RMA / Base Inference

Available local inference entry points include:

- [`ollama_inference.py`](/home/hj153lee/RMA/ollama_inference.py)
- [`base_ollama_inference.py`](/home/hj153lee/RMA/base_ollama_inference.py)

### Closed-Source / Cloud Inference

```bash
python3 cloudllm_inference.py
```

There is also a batch-oriented variant:

```bash
python3 cloudllm_inference_batch.py
```

### Ollama Inference

Single-script local inference:

```bash
python3 ollama_inference.py
```

Multi-file evaluation:

```bash
python3 ollama_inference_multi.py \
  --t base-qwen3 \
  --test_key base,complex \
  --o datasets/result/260330/scale-qwen3-4b-base.tsv
```

`ollama_inference_multi.py` computes per-file and turn-wise accuracy summaries, including plan-macro accuracy.

Current built-in `test_key` groups in [`ollama_inference_multi.py`](/home/hj153lee/RMA/ollama_inference_multi.py) include:

- `base`
- `complex`
- `swap`
- `manual_rewrited`

The script currently points `base` and `complex` to files under [`datasets/tc/capped_complex_plan5/`](/home/hj153lee/RMA/datasets/tc/capped_complex_plan5).

`--t` selects the prompt/model configuration, for example:

- `base-qwen3`
- `base-qwen3-1.7b`
- `base-llama3`
- `base-phi4`
- rewrite variants for several backbones

## Evaluation Analysis

Run the result analysis script:

```bash
python3 evaluation_analysis.py
```

Use this after inference to summarize TSV outputs and inspect plan-level performance.

## Known Practical Caveats

- many scripts assume specific local filesystem layouts and hardcoded model paths
- some legacy filtering paths default to Gemini unless explicitly overridden
- before long runs, verify the active output directory and whether the script writes intermediate JSONL, filtered JSONL, or TSV outputs

## Environment Notes

- Ensure API credentials are configured for the generation scripts you use.
- Some scripts assume local relative paths such as `apis/api_v3.0.1.jsonl` and `logs/`.
- The repository mixes OpenAI-style generation, Gemini-based filtering in some legacy paths, and local Ollama inference. Confirm which backend each script uses before long runs.

## License

This repository is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. You may share and adapt the material with appropriate attribution.
