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

Training scripts live under [`train/`](train/). The main entry points used in the recent experiments are the four scripts below.

At a high level, the four training workflows are organized as follows:

- `train/train_sentence_rewriter.py`: trains a rewrite-only adapter that maps `(conversation_history, query)` to `rewrited_query`
- `train/train_legacy_integrated.py`: trains a planner-only adapter that predicts `plan` and `arguments`
- `train/train_multitask_rma.py`: trains a shared adapter on both rewrite and planning objectives in a multitask setup
- `train/train_oneshot_rma_qwen.py`: trains a Qwen one-shot adapter that emits `rewrited_query`, `plan`, and `arguments` in one generation

### `train/train_sentence_rewriter.py`

This script trains a LoRA adapter that rewrites a raw `(conversation_history, query)` pair into a normalized `rewrited_query`. That rewritten query is then used by the later planning and inference stages.

It reads rewrite-style training data from:

- `datasets/train/*.tsv`
- `datasets/train/additional/*.tsv`

Dataset selection is opinionated:

- base files containing `_NR_` are excluded from the main rewrite set
- additional files containing `complex` or `various_nonNR` are included
- `datasets/train/it2_NR_train.tsv` is explicitly added as an extra file

Example:

```bash
python3 train/train_sentence_rewriter.py \
  --model_name "Qwen/Qwen2.5-3B-Instruct"
```

Operational notes:

- prompt template and LoRA target modules are selected automatically from `--model_name`
- by default the script also postprocesses the trained adapter by merging weights, exporting GGUF, and registering an Ollama model
- if you only want checkpoints and adapter weights, add `--skip_postprocess`
- on a new machine, you will usually want to override the default `--output_root`

### `train/train_legacy_integrated.py`

This is the unified planner-training entry point that preserves the behavior of the older `train_qwen.py`, `train_phi.py`, and `train_llama.py` scripts. It trains a model to predict tool `plan` and `arguments` from either conversation history or a rewritten query.

It reads planning data from:

- `datasets/train/*.tsv`
- `datasets/train/additional/*.tsv`

Example:

```bash
python3 train/train_legacy_integrated.py --profile qwen
```

Key options:

- `--profile {qwen,phi,llama}` selects the backbone family and legacy defaults
- `--train_type history` is the default
- `--train_type rewrite` switches the planner to use `rewrited_query` instead of conversation history
- `--skip_postprocess` keeps the run at checkpoint/adapter stage without GGUF or Ollama export

### `train/train_multitask_rma.py`

This script trains a single multitask LoRA adapter that mixes two objectives in one run:

- rewrite training: generate `rewrited_query` from `(conversation_history, query)`
- planning training: generate the final tool `plan` and `arguments` from `rewrited_query`

The rewrite side reuses the same file-selection rule as `train_sentence_rewriter.py`. The planning side loads TSVs whose names contain `nonNR` from `datasets/train/`, and also includes `datasets/train/it2_NR_train.tsv` when it exists.

Example:

```bash
python3 train/train_multitask_rma.py \
  --model_name "Qwen/Qwen3-4B"
```

Key options:

- `--mix_strategy balanced` interleaves rewrite and planning samples
- `--mix_strategy concat` concatenates the two processed datasets
- `--rewrite_sampling_prob` controls the rewrite/planning ratio when using balanced interleaving
- `--skip_postprocess` disables merge, GGUF export, and Ollama registration

### `train/train_oneshot_rma_qwen.py`

This script trains a Qwen-based one-shot RMA adapter that predicts the full structured target in one generation:

- `rewrited_query`
- `plan`
- `arguments`

Unlike the multitask trainer, this script formats each example as a single one-shot chat prompt and learns to emit the full JSON target directly.

It reads training data from:

- `datasets/train/*.tsv` files whose names contain `nonNR`
- `datasets/train/it2_NR_train.tsv` when present
- `datasets/train/additional/*.tsv`

Example:

```bash
python3 train/train_oneshot_rma_qwen.py \
  --model_name "Qwen/Qwen3-4B"
```

Key options:

- this script currently supports only Qwen-family backbones
- `--max_length` controls prompt truncation length
- `--skip_postprocess` leaves the run at checkpoint/adapter stage
- postprocessing defaults follow the same merge, GGUF export, and Ollama registration flow used by the other RMA training scripts

Historical single-purpose training scripts are still present under [`train/`](train/), but the four entry points above are the main operational scripts used in this repository.

## Inference Workflows

The main local inference and evaluation entry points used in recent experiments are the three scripts below.

### `ollama_inference_oneshot.py`

This script evaluates a one-shot model that predicts the full structured output in a single generation:

- `rewrited_query`
- `plan`
- `arguments`

It renders a Qwen-style one-shot chat prompt from `conversation_history`, `query`, candidate tools, and the API schema, then queries a single Ollama model and compares the generated JSON against the ground truth.

Example:

```bash
python3 ollama_inference_oneshot.py \
  --model_name qwen3-oneshot:latest \
  --test_key base,complex \
  --o datasets/result/260406-ablation/qwen3-oneshot.tsv
```

Key options:

- `--model_name` selects the Ollama model to evaluate
- `--test_key` selects one or more built-in split groups such as `base`, `complex`, or `swap`
- `--host` overrides the Ollama server address
- `--o` sets the output TSV path

The output TSV stores the model generation, ground truth, pass/fail results for `plan`, `arguments`, and `all`, plus the originating test file and turn.

### `ollama_inference_multi.py`

This script evaluates a single planning model over multiple evaluation files. It supports both:

- base/history prompting, where the model sees `conversation_history` and the raw `query`
- rewrite prompting, where the model sees the ground-truth `rewrited_query`

The exact model and prompt template are selected by the `--t` configuration key.

Example:

```bash
python3 ollama_inference_multi.py \
  --t base-qwen3 \
  --test_key base,complex \
  --o datasets/result/260406-ablation/qwen3-new-base.tsv
```

Key options:

- `--t` selects a predefined model and prompt-mode pair such as `base-qwen3`, `rewrite-qwen3`, `base-llama3`, or `new-base-qwen3`
- `--test_key` selects built-in file groups such as `base`, `complex`, `swap`, and `manual_rewrited`
- `--host` overrides the Ollama server address
- `--o` sets the output TSV path

This is the main script for measuring planning accuracy when the evaluation uses either the original query or the ground-truth rewritten query as input.

### `rma_plan_pipeline.py`

This script evaluates the full two-stage pipeline:

1. an RMA model rewrites the input into a `rewrited_query`
2. a planning model predicts `plan` and `arguments` from that generated rewritten query

Unlike `ollama_inference_multi.py`, this script does not use the ground-truth rewritten query. It measures end-to-end behavior of the rewrite-then-planning pipeline.

Example:

```bash
python3 rma_plan_pipeline.py \
  --model_family qwen3 \
  --test_key base,complex \
  --o datasets/result/260405-planning/qwen3.tsv
```

Key options:

- `--model_family` selects a predefined rewrite-model and planning-model pair such as `qwen3`, `llama3`, `phi4`, `qwen25`, or experiment-specific variants like `qwen3-multitask`
- `--test_key` selects built-in split groups such as `base`, `complex`, `manual`, `advanced_manual`, and `manual_rewrited`
- `--rewrite_host` and `--plan_host` let you point the two stages at different Ollama servers
- `--o` sets the output TSV path

The output TSV includes both stages of the pipeline, including the rewrite prompt, generated rewritten query, planning prompt, planning generation, and final correctness columns.

## Model Merge

For the main adapter-based training workflows, model export is already built into the training scripts. In particular, the following scripts all run the same postprocessing pipeline after training unless you explicitly disable it:

- `train/train_sentence_rewriter.py`
- `train/train_legacy_integrated.py`
- `train/train_multitask_rma.py`
- `train/train_oneshot_rma_qwen.py`

After the training step finishes, these scripts automatically do the following:

1. select the exported adapter checkpoint or final adapter directory
2. merge the LoRA adapter into the base Hugging Face model
3. convert the merged model to GGUF with `llama.cpp`
4. write an Ollama `Modelfile`
5. register the model with `ollama create`

In other words, for these four training entry points, there is normally no separate manual merge step.

Common controls:

- `--skip_postprocess` keeps only the training outputs and skips merge, GGUF export, and Ollama registration
- `--postprocess_only` reruns only the export pipeline from an existing training output directory
- `--export_checkpoint_epoch` exports a specific saved epoch checkpoint
- `--export_source {final_checkpoint,final_adapter}` chooses whether export starts from the final checkpoint or final adapter directory
- `--merged_dir`, `--gguf_path`, `--modelfile`, and `--ollama_model_name` override the default export targets
- `--base_dir`, `--base_model`, `--hf_token`, and `--force_download` control how the base model is resolved before merge
- `--ollama_host`, `--ollama_models_dir`, `--ollama_bin`, and `--llama_cpp_dir` control the local Ollama and GGUF conversion environment

Typical usage:

```bash
python3 train/train_multitask_rma.py \
  --model_name "Qwen/Qwen3-4B"
```

If you want to train first and export later, use the same script in two phases:

```bash
python3 train/train_multitask_rma.py \
  --model_name "Qwen/Qwen3-4B" \
  --skip_postprocess

python3 train/train_multitask_rma.py \
  --model_name "Qwen/Qwen3-4B" \
  --postprocess_only
```

The same postprocessing interface is shared by all four scripts above, so the same flags and workflow apply to `train/train_sentence_rewriter.py`, `train/train_legacy_integrated.py`, `train/train_multitask_rma.py`, and `train/train_oneshot_rma_qwen.py`.

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
