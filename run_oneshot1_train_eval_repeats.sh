#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROFILE="qwen"
NO_PROFILE=0
MODEL_NAME=""
PROMPT_TOKENIZER_NAME=""
RUNS="1"
TAGS=""
TAG_PREFIX="t"
GPU=""
OLLAMA_HOST="127.0.0.1:11436"
INFER_HOST=""
TEST_KEY="base,complex"
TOOLS_PATH=""
RESULT_DIR=""
OUTPUT_PREFIX=""
OUTPUT_ROOT=""
ARTIFACT_DIR=""
NUM_PARALLEL=""
TRUST_REMOTE_CODE=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_oneshot1_train_eval_repeats.sh [options]

Examples:
  ./run_oneshot1_train_eval_repeats.sh --gpu 3 --profile qwen --tags t1,t2,t3

  ./run_oneshot1_train_eval_repeats.sh \
    --gpu 3 \
    --profile qwen2.5 \
    --tags t1,t2,t3 \
    --ollama_host 127.0.0.1:11436

  ./run_oneshot1_train_eval_repeats.sh \
    --gpu 6 \
    --profile phi \
    --tags t2 \
    --ollama_host 127.0.0.1:11436

  ./run_oneshot1_train_eval_repeats.sh \
    --gpu 3 \
    --no_profile \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --tags t1,t2,t3

Options:
  --profile NAME              train_oneshot_rma_qwen.py / ollama_inference_oneshot1.py profile. Default: qwen
  --no_profile                Do not pass --profile. Requires --model_name.
  --model_name NAME           Optional train model_name override, or required with --no_profile.
  --prompt_tokenizer_name ID  Optional tokenizer for eval when no profile can infer it.
  --runs N                    Generate tags t1..tN. Default: 1
  --tags LIST                 Comma-separated tags, e.g. t1,t2,t3. Overrides --runs.
  --tag_prefix PREFIX         Prefix used with --runs. Default: t
  --gpu IDS                   CUDA_VISIBLE_DEVICES value, e.g. 3 or 0,1.
  --ollama_host HOST          Host used for ollama create. Default: 127.0.0.1:11436
  --infer_host URL            Host URL used by inference. Default: http://<ollama_host>
  --test_key KEYS             Comma-separated eval keys. Default: base,complex
  --tools_path PATH           Optional simple_api.json path passed to train and eval.
  --result_dir DIR            Eval output dir. Default: datasets/result/<profile group>
  --output_prefix PREFIX      Eval filename prefix. Default: <group>-oneshot1
  --output_root DIR           Optional train output_root override.
  --artifact_dir DIR          Optional parent dir for merged/gguf/modelfile artifacts.
                              Default: dirname(output_root), or /mnt/data/hj153lee.
  --num_parallel N            Optional inference parallelism passed as --multi N.
  --trust_remote_code VALUE   auto, true, or false. Default: false for phi, otherwise train script default.
  --dry_run                   Print commands without executing.
  -h, --help                  Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"; NO_PROFILE=0; shift 2 ;;
    --no_profile)
      NO_PROFILE=1; shift ;;
    --model_name)
      MODEL_NAME="$2"; shift 2 ;;
    --prompt_tokenizer_name)
      PROMPT_TOKENIZER_NAME="$2"; shift 2 ;;
    --runs)
      RUNS="$2"; TAGS=""; shift 2 ;;
    --tags)
      TAGS="$2"; shift 2 ;;
    --tag_prefix)
      TAG_PREFIX="$2"; shift 2 ;;
    --gpu)
      GPU="$2"; shift 2 ;;
    --ollama_host)
      OLLAMA_HOST="$2"; shift 2 ;;
    --infer_host)
      INFER_HOST="$2"; shift 2 ;;
    --test_key)
      TEST_KEY="$2"; shift 2 ;;
    --tools_path)
      TOOLS_PATH="$2"; shift 2 ;;
    --result_dir)
      RESULT_DIR="$2"; shift 2 ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"; shift 2 ;;
    --output_root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --artifact_dir)
      ARTIFACT_DIR="$2"; shift 2 ;;
    --num_parallel|--multi)
      NUM_PARALLEL="$2"; shift 2 ;;
    --trust_remote_code|--remote_trust_code)
      TRUST_REMOTE_CODE="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ "$NO_PROFILE" -eq 1 && -z "$MODEL_NAME" ]]; then
  echo "--no_profile requires --model_name" >&2
  exit 2
fi

if [[ "$PROFILE" == "qwen25" ]]; then
  PROFILE="qwen2.5"
fi

default_model_name_for_profile() {
  case "$1" in
    qwen) echo "Qwen/Qwen3-4B" ;;
    qwen2.5|qwen25) echo "Qwen/Qwen2.5-3B-Instruct" ;;
    qwen3-1.7b) echo "Qwen/Qwen3-1.7B" ;;
    qwen3-0.6b) echo "Qwen/Qwen3-0.6B" ;;
    phi) echo "microsoft/Phi-4-mini-instruct" ;;
    llama) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
    gemma) echo "google/gemma-3-4b-it" ;;
    glm-edge-1.5b) echo "zai-org/glm-edge-1.5b-chat" ;;
    glm-edge-4b) echo "zai-org/glm-edge-4b-chat" ;;
    smollm2-1.7b) echo "HuggingFaceTB/SmolLM2-1.7B" ;;
    smollm2-1.7b-instruct) echo "HuggingFaceTB/SmolLM2-1.7B-Instruct" ;;
    smollm3-3b) echo "HuggingFaceTB/SmolLM3-3B" ;;
    falcon3-1b) echo "tiiuae/Falcon3-1B-Instruct" ;;
    falcon3-1b-base) echo "tiiuae/Falcon3-1B-Base" ;;
    falcon3-3b) echo "tiiuae/Falcon3-3B-Instruct" ;;
    falcon3-3b-base) echo "tiiuae/Falcon3-3B-Base" ;;
    exaone4-1.2b) echo "LGAI-EXAONE/EXAONE-4.0-1.2B" ;;
    olmo2-1b) echo "allenai/OLMo-2-0425-1B" ;;
    olmo2-1b-instruct) echo "allenai/OLMo-2-0425-1B-Instruct" ;;
    granite3.3-2b) echo "ibm-granite/granite-3.3-2b-instruct" ;;
    lfm2.5-1.2b) echo "LiquidAI/LFM2.5-1.2B-Instruct" ;;
    *)
      if [[ -z "$MODEL_NAME" ]]; then
        echo "Unknown profile and no --model_name provided: $1" >&2
        exit 2
      fi
      echo "$MODEL_NAME"
      ;;
  esac
}

infer_model_slug() {
  local model_name="$1"
  local lower="${model_name,,}"
  if [[ "$lower" == *"qwen3"* ]]; then
    echo "qwen3"
  elif [[ "$lower" == *"qwen2.5"* || "$lower" == *"qwen25"* ]]; then
    echo "qwen25"
  elif [[ "$lower" == *"phi-4"* || "$lower" == *"phi4"* ]]; then
    echo "phi4"
  elif [[ "$lower" == *"llama"* ]]; then
    echo "llama3"
  elif [[ "$lower" == *"gemma"* ]]; then
    echo "gemma"
  else
    basename "$model_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
  fi
}

default_result_group() {
  case "$1" in
    qwen|qwen3-1.7b|qwen3-0.6b) echo "qwen3" ;;
    qwen2.5|qwen25) echo "qwen2.5" ;;
    phi) echo "phi4" ;;
    llama) echo "llama3" ;;
    gemma) echo "gemma" ;;
    granite3.3-2b) echo "granite" ;;
    glm-edge-1.5b) echo "glm-1.5" ;;
    glm-edge-4b) echo "glm4" ;;
    *) echo "$1" | sed -E 's/[^a-zA-Z0-9._-]+/-/g' ;;
  esac
}

infer_host_url() {
  local host="$1"
  if [[ "$host" == http://* || "$host" == https://* ]]; then
    echo "$host"
  else
    echo "http://${host}"
  fi
}

if [[ -z "$TAGS" ]]; then
  if ! [[ "$RUNS" =~ ^[0-9]+$ ]] || [[ "$RUNS" -lt 1 ]]; then
    echo "--runs must be a positive integer: $RUNS" >&2
    exit 2
  fi
  tags=()
  for ((i = 1; i <= RUNS; i++)); do
    tags+=("${TAG_PREFIX}${i}")
  done
else
  IFS=',' read -r -a tags <<< "$TAGS"
fi

if [[ -n "$MODEL_NAME" ]]; then
  EFFECTIVE_MODEL_NAME="$MODEL_NAME"
else
  EFFECTIVE_MODEL_NAME="$(default_model_name_for_profile "$PROFILE")"
fi
MODEL_SLUG="$(infer_model_slug "$EFFECTIVE_MODEL_NAME")"

if [[ -z "$TRUST_REMOTE_CODE" ]]; then
  lower_model_name="${EFFECTIVE_MODEL_NAME,,}"
  if [[ "$lower_model_name" == *"phi-4"* || "$lower_model_name" == *"phi4"* ]]; then
    TRUST_REMOTE_CODE="false"
  fi
fi

if [[ "$NO_PROFILE" -eq 1 ]]; then
  RESULT_GROUP="$MODEL_SLUG"
else
  RESULT_GROUP="$(default_result_group "$PROFILE")"
fi

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="datasets/result/${RESULT_GROUP}"
fi

if [[ -z "$OUTPUT_PREFIX" ]]; then
  OUTPUT_PREFIX="${RESULT_GROUP}-oneshot1"
fi

if [[ -z "$INFER_HOST" ]]; then
  INFER_HOST="$(infer_host_url "$OLLAMA_HOST")"
fi

if [[ -z "$ARTIFACT_DIR" ]]; then
  if [[ -n "$OUTPUT_ROOT" ]]; then
    ARTIFACT_DIR="$(dirname "$OUTPUT_ROOT")"
  else
    ARTIFACT_DIR="/mnt/data/hj153lee"
  fi
fi

mkdir -p "$RESULT_DIR" "$ARTIFACT_DIR"

run_cmd() {
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

for tag in "${tags[@]}"; do
  if [[ -z "$tag" ]]; then
    continue
  fi

  output_tag="$tag"
  ollama_model_name="${MODEL_SLUG}-oneshot1-${tag}"
  artifact_stem="${MODEL_SLUG}-rma-oneshot1-${tag}"
  merged_dir="${ARTIFACT_DIR}/${artifact_stem}-merged"
  gguf_path="${ARTIFACT_DIR}/${artifact_stem}-merged.gguf"
  modelfile="${HOME}/Modelfile.${artifact_stem}.gguf"
  output_path="${RESULT_DIR}/${OUTPUT_PREFIX}-${tag}.tsv"

  train_cmd=(python3 train/train_oneshot_rma_qwen.py
    --output_tag "$output_tag"
    --ollama_model_name "$ollama_model_name"
    --ollama_host "$OLLAMA_HOST"
    --merged_dir "$merged_dir"
    --gguf_path "$gguf_path"
    --modelfile "$modelfile")

  eval_cmd=(python3 ollama_inference_oneshot1.py
    --model_name "${ollama_model_name}:latest"
    --host "$INFER_HOST"
    --test_key "$TEST_KEY"
    --o "$output_path")

  if [[ "$NO_PROFILE" -eq 0 ]]; then
    train_cmd+=(--profile "$PROFILE")
    eval_cmd+=(--profile "$PROFILE")
  fi

  if [[ -n "$MODEL_NAME" ]]; then
    train_cmd+=(--model_name "$MODEL_NAME")
  fi

  if [[ -n "$OUTPUT_ROOT" ]]; then
    train_cmd+=(--output_root "$OUTPUT_ROOT")
  fi

  if [[ -n "$TOOLS_PATH" ]]; then
    train_cmd+=(--tools_path "$TOOLS_PATH")
    eval_cmd+=(--tools_path "$TOOLS_PATH")
  fi

  if [[ -n "$TRUST_REMOTE_CODE" ]]; then
    train_cmd+=(--trust_remote_code "$TRUST_REMOTE_CODE")
  fi

  if [[ -n "$PROMPT_TOKENIZER_NAME" ]]; then
    eval_cmd+=(--prompt_tokenizer_name "$PROMPT_TOKENIZER_NAME")
  elif [[ -n "$MODEL_NAME" ]]; then
    eval_cmd+=(--prompt_tokenizer_name "$EFFECTIVE_MODEL_NAME")
  fi

  if [[ -n "$NUM_PARALLEL" ]]; then
    eval_cmd+=(--multi "$NUM_PARALLEL")
  fi

  echo
  echo "=== One-shot1 run tag: $tag ==="
  echo "profile: $([[ "$NO_PROFILE" -eq 1 ]] && echo '<none>' || echo "$PROFILE")"
  echo "model_name: $EFFECTIVE_MODEL_NAME"
  echo "ollama_model_name: ${ollama_model_name}:latest"
  echo "ollama_host: $OLLAMA_HOST"
  echo "infer_host: $INFER_HOST"
  echo "trust_remote_code: ${TRUST_REMOTE_CODE:-<train default>}"
  echo "output: $output_path"
  echo "merged_dir: $merged_dir"
  echo "gguf_path: $gguf_path"

  if [[ -n "$GPU" ]]; then
    run_cmd env "CUDA_VISIBLE_DEVICES=${GPU}" "${train_cmd[@]}"
  else
    run_cmd "${train_cmd[@]}"
  fi

  run_cmd "${eval_cmd[@]}"
done
