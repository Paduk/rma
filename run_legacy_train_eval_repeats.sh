#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROFILE="granite3.3-2b"
MODEL_NAME=""
TRAIN_TYPE="history"
RUNS="3"
TAGS=""
TAG_PREFIX="t"
GPU=""
OLLAMA_HOST="127.0.0.1:11435"
INFER_HOST=""
TEST_KEY="base,complex"
LLAMA_CPP_DIR=""
RESULT_DIR=""
OUTPUT_PREFIX=""
EVAL_T=""
NUM_PARALLEL=""
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_legacy_train_eval_repeats.sh [options]

Examples:
  ./run_legacy_train_eval_repeats.sh --gpu 7 --profile granite3.3-2b --train_type history --tags t1,t2,t3

  ./run_legacy_train_eval_repeats.sh --gpu 7 --profile granite3.3-2b --train_type history --runs 5

  ./run_legacy_train_eval_repeats.sh \
    --gpu 7 \
    --profile glm-edge-4b \
    --train_type history \
    --tags t1,t2,t3 \
    --result_dir datasets/result/glm4 \
    --output_prefix glm4-new-base

  ./run_legacy_train_eval_repeats.sh --gpu 7 --profile qwen2.5 --train_type history --tags t1,t2,t3

For arbitrary --profile generic / --model_name combinations, pass --eval_t explicitly
if ollama_inference_multi.py cannot infer the test type from the profile.

Options:
  --profile NAME          train_legacy_integrated.py profile. Default: granite3.3-2b
  --model_name NAME       Optional train model_name override.
  --train_type TYPE       history or rewrite. Default: history
  --runs N                Generate tags t1..tN. Default: 3
  --tags LIST             Comma-separated tags, e.g. t1,t2,t3. Overrides --runs.
  --tag_prefix PREFIX     Prefix used with --runs. Default: t
  --gpu IDS               CUDA_VISIBLE_DEVICES value, e.g. 7 or 0,1.
  --ollama_host HOST      Host used for ollama create. Default: 127.0.0.1:11435
  --infer_host URL        Host URL used by inference. Default: http://<ollama_host>
  --test_key KEYS         Comma-separated eval keys. Default: base,complex
  --llama_cpp_dir DIR     llama.cpp dir used for GGUF conversion.
                          Default for smollm3: /home/hj153lee/llama.cpp-smollm3
  --result_dir DIR        Eval output dir. Default: datasets/result/<profile group>
  --output_prefix PREFIX  Eval filename prefix. Default: <group>-new-base for history
  --eval_t TYPE           Override ollama_inference_multi.py --t value.
  --num_parallel N        Optional inference parallelism passed as --multi N.
  --dry_run               Print commands without executing.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"; shift 2 ;;
    --model_name)
      MODEL_NAME="$2"; shift 2 ;;
    --train_type)
      TRAIN_TYPE="$2"; shift 2 ;;
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
    --llama_cpp_dir)
      LLAMA_CPP_DIR="$2"; shift 2 ;;
    --result_dir)
      RESULT_DIR="$2"; shift 2 ;;
    --output_prefix)
      OUTPUT_PREFIX="$2"; shift 2 ;;
    --eval_t)
      EVAL_T="$2"; shift 2 ;;
    --num_parallel|--multi)
      NUM_PARALLEL="$2"; shift 2 ;;
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

if [[ "$TRAIN_TYPE" != "history" && "$TRAIN_TYPE" != "rewrite" ]]; then
  echo "--train_type must be history or rewrite: $TRAIN_TYPE" >&2
  exit 2
fi

if [[ "$PROFILE" == "qwen25" ]]; then
  PROFILE="qwen2.5"
fi

default_model_name_for_profile() {
  case "$1" in
    qwen) echo "Qwen/Qwen3-4B" ;;
    qwen2.5|qwen25) echo "Qwen/Qwen2.5-3B-Instruct" ;;
    phi) echo "microsoft/Phi-4-mini-instruct" ;;
    llama) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
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
    generic)
      if [[ -z "$MODEL_NAME" ]]; then
        echo "--profile generic requires --model_name" >&2
        exit 2
      fi
      echo "$MODEL_NAME"
      ;;
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

default_eval_t() {
  local profile="$1"
  local train_type="$2"

  if [[ "$train_type" == "history" ]]; then
    case "$profile" in
      qwen) echo "new-base-qwen3" ;;
      qwen2.5|qwen25) echo "history-qwen2.5" ;;
      phi) echo "new-base-phi4" ;;
      llama) echo "new-base-llama3" ;;
      *) echo "history-${profile}" ;;
    esac
  else
    case "$profile" in
      qwen) echo "rewrite-qwen3" ;;
      qwen2.5|qwen25) echo "rewrite-qwen2.5" ;;
      phi) echo "rewrite-phi4" ;;
      llama) echo "rewrite-llama3" ;;
      *) echo "rewrite-${profile}" ;;
    esac
  fi
}

default_result_group() {
  case "$1" in
    qwen) echo "qwen3" ;;
    qwen2.5|qwen25) echo "qwen2.5" ;;
    phi) echo "phi4" ;;
    llama) echo "llama3" ;;
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
if [[ -z "$LLAMA_CPP_DIR" ]]; then
  lower_model_name="${EFFECTIVE_MODEL_NAME,,}"
  if [[ "$lower_model_name" == *"smollm3"* ]]; then
    LLAMA_CPP_DIR="/home/hj153lee/llama.cpp-smollm3"
  fi
fi
if [[ "$PROFILE" == "generic" ]]; then
  RESULT_GROUP="$MODEL_SLUG"
else
  RESULT_GROUP="$(default_result_group "$PROFILE")"
fi

if [[ -z "$RESULT_DIR" ]]; then
  RESULT_DIR="datasets/result/${RESULT_GROUP}"
fi

if [[ -z "$OUTPUT_PREFIX" ]]; then
  if [[ "$TRAIN_TYPE" == "history" ]]; then
    OUTPUT_PREFIX="${RESULT_GROUP}-new-base"
  else
    OUTPUT_PREFIX="${RESULT_GROUP}-rewrite"
  fi
fi

if [[ -z "$INFER_HOST" ]]; then
  INFER_HOST="$(infer_host_url "$OLLAMA_HOST")"
fi

mkdir -p "$RESULT_DIR"

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

  eval_t="${EVAL_T:-$(default_eval_t "$PROFILE" "$TRAIN_TYPE")}"
  ollama_model_name="${MODEL_SLUG}-${PROFILE}-${TRAIN_TYPE}-${tag}"
  output_path="${RESULT_DIR}/${OUTPUT_PREFIX}-${tag}.tsv"

  train_cmd=(python train/train_legacy_integrated.py
    --profile "$PROFILE"
    --train_type "$TRAIN_TYPE"
    --prefix "$tag"
    --ollama_host "$OLLAMA_HOST"
    --ollama_model_name "$ollama_model_name")

  if [[ -n "$MODEL_NAME" ]]; then
    train_cmd+=(--model_name "$MODEL_NAME")
  fi

  if [[ -n "$LLAMA_CPP_DIR" ]]; then
    train_cmd+=(--llama_cpp_dir "$LLAMA_CPP_DIR")
  fi

  eval_cmd=(python3 ollama_inference_multi.py
    --t "$eval_t"
    --model "${ollama_model_name}:latest"
    --test_key "$TEST_KEY"
    --o "$output_path"
    --host "$INFER_HOST")

  if [[ -n "$NUM_PARALLEL" ]]; then
    eval_cmd+=(--multi "$NUM_PARALLEL")
  fi

  echo
  echo "=== Run tag: $tag ==="
  echo "profile: $PROFILE"
  echo "model_name: $EFFECTIVE_MODEL_NAME"
  echo "train_type: $TRAIN_TYPE"
  echo "ollama_model_name: ${ollama_model_name}:latest"
  echo "eval_t: $eval_t"
  echo "output: $output_path"
  echo "llama_cpp_dir: ${LLAMA_CPP_DIR:-<train default>}"

  if [[ -n "$GPU" ]]; then
    run_cmd env "CUDA_VISIBLE_DEVICES=${GPU}" "${train_cmd[@]}"
  else
    run_cmd "${train_cmd[@]}"
  fi

  run_cmd "${eval_cmd[@]}"
done
