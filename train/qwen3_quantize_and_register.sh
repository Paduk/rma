#!/usr/bin/env bash
set -euo pipefail

ENV_ACTIVATE='source /mnt/data/miniconda3/bin/activate && conda activate mobile-agent-v3-py311'
MODEL_DIR="/home/hj153lee/RMA/qwen3-1.7b-merged"
LLAMA_CPP_DIR="/home/hj153lee/llama.cpp"
ARTIFACT_DIR="/home/hj153lee/qwen3-1.7b-artifacts"
FULL_GGUF="${ARTIFACT_DIR}/qwen3-1.7b-rma-f16.gguf"
Q4_GGUF="${ARTIFACT_DIR}/qwen3-1.7b-rma-q4_k_m.gguf"
MODELFILE="${ARTIFACT_DIR}/qwen3-1.7b-rma-q4_k_m.Modelfile"
OLLAMA_MODEL_NAME="qwen3-1.7b-rma-q4km"
HF_SMOKE_OUTPUT="${ARTIFACT_DIR}/hf_smoke_test.json"
OLLAMA_EVAL_OUTPUT="${ARTIFACT_DIR}/ollama_eval_q4km.json"
PROMPTS_FILE="/home/hj153lee/RMA/train/ollama_eval_prompts.jsonl"
LLAMA_BUILD_DIR="${LLAMA_CPP_DIR}/build"
LLAMA_QUANTIZE_BIN="${LLAMA_BUILD_DIR}/bin/llama-quantize"
LLAMA_CLI_BIN="${LLAMA_BUILD_DIR}/bin/llama-cli"
OLLAMA_MODELS_DIR="/home/hj153lee/.ollama-qwen3-test"
OLLAMA_HOST_VALUE="127.0.0.1:11435"
OLLAMA_API_HOST="http://${OLLAMA_HOST_VALUE}"
OLLAMA_LOG="${ARTIFACT_DIR}/ollama_serve.log"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Merged model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"
mkdir -p "${OLLAMA_MODELS_DIR}"

for required_file in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  if [[ ! -f "${MODEL_DIR}/${required_file}" ]]; then
    echo "Missing required file: ${MODEL_DIR}/${required_file}" >&2
    exit 1
  fi
done

if [[ ! -x "${LLAMA_QUANTIZE_BIN}" || ! -x "${LLAMA_CLI_BIN}" ]]; then
  cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "${LLAMA_BUILD_DIR}" --target llama-quantize llama-cli -j"$(nproc)"
fi

bash -lc "${ENV_ACTIVATE} && python /home/hj153lee/RMA/train/hf_smoke_test.py --model-dir '${MODEL_DIR}' --output '${HF_SMOKE_OUTPUT}'"

if [[ ! -f "${FULL_GGUF}" ]]; then
  bash -lc "${ENV_ACTIVATE} && cd '${LLAMA_CPP_DIR}' && python convert_hf_to_gguf.py '${MODEL_DIR}' --outfile '${FULL_GGUF}' --outtype f16"
fi

if [[ ! -f "${Q4_GGUF}" ]]; then
  "${LLAMA_QUANTIZE_BIN}" "${FULL_GGUF}" "${Q4_GGUF}" Q4_K_M
fi

cat > "${MODELFILE}" <<EOF
FROM ./${Q4_GGUF##*/}
PARAMETER temperature 0
EOF

if ! curl -fsS "${OLLAMA_API_HOST}/api/tags" >/dev/null 2>&1; then
  nohup env OLLAMA_HOST="${OLLAMA_HOST_VALUE}" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" \
    ollama serve >"${OLLAMA_LOG}" 2>&1 &
  for _ in $(seq 1 30); do
    if curl -fsS "${OLLAMA_API_HOST}/api/tags" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

env OLLAMA_HOST="${OLLAMA_HOST_VALUE}" OLLAMA_MODELS="${OLLAMA_MODELS_DIR}" \
  bash -lc "cd '${ARTIFACT_DIR}' && ollama create '${OLLAMA_MODEL_NAME}' -f '${MODELFILE}'"

bash -lc "${ENV_ACTIVATE} && python /home/hj153lee/RMA/train/ollama_eval.py --model '${OLLAMA_MODEL_NAME}' --host '${OLLAMA_API_HOST}' --prompts '${PROMPTS_FILE}' --output '${OLLAMA_EVAL_OUTPUT}'"

echo "Artifacts:"
echo "  HF smoke: ${HF_SMOKE_OUTPUT}"
echo "  Full GGUF: ${FULL_GGUF}"
echo "  Quantized GGUF: ${Q4_GGUF}"
echo "  Modelfile: ${MODELFILE}"
echo "  Ollama eval: ${OLLAMA_EVAL_OUTPUT}"
echo "  Ollama model: ${OLLAMA_MODEL_NAME}"
echo "  Ollama host: ${OLLAMA_API_HOST}"
echo "  Ollama models dir: ${OLLAMA_MODELS_DIR}"
