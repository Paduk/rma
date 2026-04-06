#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-o4-mini}"
FILTER_MODEL="${FILTER_MODEL:-o4-mini}"
INPUT_DATADIR="${INPUT_DATADIR:-/home/hj153lee/RMA/o4_datagen}"
OUTPUT_DATADIR="${OUTPUT_DATADIR:-$ROOT_DIR/picky_complex_runs}"
API_FILE="${API_FILE:-apis/api_v3.0.1.jsonl}"
TURN="${TURN:-5}"
REFERED_TURN="${REFERED_TURN:-}"

TARGET_FILE="${TARGET_FILE:-}"
TARGET_COUNT="${TARGET_COUNT:-}"
OVERGEN_COUNT="${OVERGEN_COUNT:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_FILTERING="${RUN_FILTERING:-1}"

mkdir -p "$OUTPUT_DATADIR" logs

TARGET_ARGS=(--target-mode complex)
if [[ -n "$TARGET_FILE" ]]; then
  TARGET_ARGS+=(--target-file "$TARGET_FILE")
fi
if [[ -n "$TARGET_COUNT" ]]; then
  TARGET_ARGS+=(--target-count "$TARGET_COUNT")
fi
if [[ -n "$OVERGEN_COUNT" ]]; then
  TARGET_ARGS+=(--overgen-count "$OVERGEN_COUNT")
fi
if [[ -n "$REFERED_TURN" ]]; then
  TARGET_ARGS+=(--refered-turn "$REFERED_TURN")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  TARGET_ARGS+=(--dry-run)
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

run_turn_3_complex() {
  run_cmd python3 difficulty_s3_generator.py \
    --t "${INPUT_DATADIR}/it3_s1_filtered.jsonl" \
    --api "$API_FILE" \
    --o "${OUTPUT_DATADIR}/it2_s3_complex.jsonl" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step complex_s3 \
    --t1 "${INPUT_DATADIR}/it2_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it2_s3_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s3_complex_idxed.jsonl" \
    --model "$MODEL"

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it2_s3_complex_idxed.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s4_complex.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it2_s3_complex_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it2_s4_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s1_complex.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it3_s1_complex.jsonl" \
      --o "logs/it3_s1_complex_log.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

run_turn_4_complex() {
  run_cmd python3 difficulty_s3_generator.py \
    --t "${INPUT_DATADIR}/it4_s1_filtered.jsonl" \
    --api "$API_FILE" \
    --o "${OUTPUT_DATADIR}/it3_s3_complex.jsonl" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step complex_s3 \
    --t1 "${INPUT_DATADIR}/it3_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it3_s3_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s3_complex_idxed.jsonl" \
    --model "$MODEL"

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it3_s3_complex_idxed.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s4_complex.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it3_s3_complex_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it3_s4_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s1_complex.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it4_s1_complex.jsonl" \
      --o "logs/it4_s1_complex_log.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

run_turn_5_complex() {
  run_cmd python3 difficulty_s3_generator.py \
    --t "${INPUT_DATADIR}/it5_s1_filtered.jsonl" \
    --api "$API_FILE" \
    --o "${OUTPUT_DATADIR}/it4_s3_complex.jsonl" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step complex_s3 \
    --t1 "${INPUT_DATADIR}/it4_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it4_s3_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s3_complex_idxed.jsonl" \
    --model "$MODEL"

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it4_s3_complex_idxed.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s4_complex.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it4_s3_complex_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it4_s4_complex.jsonl" \
    --o "${OUTPUT_DATADIR}/it5_s1_complex.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it5_s1_complex.jsonl" \
      --o "logs/it5_s1_complex_log.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

echo "ROOT_DIR=$ROOT_DIR"
echo "TURN=$TURN"
echo "REFERED_TURN=${REFERED_TURN:-<none>}"
echo "MODEL=$MODEL"
echo "FILTER_MODEL=$FILTER_MODEL"
echo "INPUT_DATADIR=$INPUT_DATADIR"
echo "OUTPUT_DATADIR=$OUTPUT_DATADIR"
echo "API_FILE=$API_FILE"
echo "TARGET_FILE=${TARGET_FILE:-<none>}"
echo "TARGET_COUNT=${TARGET_COUNT:-<none>}"
echo "OVERGEN_COUNT=${OVERGEN_COUNT:-<none>}"
echo "DRY_RUN=$DRY_RUN"
echo "RUN_FILTERING=$RUN_FILTERING"

case "$TURN" in
  3) run_turn_3_complex ;;
  4) run_turn_4_complex ;;
  5) run_turn_5_complex ;;
  *)
    echo "Unsupported TURN: $TURN" >&2
    exit 1
    ;;
esac
