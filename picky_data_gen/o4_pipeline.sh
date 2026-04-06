#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-o4-mini}"
FILTER_MODEL="${FILTER_MODEL:-o4-mini}"
INPUT_DATADIR="${INPUT_DATADIR:-/home/hj153lee/RMA/o4_datagen}"
OUTPUT_DATADIR="${OUTPUT_DATADIR:-$ROOT_DIR/picky_runs}"
API_FILE="${API_FILE:-apis/api_v3.0.1.jsonl}"
TURN="${TURN:-2}"

TARGET_FILE="${TARGET_FILE:-}"
TARGET_COUNT="${TARGET_COUNT:-}"
OVERGEN_COUNT="${OVERGEN_COUNT:-}"
DRY_RUN="${DRY_RUN:-0}"

RUN_DEDUP="${RUN_DEDUP:-1}"
RUN_REWRITE="${RUN_REWRITE:-1}"
RUN_FILLING="${RUN_FILLING:-0}"
RUN_BALANCING="${RUN_BALANCING:-0}"
RUN_FILTERING="${RUN_FILTERING:-0}"

mkdir -p "$OUTPUT_DATADIR" logs

TARGET_ARGS=()
if [[ -n "$TARGET_FILE" ]]; then
  TARGET_ARGS+=(--target-file "$TARGET_FILE")
fi
if [[ -n "$TARGET_COUNT" ]]; then
  TARGET_ARGS+=(--target-count "$TARGET_COUNT")
fi
if [[ -n "$OVERGEN_COUNT" ]]; then
  TARGET_ARGS+=(--overgen-count "$OVERGEN_COUNT")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  TARGET_ARGS+=(--dry-run)
fi

run_cmd() {
  echo "+ $*"
  "$@"
}

localize_dedup_input() {
  local source_path="$1"
  local target_path="${OUTPUT_DATADIR}/$(basename "$source_path")"

  if [[ "$source_path" == "$target_path" ]]; then
    echo "$source_path"
    return 0
  fi

  cp "$source_path" "$target_path"
  echo "$target_path"
}

run_dedup() {
  local localized_files=()
  local source_file

  for source_file in "$@"; do
    localized_files+=("$(localize_dedup_input "$source_file")")
  done

  run_cmd python3 dataset_spliter.py --t_list "${localized_files[@]}"
}

resolve_turn_input() {
  local filtered_path="$1"
  local raw_path="$2"

  if [[ -f "$filtered_path" ]]; then
    echo "$filtered_path"
    return 0
  fi

  if [[ -f "$raw_path" ]]; then
    echo "$raw_path"
    return 0
  fi

  echo "Missing input files: $filtered_path and $raw_path" >&2
  return 1
}

run_turn_2() {
  run_cmd python3 batch_s2_generator.py \
    --s "${INPUT_DATADIR}/it1_s1.jsonl" \
    --o "${OUTPUT_DATADIR}/it1_s2.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step s2 \
    --t1 "${INPUT_DATADIR}/it1_s1.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it1_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it1_s2_idxed.jsonl"

  run_cmd python3 batch_s3_generator.py \
    --s "${OUTPUT_DATADIR}/it1_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it1_s3.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s3 \
    --t1 "${OUTPUT_DATADIR}/it1_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it1_s3.jsonl" \
    --o "${OUTPUT_DATADIR}/it1_s3_idxed.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_DEDUP" == "1" ]]; then
    run_dedup "${OUTPUT_DATADIR}/it1_s3_idxed.jsonl"
  fi

  if [[ "$RUN_REWRITE" == "1" ]]; then
    run_cmd python3 batch_s3_rewriter.py \
      --s "${OUTPUT_DATADIR}/it1_s3_idxed_dedup.jsonl" \
      --o "${OUTPUT_DATADIR}/it1_s3_idxed_rewrite.jsonl" \
      --model "$MODEL"
  fi

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it1_s3_idxed_rewrite.jsonl" \
    --o "${OUTPUT_DATADIR}/it1_s4.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it1_s3_idxed_rewrite.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it1_s4.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s1_nonnr.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILLING" == "1" ]]; then
    run_cmd python3 filling_datas.py \
      --s1 "${INPUT_DATADIR}/it1_s1_spare.jsonl" \
      --s2 "${OUTPUT_DATADIR}/it1_s2_idxed.jsonl" \
      --o "${OUTPUT_DATADIR}/it1_supplyments.jsonl"
  fi

  if [[ "$RUN_BALANCING" == "1" ]]; then
    run_cmd python3 balancing_data.py \
      --t1 "${OUTPUT_DATADIR}/it2_s1_nonnr.jsonl" \
      --t2 "${OUTPUT_DATADIR}/it1_supplyments.jsonl" \
      --o "${OUTPUT_DATADIR}/it2_s1.jsonl" \
      --model "$MODEL"
  fi

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it2_s1.jsonl" \
      --o "logs/it2_s1_log.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

run_turn_3() {
  local turn_input
  turn_input="$(resolve_turn_input "${INPUT_DATADIR}/it2_s1_filtered.jsonl" "${INPUT_DATADIR}/it2_s1.jsonl")"

  run_cmd python3 batch_s2_generator.py \
    --s "$turn_input" \
    --o "${OUTPUT_DATADIR}/it2_s2.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step s2 \
    --t1 "$turn_input" \
    --t2 "${OUTPUT_DATADIR}/it2_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s2_idxed.jsonl"

  run_cmd python3 batch_s3_generator.py \
    --s "${OUTPUT_DATADIR}/it2_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s3.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s3 \
    --t1 "${OUTPUT_DATADIR}/it2_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it2_s3.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s3_idxed.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_DEDUP" == "1" ]]; then
    run_dedup \
      "${INPUT_DATADIR}/it1_s3_idxed_rewrite.jsonl" \
      "${OUTPUT_DATADIR}/it2_s3_idxed.jsonl"
  fi

  if [[ "$RUN_REWRITE" == "1" ]]; then
    run_cmd python3 batch_s3_rewriter.py \
      --s "${OUTPUT_DATADIR}/it2_s3_idxed_dedup.jsonl" \
      --o "${OUTPUT_DATADIR}/it2_s3_idxed_rewrite.jsonl" \
      --model "$MODEL"
  fi

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it2_s3_idxed_rewrite.jsonl" \
    --o "${OUTPUT_DATADIR}/it2_s4.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it2_s3_idxed_rewrite.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it2_s4.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s1_nonnr.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILLING" == "1" ]]; then
    run_cmd python3 filling_datas.py \
      --s1 "${INPUT_DATADIR}/it1_s1_spare.jsonl" \
      --s2 "${OUTPUT_DATADIR}/it2_s2_idxed.jsonl" \
      --o "${OUTPUT_DATADIR}/it2_supplyments.jsonl"
  fi

  if [[ "$RUN_BALANCING" == "1" ]]; then
    run_cmd python3 balancing_data.py \
      --t1 "${OUTPUT_DATADIR}/it3_s1_nonnr.jsonl" \
      --t2 "${OUTPUT_DATADIR}/it2_supplyments.jsonl" \
      --o "${OUTPUT_DATADIR}/it3_s1.jsonl" \
      --model "$MODEL"
  fi

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it3_s1.jsonl" \
      --o "logs/it3_s1_log_gem.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

run_turn_4() {
  local turn_input
  turn_input="$(resolve_turn_input "${INPUT_DATADIR}/it3_s1_filtered.jsonl" "${INPUT_DATADIR}/it3_s1.jsonl")"

  run_cmd python3 batch_s2_generator.py \
    --s "$turn_input" \
    --o "${OUTPUT_DATADIR}/it3_s2.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step s2 \
    --t1 "$turn_input" \
    --t2 "${OUTPUT_DATADIR}/it3_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s2_idxed.jsonl"

  run_cmd python3 batch_s3_generator.py \
    --s "${OUTPUT_DATADIR}/it3_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s3.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s3 \
    --t1 "${OUTPUT_DATADIR}/it3_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it3_s3.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s3_idxed.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_DEDUP" == "1" ]]; then
    run_dedup \
      "${INPUT_DATADIR}/it1_s3_idxed_rewrite_dedup.jsonl" \
      "${INPUT_DATADIR}/it2_s3_idxed_rewrite.jsonl" \
      "${OUTPUT_DATADIR}/it3_s3_idxed.jsonl"
  fi

  if [[ "$RUN_REWRITE" == "1" ]]; then
    run_cmd python3 batch_s3_rewriter.py \
      --s "${OUTPUT_DATADIR}/it3_s3_idxed_dedup.jsonl" \
      --o "${OUTPUT_DATADIR}/it3_s3_idxed_rewrite.jsonl" \
      --model "$MODEL"
  fi

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it3_s3_idxed_rewrite.jsonl" \
    --o "${OUTPUT_DATADIR}/it3_s4.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it3_s3_idxed_rewrite.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it3_s4.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s1_nonnr.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILLING" == "1" ]]; then
    run_cmd python3 filling_datas.py \
      --s1 "${INPUT_DATADIR}/it1_s1_spare.jsonl" \
      --s2 "${OUTPUT_DATADIR}/it3_s2_idxed.jsonl" \
      --o "${OUTPUT_DATADIR}/it3_supplyments.jsonl"
  fi

  if [[ "$RUN_BALANCING" == "1" ]]; then
    run_cmd python3 balancing_data.py \
      --t1 "${OUTPUT_DATADIR}/it4_s1_nonnr.jsonl" \
      --t2 "${OUTPUT_DATADIR}/it3_supplyments.jsonl" \
      --o "${OUTPUT_DATADIR}/it4_s1.jsonl" \
      --model "$MODEL"
  fi

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it4_s1.jsonl" \
      --o "logs/it4_s1_log_gem.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

run_turn_5() {
  local turn_input
  turn_input="$(resolve_turn_input "${INPUT_DATADIR}/it4_s1_filtered.jsonl" "${INPUT_DATADIR}/it4_s1.jsonl")"

  run_cmd python3 batch_s2_generator.py \
    --s "$turn_input" \
    --o "${OUTPUT_DATADIR}/it4_s2.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  [[ "$DRY_RUN" == "1" ]] && return 0

  run_cmd python3 dataset_integration.py --step s2 \
    --t1 "$turn_input" \
    --t2 "${OUTPUT_DATADIR}/it4_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s2_idxed.jsonl"

  run_cmd python3 batch_s3_generator_difficult.py \
    --s "${OUTPUT_DATADIR}/it4_s2.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s3.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s3 \
    --t1 "${OUTPUT_DATADIR}/it4_s2_idxed.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it4_s3.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s3_idxed.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_DEDUP" == "1" ]]; then
    run_dedup \
      "${INPUT_DATADIR}/it1_s3_idxed_rewrite_dedup_dedup.jsonl" \
      "${INPUT_DATADIR}/it2_s3_idxed_rewrite_dedup.jsonl" \
      "${INPUT_DATADIR}/it3_s3_idxed_rewrite.jsonl" \
      "${OUTPUT_DATADIR}/it4_s3_idxed.jsonl"
  fi

  if [[ "$RUN_REWRITE" == "1" ]]; then
    run_cmd python3 batch_s3_rewriter.py \
      --s "${OUTPUT_DATADIR}/it4_s3_idxed_dedup.jsonl" \
      --o "${OUTPUT_DATADIR}/it4_s3_idxed_rewrite.jsonl" \
      --model "$MODEL"
  fi

  run_cmd python3 batch_s4_generator.py \
    --s "${OUTPUT_DATADIR}/it4_s3_idxed_rewrite.jsonl" \
    --o "${OUTPUT_DATADIR}/it4_s4.jsonl" \
    --api "$API_FILE" \
    --model "$MODEL" \
    "${TARGET_ARGS[@]}"

  run_cmd python3 dataset_integration.py --step s4 \
    --t1 "${OUTPUT_DATADIR}/it4_s3_idxed_rewrite.jsonl" \
    --t2 "${OUTPUT_DATADIR}/it4_s4.jsonl" \
    --o "${OUTPUT_DATADIR}/it5_s1_nonnr.jsonl" \
    --model "$MODEL"

  if [[ "$RUN_FILLING" == "1" ]]; then
    run_cmd python3 filling_datas.py \
      --s1 "${INPUT_DATADIR}/it1_s1_spare.jsonl" \
      --s2 "${OUTPUT_DATADIR}/it4_s2_idxed.jsonl" \
      --o "${OUTPUT_DATADIR}/it4_supplyments.jsonl"
  fi

  if [[ "$RUN_BALANCING" == "1" ]]; then
    run_cmd python3 balancing_data.py \
      --t1 "${OUTPUT_DATADIR}/it5_s1_nonnr.jsonl" \
      --t2 "${OUTPUT_DATADIR}/it4_supplyments.jsonl" \
      --o "${OUTPUT_DATADIR}/it5_s1.jsonl" \
      --model "$MODEL"
  fi

  if [[ "$RUN_FILTERING" == "1" ]]; then
    run_cmd python3 filter_every_interation.py \
      --t "${OUTPUT_DATADIR}/it5_s1.jsonl" \
      --o "logs/it5_s1_log_gem.jsonl" \
      --model "$MODEL" \
      --filter-model "$FILTER_MODEL"
  fi
}

echo "ROOT_DIR=$ROOT_DIR"
echo "TURN=$TURN"
echo "MODEL=$MODEL"
echo "FILTER_MODEL=$FILTER_MODEL"
echo "INPUT_DATADIR=$INPUT_DATADIR"
echo "OUTPUT_DATADIR=$OUTPUT_DATADIR"
echo "API_FILE=$API_FILE"
echo "TARGET_FILE=${TARGET_FILE:-<none>}"
echo "TARGET_COUNT=${TARGET_COUNT:-<none>}"
echo "OVERGEN_COUNT=${OVERGEN_COUNT:-<none>}"
echo "DRY_RUN=$DRY_RUN"
echo "RUN_DEDUP=$RUN_DEDUP RUN_REWRITE=$RUN_REWRITE RUN_FILLING=$RUN_FILLING RUN_BALANCING=$RUN_BALANCING RUN_FILTERING=$RUN_FILTERING"

case "$TURN" in
  2) run_turn_2 ;;
  3) run_turn_3 ;;
  4) run_turn_4 ;;
  5) run_turn_5 ;;
  *)
    echo "Unsupported TURN: $TURN" >&2
    exit 1
    ;;
esac
