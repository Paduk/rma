#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-o4-mini}" # o4-mini-2025-04-16
DATADIR="${DATADIR:-difficult_o4}"
API_FILE="${API_FILE:-apis/api_v3.0.1.jsonl}"
TURN="${TURN:-}"

RUN_TURN_1="${RUN_TURN_1:-0}"
RUN_TURN_2="${RUN_TURN_2:-0}"
RUN_TURN_3="${RUN_TURN_3:-0}"
RUN_TURN_4="${RUN_TURN_4:-0}"
RUN_TURN_5="${RUN_TURN_5:-0}"

mkdir -p "$DATADIR" logs

run_cmd() {
  echo "+ $*"
  "$@"
}

run_turn_1() {
  run_cmd python3 s1_generator.py --o "${DATADIR}/it1_s1.jsonl" --api "${API_FILE}" --model "$MODEL"
}

run_turn_2() {
  run_cmd python3 batch_s2_generator.py --s "${DATADIR}/it1_s1.jsonl" --o "${DATADIR}/it1_s2.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s2 --t1 "${DATADIR}/it1_s1.jsonl" --t2 "${DATADIR}/it1_s2.jsonl" --o "${DATADIR}/it1_s2_idxed.jsonl"
  run_cmd python3 batch_s3_generator.py --s "${DATADIR}/it1_s2.jsonl" --o "${DATADIR}/it1_s3.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s3 --t1 "${DATADIR}/it1_s2_idxed.jsonl" --t2 "${DATADIR}/it1_s3.jsonl" --o "${DATADIR}/it1_s3_idxed.jsonl" --model "$MODEL"
  run_cmd python3 dataset_spliter.py --t_list "${DATADIR}/it1_s3_idxed.jsonl"
  run_cmd python3 batch_s3_rewriter.py --s "${DATADIR}/it1_s3_idxed_dedup.jsonl" --o "${DATADIR}/it1_s3_idxed_rewrite.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it1_s3_idxed_rewrite.jsonl" --o "${DATADIR}/it1_s4.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 filling_datas.py --s1 "${DATADIR}/it1_s1_spare.jsonl" --s2 "${DATADIR}/it1_s2_idxed.jsonl" --o "${DATADIR}/it1_supplyments.jsonl"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it1_s3_idxed_rewrite.jsonl" --t2 "${DATADIR}/it1_s4.jsonl" --o "${DATADIR}/it2_s1_nonnr.jsonl" --model "$MODEL"
  run_cmd python3 balancing_data.py --t1 "${DATADIR}/it2_s1_nonnr.jsonl" --t2 "${DATADIR}/it1_supplyments.jsonl" --o "${DATADIR}/it2_s1.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it2_s1.jsonl" --o "logs/it2_s1_log.jsonl" --model "$MODEL"
}

run_turn_3() {
  run_cmd python3 batch_s2_generator.py --s "${DATADIR}/it2_s1_filtered.jsonl" --o "${DATADIR}/it2_s2.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s2 --t1 "${DATADIR}/it2_s1_filtered.jsonl" --t2 "${DATADIR}/it2_s2.jsonl" --o "${DATADIR}/it2_s2_idxed.jsonl"
  run_cmd python3 batch_s3_generator.py --s "${DATADIR}/it2_s2.jsonl" --o "${DATADIR}/it2_s3.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s3 --t1 "${DATADIR}/it2_s2_idxed.jsonl" --t2 "${DATADIR}/it2_s3.jsonl" --o "${DATADIR}/it2_s3_idxed.jsonl" --model "$MODEL"
  run_cmd python3 dataset_spliter.py --t_list "${DATADIR}/it1_s3_idxed_rewrite.jsonl" "${DATADIR}/it2_s3_idxed.jsonl"
  run_cmd python3 batch_s3_rewriter.py --s "${DATADIR}/it2_s3_idxed_dedup.jsonl" --o "${DATADIR}/it2_s3_idxed_rewrite.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it2_s3_idxed_rewrite.jsonl" --o "${DATADIR}/it2_s4.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 filling_datas.py --s1 "${DATADIR}/it1_s1_spare.jsonl" --s2 "${DATADIR}/it2_s2_idxed.jsonl" --o "${DATADIR}/it2_supplyments.jsonl"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it2_s3_idxed_rewrite.jsonl" --t2 "${DATADIR}/it2_s4.jsonl" --o "${DATADIR}/it3_s1_nonnr.jsonl" --model "$MODEL"
  run_cmd python3 balancing_data.py --t1 "${DATADIR}/it3_s1_nonnr.jsonl" --t2 "${DATADIR}/it2_supplyments.jsonl" --o "${DATADIR}/it3_s1.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it3_s1.jsonl" --o "logs/it3_s1_log_gem.jsonl" --model "$MODEL"
}

run_turn_4() {
  run_cmd python3 batch_s2_generator.py --s "${DATADIR}/it3_s1_filtered.jsonl" --o "${DATADIR}/it3_s2.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s2 --t1 "${DATADIR}/it3_s1_filtered.jsonl" --t2 "${DATADIR}/it3_s2.jsonl" --o "${DATADIR}/it3_s2_idxed.jsonl"
  run_cmd python3 batch_s3_generator.py --s "${DATADIR}/it3_s2.jsonl" --o "${DATADIR}/it3_s3.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s3 --t1 "${DATADIR}/it3_s2_idxed.jsonl" --t2 "${DATADIR}/it3_s3.jsonl" --o "${DATADIR}/it3_s3_idxed.jsonl" --model "$MODEL"
  run_cmd python3 dataset_spliter.py --t_list "${DATADIR}/it1_s3_idxed_rewrite_dedup.jsonl" "${DATADIR}/it2_s3_idxed_rewrite.jsonl" "${DATADIR}/it3_s3_idxed.jsonl"
  run_cmd python3 batch_s3_rewriter.py --s "${DATADIR}/it3_s3_idxed_dedup.jsonl" --o "${DATADIR}/it3_s3_idxed_rewrite.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it3_s3_idxed_rewrite.jsonl" --o "${DATADIR}/it3_s4.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 filling_datas.py --s1 "${DATADIR}/it1_s1_spare.jsonl" --s2 "${DATADIR}/it3_s2_idxed.jsonl" --o "${DATADIR}/it3_supplyments.jsonl"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it3_s3_idxed_rewrite.jsonl" --t2 "${DATADIR}/it3_s4.jsonl" --o "${DATADIR}/it4_s1_nonnr.jsonl" --model "$MODEL"
  run_cmd python3 balancing_data.py --t1 "${DATADIR}/it4_s1_nonnr.jsonl" --t2 "${DATADIR}/it3_supplyments.jsonl" --o "${DATADIR}/it4_s1.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it4_s1.jsonl" --o "logs/it4_s1_log_gem.jsonl" --model "$MODEL"
}

run_turn_5() {
  run_cmd python3 batch_s2_generator.py --s "${DATADIR}/it4_s1_filtered.jsonl" --o "${DATADIR}/it4_s2.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s2 --t1 "${DATADIR}/it4_s1_filtered.jsonl" --t2 "${DATADIR}/it4_s2.jsonl" --o "${DATADIR}/it4_s2_idxed.jsonl"
  run_cmd python3 batch_s3_generator_difficult.py --s "${DATADIR}/it4_s2.jsonl" --o "${DATADIR}/it4_s3.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s3 --t1 "${DATADIR}/it4_s2_idxed.jsonl" --t2 "${DATADIR}/it4_s3.jsonl" --o "${DATADIR}/it4_s3_idxed.jsonl" --model "$MODEL"
  run_cmd python3 dataset_spliter.py --t_list "${DATADIR}/it1_s3_idxed_rewrite_dedup_dedup.jsonl" "${DATADIR}/it2_s3_idxed_rewrite_dedup.jsonl" "${DATADIR}/it3_s3_idxed_rewrite.jsonl" "${DATADIR}/it4_s3_idxed.jsonl"
  run_cmd python3 batch_s3_rewriter.py --s "${DATADIR}/it4_s3_idxed_dedup.jsonl" --o "${DATADIR}/it4_s3_idxed_rewrite.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it4_s3_idxed_rewrite.jsonl" --o "${DATADIR}/it4_s4.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 filling_datas.py --s1 "${DATADIR}/it1_s1_spare.jsonl" --s2 "${DATADIR}/it4_s2_idxed.jsonl" --o "${DATADIR}/it4_supplyments.jsonl"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it4_s3_idxed_rewrite.jsonl" --t2 "${DATADIR}/it4_s4.jsonl" --o "${DATADIR}/it5_s1_nonnr.jsonl" --model "$MODEL"
  run_cmd python3 balancing_data.py --t1 "${DATADIR}/it5_s1_nonnr.jsonl" --t2 "${DATADIR}/it4_supplyments.jsonl" --o "${DATADIR}/it5_s1.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it5_s1.jsonl" --o "logs/it5_s1_log_gem.jsonl" --model "$MODEL"
}

echo "ROOT_DIR=$ROOT_DIR"
echo "MODEL=$MODEL"
echo "DATADIR=$DATADIR"
echo "API_FILE=$API_FILE"
echo "TURN=${TURN:-<all>}"
echo "RUN_TURN_1=$RUN_TURN_1 RUN_TURN_2=$RUN_TURN_2 RUN_TURN_3=$RUN_TURN_3 RUN_TURN_4=$RUN_TURN_4 RUN_TURN_5=$RUN_TURN_5"

if [[ -z "$TURN" && "$RUN_TURN_1" != "1" && "$RUN_TURN_2" != "1" && "$RUN_TURN_3" != "1" && "$RUN_TURN_4" != "1" && "$RUN_TURN_5" != "1" ]]; then
  RUN_TURN_1=1
  RUN_TURN_2=1
  RUN_TURN_3=1
  RUN_TURN_4=1
  RUN_TURN_5=1
fi

if [[ "$RUN_TURN_1" == "1" ]]; then
  run_turn_1
fi
if [[ "$RUN_TURN_2" == "1" ]]; then
  run_turn_2
fi
if [[ "$RUN_TURN_3" == "1" ]]; then
  run_turn_3
fi
if [[ "$RUN_TURN_4" == "1" ]]; then
  run_turn_4
fi
if [[ "$RUN_TURN_5" == "1" ]]; then
  run_turn_5
fi

if [[ -n "$TURN" ]]; then
  case "$TURN" in
    1) [[ "$RUN_TURN_1" != "1" ]] && run_turn_1 ;;
    2) [[ "$RUN_TURN_2" != "1" ]] && run_turn_2 ;;
    3) [[ "$RUN_TURN_3" != "1" ]] && run_turn_3 ;;
    4) [[ "$RUN_TURN_4" != "1" ]] && run_turn_4 ;;
    5) [[ "$RUN_TURN_5" != "1" ]] && run_turn_5 ;;
    *) echo "Unsupported TURN: $TURN" >&2; exit 1 ;;
  esac
fi
