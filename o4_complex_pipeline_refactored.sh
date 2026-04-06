#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-o4-mini}" # o4-mini-2025-04-16
DATADIR="${DATADIR:-o4_datagen}"
API_FILE="${API_FILE:-apis/api_v3.0.1.jsonl}"
TURN="${TURN:-}"

RUN_TURN_3_COMPLEX="${RUN_TURN_3_COMPLEX:-0}"
RUN_TURN_4_COMPLEX="${RUN_TURN_4_COMPLEX:-0}"
RUN_TURN_5_COMPLEX="${RUN_TURN_5_COMPLEX:-0}"

mkdir -p "$DATADIR" logs

run_cmd() {
  echo "+ $*"
  "$@"
}

run_turn_3_complex() {
  run_cmd python3 difficulty_s3_generator.py --t "${DATADIR}/it3_s1_filtered.jsonl" --api "${API_FILE}" --o "${DATADIR}/it2_s3_complex.jsonl" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step complex_s3 --t1 "${DATADIR}/it2_s2_idxed.jsonl" --t2 "${DATADIR}/it2_s3_complex.jsonl" --o "${DATADIR}/it2_s3_complex_idxed.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it2_s3_complex_idxed.jsonl" --o "${DATADIR}/it2_s4_complex.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it2_s3_complex_idxed.jsonl" --t2 "${DATADIR}/it2_s4_complex.jsonl" --o "${DATADIR}/it3_s1_complex.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it3_s1_complex.jsonl" --o "logs/it4_s1_log.jsonl" --model "$MODEL"
}

run_turn_4_complex() {
  run_cmd python3 difficulty_s3_generator.py --t "${DATADIR}/it4_s1_filtered.jsonl" --api "${API_FILE}" --o "${DATADIR}/it3_s3_complex.jsonl" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step complex_s3 --t1 "${DATADIR}/it3_s2_idxed.jsonl" --t2 "${DATADIR}/it3_s3_complex.jsonl" --o "${DATADIR}/it3_s3_complex_idxed.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it3_s3_complex_idxed.jsonl" --o "${DATADIR}/it3_s4_complex.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it3_s3_complex_idxed.jsonl" --t2 "${DATADIR}/it3_s4_complex.jsonl" --o "${DATADIR}/it4_s1_complex.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it4_s1_complex.jsonl" --o "logs/it5_s1_log.jsonl" --model "$MODEL"
}

run_turn_5_complex() {
  run_cmd python3 difficulty_s3_generator.py --t "${DATADIR}/it5_s1_filtered.jsonl" --api "${API_FILE}" --o "${DATADIR}/it4_s3_complex.jsonl" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step complex_s3 --t1 "${DATADIR}/it4_s2_idxed.jsonl" --t2 "${DATADIR}/it4_s3_complex.jsonl" --o "${DATADIR}/it4_s3_complex_idxed.jsonl" --model "$MODEL"
  run_cmd python3 batch_s4_generator.py --s "${DATADIR}/it4_s3_complex_idxed.jsonl" --o "${DATADIR}/it4_s4_complex.jsonl" --api "${API_FILE}" --model "$MODEL"
  run_cmd python3 dataset_integration.py --step s4 --t1 "${DATADIR}/it4_s3_complex_idxed.jsonl" --t2 "${DATADIR}/it4_s4_complex.jsonl" --o "${DATADIR}/it5_s1_complex.jsonl" --model "$MODEL"
  run_cmd python3 filter_every_interation.py --t "${DATADIR}/it5_s1_complex.jsonl" --o "logs/it5_s1_log.jsonl" --model "$MODEL"
}

echo "ROOT_DIR=$ROOT_DIR"
echo "MODEL=$MODEL"
echo "DATADIR=$DATADIR"
echo "API_FILE=$API_FILE"
echo "TURN=${TURN:-<all>}"
echo "RUN_TURN_3_COMPLEX=$RUN_TURN_3_COMPLEX RUN_TURN_4_COMPLEX=$RUN_TURN_4_COMPLEX RUN_TURN_5_COMPLEX=$RUN_TURN_5_COMPLEX"

if [[ -z "$TURN" && "$RUN_TURN_3_COMPLEX" != "1" && "$RUN_TURN_4_COMPLEX" != "1" && "$RUN_TURN_5_COMPLEX" != "1" ]]; then
  RUN_TURN_3_COMPLEX=1
  RUN_TURN_4_COMPLEX=1
  RUN_TURN_5_COMPLEX=1
fi

if [[ "$RUN_TURN_3_COMPLEX" == "1" ]]; then
  run_turn_3_complex
fi
if [[ "$RUN_TURN_4_COMPLEX" == "1" ]]; then
  run_turn_4_complex
fi
if [[ "$RUN_TURN_5_COMPLEX" == "1" ]]; then
  run_turn_5_complex
fi

if [[ -n "$TURN" ]]; then
  case "$TURN" in
    3) [[ "$RUN_TURN_3_COMPLEX" != "1" ]] && run_turn_3_complex ;;
    4) [[ "$RUN_TURN_4_COMPLEX" != "1" ]] && run_turn_4_complex ;;
    5) [[ "$RUN_TURN_5_COMPLEX" != "1" ]] && run_turn_5_complex ;;
    *) echo "Unsupported TURN: $TURN" >&2; exit 1 ;;
  esac
fi
