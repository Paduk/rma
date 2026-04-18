#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INTERVAL="${INTERVAL:-30}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  ./run_rewrite_eval_after_idle.sh [--interval SECONDS] [--dry_run]

Waits until each target Ollama host has no active runner process, checking every
30 seconds by default, then runs the rewrite-mode evaluation assigned to that
host. The jobs are watched concurrently.

Options:
  --interval SECONDS   Poll interval. Default: 30
  --dry_run            Print commands without executing them.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      INTERVAL="$2"; shift 2 ;;
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

if ! [[ "$INTERVAL" =~ ^[0-9]+$ ]] || [[ "$INTERVAL" -lt 1 ]]; then
  echo "--interval must be a positive integer: $INTERVAL" >&2
  exit 2
fi

normalize_host() {
  local host="$1"
  host="${host#http://}"
  host="${host#https://}"
  host="${host%/}"
  if [[ "$host" == localhost:* ]]; then
    host="127.0.0.1:${host#localhost:}"
  fi
  echo "$host"
}

serve_pids_for_host() {
  local wanted
  wanted="$(normalize_host "$1")"

  local pid env_host normalized_env_host
  for pid in $(pgrep -u "$USER" -f 'ollama serve' || true); do
    if [[ ! -r "/proc/${pid}/environ" ]]; then
      continue
    fi
    env_host="$(
      tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null \
        | awk -F= '$1 == "OLLAMA_HOST" {print $2; exit}'
    )"
    if [[ -z "$env_host" ]]; then
      continue
    fi
    normalized_env_host="$(normalize_host "$env_host")"
    if [[ "$normalized_env_host" == "$wanted" ]]; then
      echo "$pid"
    fi
  done
}

runner_pids_for_host() {
  local host="$1"
  local serve_pid
  for serve_pid in $(serve_pids_for_host "$host"); do
    pgrep -P "$serve_pid" -f 'ollama runner' || true
  done
}

runner_ports_for_pids() {
  local pid
  for pid in "$@"; do
    ps -o args= -p "$pid" 2>/dev/null \
      | sed -nE 's/.* --port ([0-9]+).*/\1/p'
  done
}

wait_for_host_idle() {
  local label="$1"
  local host="$2"
  local normalized_host
  normalized_host="$(normalize_host "$host")"

  while true; do
    mapfile -t runner_pids < <(runner_pids_for_host "$normalized_host")
    if [[ "${#runner_pids[@]}" -eq 0 ]]; then
      echo "[$(date '+%F %T')] ${label}: ${normalized_host} is idle."
      return 0
    fi

    mapfile -t runner_ports < <(runner_ports_for_pids "${runner_pids[@]}")
    echo "[$(date '+%F %T')] ${label}: waiting for ${normalized_host}; runner_pids=${runner_pids[*]} runner_ports=${runner_ports[*]}"
    ps -o pid,ppid,etime,stat,pcpu,pmem,args -p "$(IFS=,; echo "${runner_pids[*]}")" || true
    sleep "$INTERVAL"
  done
}

run_job() {
  local label="$1"
  local host="$2"
  shift 2

  wait_for_host_idle "$label" "$host"
  echo "[$(date '+%F %T')] ${label}: running command"
  echo "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_job "phi4 rewrite" "127.0.0.1:11436" \
  python3 ollama_inference_multi.py \
    --t rewrite-phi4 \
    --test_key base,complex \
    --model phi4-phi-rewrite-all_linear:latest \
    --o datasets/result/phi4/phi4-rewrite-all_linear.tsv \
    --host http://localhost:11436 &

run_job "glm-edge-1.5b rewrite" "127.0.0.1:11438" \
  python3 ollama_inference_multi.py \
    --t rewrite-glm-edge-1.5b \
    --test_key base,complex \
    --model glm-edge-1-5b-chat-glm-edge-1.5b-rewrite-all_linear:latest \
    --o datasets/result/glm-1.5/glm-1.5-rewrite-all_linear.tsv \
    --host http://localhost:11438 &

run_job "smollm3-3b rewrite" "127.0.0.1:11440" \
  python3 ollama_inference_multi.py \
    --t rewrite-smollm3-3b \
    --test_key base,complex \
    --model smollm3-3b-smollm3-3b-rewrite-all_linear:latest \
    --o datasets/result/smollm3/smollm3-rewrite-all_linear.tsv \
    --host http://localhost:11440 &

run_job "granite3.3-2b rewrite" "127.0.0.1:11437" \
  python3 ollama_inference_multi.py \
    --t rewrite-granite3.3-2b \
    --test_key base,complex \
    --model granite-3-3-2b-instruct-granite3.3-2b-rewrite-all_linear:latest \
    --o datasets/result/granite/granite-rewrite-all_linear.tsv \
    --host http://localhost:11437 &

wait
