#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STATUS_PATH="outputs/queue_logs/emnlp_axisfix_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_axisfix_queue.lock"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs
LOG_PATH="outputs/queue_logs/relay_after_axisfix.log"
exec >> "$LOG_PATH" 2>&1

seen_queue=0
relay_start_epoch="$(date +%s)"

status_is_fresh() {
  [[ -f "$STATUS_PATH" ]] || return 1
  local status_mtime
  status_mtime="$(stat -c %Y "$STATUS_PATH" 2>/dev/null || echo 0)"
  [[ "$status_mtime" -ge "$relay_start_epoch" ]]
}

echo "[$(date -Is)] relay started: waiting for axisfix queue"
while true; do
  if [[ $seen_queue -eq 0 ]]; then
    if ps -ef | grep -E "scripts/run_emnlp_axisfix_queue.sh" | grep -v grep >/dev/null; then
      seen_queue=1
      echo "[$(date -Is)] observed active axisfix queue"
    elif [[ -f "$LOCK_PATH" ]]; then
      seen_queue=1
      echo "[$(date -Is)] observed axisfix lock"
    elif status_is_fresh && grep -q $'\tSTART\t' "$STATUS_PATH" 2>/dev/null; then
      seen_queue=1
      echo "[$(date -Is)] observed fresh axisfix status file"
    fi
  fi
  if [[ $seen_queue -eq 0 ]]; then
    echo "[$(date -Is)] axisfix not started yet; sleep ${WAIT_SECONDS}s"
    sleep "$WAIT_SECONDS"
    continue
  fi
  if ps -ef | grep -E "scripts/run_emnlp_axisfix_queue.sh|scripts/run_structured_verifier_eval.py" | grep -v grep >/dev/null; then
    echo "[$(date -Is)] axisfix still running; sleep ${WAIT_SECONDS}s"
    sleep "$WAIT_SECONDS"
    continue
  fi
  break
done

echo "[$(date -Is)] axisfix finished or idle; starting hybrid queue"
bash scripts/run_emnlp_hybrid_queue.sh || true

echo "[$(date -Is)] hybrid queue finished; starting coder prompt queue"
bash scripts/run_emnlp_coder_prompt_queue.sh || true

echo "[$(date -Is)] relay finished"
