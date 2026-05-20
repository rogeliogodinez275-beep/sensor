#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
STATUS_PATH="outputs/queue_logs/emnlp_coder_hybrid_prompt_status.tsv"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs
LOG_PATH="outputs/queue_logs/relay_after_coder_hybrid_prompt.log"
exec >> "$LOG_PATH" 2>&1

seen_queue=0
echo "[$(date -Is)] relay started: waiting for coder hybrid prompt queue"
while true; do
  if [[ $seen_queue -eq 0 ]]; then
    if ps -ef | grep -E "scripts/run_emnlp_coder_hybrid_prompt_queue.sh" | grep -v grep >/dev/null; then
      seen_queue=1
      echo "[$(date -Is)] observed active coder hybrid prompt queue"
    elif [[ -f "$STATUS_PATH" ]] && grep -q $'\tSTART\t' "$STATUS_PATH" 2>/dev/null; then
      seen_queue=1
      echo "[$(date -Is)] observed coder hybrid prompt status file"
    fi
  fi
  if [[ $seen_queue -eq 0 ]]; then
    echo "[$(date -Is)] coder hybrid prompt not started yet; sleep ${WAIT_SECONDS}s"
    sleep "$WAIT_SECONDS"
    continue
  fi
  if ps -ef | grep -E "scripts/run_emnlp_coder_hybrid_prompt_queue.sh|scripts/run_qwen_llm_eval.py" | grep -v grep >/dev/null; then
    echo "[$(date -Is)] coder hybrid prompt still running; sleep ${WAIT_SECONDS}s"
    sleep "$WAIT_SECONDS"
    continue
  fi
  break
done

echo "[$(date -Is)] coder hybrid prompt finished or idle; starting coder fewshot queue"
bash scripts/run_emnlp_coder_fewshot_queue.sh || true

echo "[$(date -Is)] relay finished"
