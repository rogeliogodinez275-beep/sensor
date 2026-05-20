#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_support_ablation_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_support_ablation_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_support_ablation_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_support_ablation_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Support ablation queue already running with pid=$old_pid"
    exit 0
  fi
fi
echo "$$" > "$LOCK_PATH"
trap 'rm -f "$LOCK_PATH"' EXIT

log_status() {
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" >> "$STATUS_PATH"
}

outputs_ready() {
  local expected_outputs="$1"
  if [[ -z "$expected_outputs" ]]; then
    return 1
  fi
  IFS=',' read -r -a paths <<< "$expected_outputs"
  local candidate_path
  for candidate_path in "${paths[@]}"; do
    if [[ ! -s "$candidate_path" ]]; then
      return 1
    fi
  done
  return 0
}

wait_for_llm_slot() {
  local own_pid="$$"
  while pgrep -af "scripts/run_qwen_llm_eval.py" | awk -v own="$own_pid" '$1 != own {found=1} END {exit found ? 0 : 1}'; do
    echo "[$(date -Is)] Waiting for active run_qwen_llm_eval.py before support ablation job"
    sleep "$WAIT_SECONDS"
  done
}

run_step() {
  local name="$1"
  local timeout_seconds="$2"
  local expected_outputs="$3"
  shift 3
  if outputs_ready "$expected_outputs"; then
    echo "[$(date -Is)] SKIP $name expected=$expected_outputs"
    log_status "$name" "SKIP" "$expected_outputs"
    return 0
  fi
  echo "[$(date -Is)] START $name timeout=${timeout_seconds}s"
  log_status "$name" "START" "timeout=${timeout_seconds}s"
  timeout "${timeout_seconds}s" "$@"
  local code=$?
  if [[ $code -eq 0 ]]; then
    echo "[$(date -Is)] DONE $name"
    log_status "$name" "DONE" "exit=0"
  elif [[ $code -eq 124 ]]; then
    echo "[$(date -Is)] TIMEOUT $name"
    log_status "$name" "TIMEOUT" "exit=124"
    FAILURES=$((FAILURES + 1))
  else
    echo "[$(date -Is)] FAIL $name exit=$code"
    log_status "$name" "FAIL" "exit=$code"
    FAILURES=$((FAILURES + 1))
  fi
  return "$code"
}

run_llm() {
  local name="$1"
  local output_prefix="$2"
  local support_seed="$3"
  local negative_count="$4"
  local extra_args=(--support-seed "$support_seed")
  if [[ "$negative_count" != "all" ]]; then
    extra_args+=(--support-negative-count "$negative_count")
  fi
  wait_for_llm_slot
  run_step \
    "$name" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --workspace "$WORKSPACE" \
      --benchmark-path data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 96 \
      --device cuda \
      --prompt-style strict_json \
      "${extra_args[@]}" || true
}

echo "EMNLP support ablation queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step \
  "pytest_support_ablation" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_qwen_llm_eval.py tests/test_emnlp_support_ablation_queue.py -q || true

run_llm \
  "llm_mhealth_hard_v3_support_order_seed7001" \
  "outputs/qwen_llm_mhealth_hard_v3_support_order_seed7001" \
  7001 \
  all
# Explicit output: outputs/qwen_llm_mhealth_hard_v3_support_order_seed7001_metrics.json.

run_llm \
  "llm_mhealth_hard_v3_support_order_seed7002" \
  "outputs/qwen_llm_mhealth_hard_v3_support_order_seed7002" \
  7002 \
  all
# Explicit output: outputs/qwen_llm_mhealth_hard_v3_support_order_seed7002_metrics.json.

run_llm \
  "llm_mhealth_hard_v3_support_balanced_neg2" \
  "outputs/qwen_llm_mhealth_hard_v3_support_balanced_neg2" \
  42 \
  2
# Explicit output: outputs/qwen_llm_mhealth_hard_v3_support_balanced_neg2_metrics.json.

run_llm \
  "llm_mhealth_hard_v3_support_balanced_neg3" \
  "outputs/qwen_llm_mhealth_hard_v3_support_balanced_neg3" \
  42 \
  3

run_step \
  "emnlp_report_after_support_ablation" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP SUPPORT ABLATION QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
