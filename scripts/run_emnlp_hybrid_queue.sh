#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_hybrid_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_hybrid_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_hybrid_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_hybrid_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Hybrid queue already running with pid=$old_pid"
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

run_hybrid() {
  local name="$1"
  local structured_rows="$2"
  local direct_rows="$3"
  local output_prefix="outputs/${name}"
  run_step \
    "$name" \
    "$STEP_TIMEOUT_SHORT" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_hybrid_verifier_eval.py \
      --workspace "$WORKSPACE" \
      --structured-rows "$structured_rows" \
      --direct-rows "$direct_rows" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --system-name "$name" || true
}

echo "EMNLP hybrid queue run id: $RUN_ID"

run_step \
  "pytest_hybrid_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_hybrid_verifier_eval.py tests/test_emnlp_hybrid_queue.py -q || true

run_hybrid hybrid_regex_qwen_ucihar_hard_v3 outputs/axisfix_structured_regex_ucihar_hard_v3_rows.jsonl outputs/qwen_llm_ucihar_hard_v3_rows.jsonl
run_hybrid hybrid_regex_qwen_wisdm_hard_v3 outputs/axisfix_structured_regex_wisdm_hard_v3_rows.jsonl outputs/qwen_llm_wisdm_hard_v3_rows.jsonl
run_hybrid hybrid_regex_qwen_mhealth_hard_v3 outputs/axisfix_structured_regex_mhealth_hard_v3_rows.jsonl outputs/qwen_llm_mhealth_hard_v3_rows.jsonl

run_hybrid hybrid_regex_coder_ucihar_hard_v3 outputs/axisfix_structured_regex_ucihar_hard_v3_rows.jsonl outputs/coder_llm_ucihar_hard_v3_rows.jsonl
run_hybrid hybrid_regex_coder_wisdm_hard_v3 outputs/axisfix_structured_regex_wisdm_hard_v3_rows.jsonl outputs/coder_llm_wisdm_hard_v3_rows.jsonl
run_hybrid hybrid_regex_coder_mhealth_hard_v3 outputs/axisfix_structured_regex_mhealth_hard_v3_rows.jsonl outputs/coder_llm_mhealth_hard_v3_rows.jsonl

run_step \
  "emnlp_report_after_hybrid_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP HYBRID QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
exit "$FAILURES"
