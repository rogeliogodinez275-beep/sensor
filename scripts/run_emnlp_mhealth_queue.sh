#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_BUILD="${STEP_TIMEOUT_BUILD:-3600}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/samples

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_mhealth_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_mhealth_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_mhealth_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "MHEALTH queue already running with pid=$old_pid"
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

run_llm() {
  local name="$1"
  local benchmark_path="$2"
  local output_prefix="$3"
  local max_records="$4"
  local metrics_path="${output_prefix}_metrics.json"
  local rows_path="${output_prefix}_rows.jsonl"
  run_step \
    "$name" \
    "$STEP_TIMEOUT_LLM" \
    "${metrics_path},${rows_path}" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --workspace "$WORKSPACE" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "$metrics_path" \
      --output-rows "$rows_path" \
      --max-records "$max_records" \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 96 \
      --device cuda || true
}

echo "EMNLP MHEALTH queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step \
  "pytest_mhealth" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_mhealth_data.py tests/test_emnlp_mhealth_queue.py -q || true

run_step \
  "build_mhealth_hard" \
  "$STEP_TIMEOUT_BUILD" \
  "data/benchmark/mhealth_sensorfact_hard_v2_test.jsonl,data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" \
  "$PYTHON_BIN" scripts/build_mhealth_benchmark.py \
    --workspace "$WORKSPACE" \
    --download \
    --hard-variants v2,v3 \
    --max-train-windows -1 \
    --max-test-windows -1 || true

run_step \
  "sample_mhealth_hard_v3_1024" \
  "$STEP_TIMEOUT_SHORT" \
  "data/benchmark/samples/mhealth_hard_v3_sample_1024_seed2026.jsonl" \
  "$PYTHON_BIN" scripts/sample_jsonl.py \
    --input data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
    --output data/benchmark/samples/mhealth_hard_v3_sample_1024_seed2026.jsonl \
    --sample-size 1024 \
    --seed 2026 || true

run_llm \
  "llm_mhealth_hard_v3_sample_1024" \
  "data/benchmark/samples/mhealth_hard_v3_sample_1024_seed2026.jsonl" \
  "outputs/qwen_llm_mhealth_hard_v3_sample1024" \
  -1

# Explicit outputs: outputs/qwen_llm_mhealth_hard_v3_sample1024_metrics.json and outputs/qwen_llm_mhealth_hard_v3_sample1024_rows.jsonl.

run_step \
  "supervised_mhealth_hard_v3" \
  "$STEP_TIMEOUT_SHORT" \
  "outputs/supervised_mhealth_hard_v3_metrics.json,outputs/supervised_mhealth_hard_v3_rows.jsonl" \
  "$PYTHON_BIN" scripts/run_supervised_baseline.py \
    --workspace "$WORKSPACE" \
    --train-path data/benchmark/mhealth_sensorfact_train.jsonl \
    --eval-path data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
    --output-metrics outputs/supervised_mhealth_hard_v3_metrics.json \
    --output-rows outputs/supervised_mhealth_hard_v3_rows.jsonl \
    --feature-mode oracle_fields \
    --hard-variant v3 || true

run_step \
  "supervised_numeric_mhealth_hard_v3" \
  "$STEP_TIMEOUT_SHORT" \
  "outputs/supervised_numeric_mhealth_hard_v3_metrics.json,outputs/supervised_numeric_mhealth_hard_v3_rows.jsonl" \
  "$PYTHON_BIN" scripts/run_supervised_baseline.py \
    --workspace "$WORKSPACE" \
    --train-path data/benchmark/mhealth_sensorfact_train.jsonl \
    --eval-path data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
    --output-metrics outputs/supervised_numeric_mhealth_hard_v3_metrics.json \
    --output-rows outputs/supervised_numeric_mhealth_hard_v3_rows.jsonl \
    --feature-mode numeric_only \
    --hard-variant v3 || true

run_step \
  "emnlp_report_after_mhealth_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP MHEALTH QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
