#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MODEL_DIR="${MODEL_DIR:-models/Qwen_Qwen2.5-Coder-7B-Instruct}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs models

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_coder_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_coder_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_coder_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_coder_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Coder queue already running with pid=$old_pid"
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
  while ps -ef | grep -E "scripts/run_(qwen_llm_eval|structured_verifier_eval)\.py" | grep -v grep >/dev/null; do
    echo "[$(date -Is)] Waiting for active LLM job before coder step"
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

run_download() {
  run_step \
    "download_coder_model" \
    "$STEP_TIMEOUT_LLM" \
    "$MODEL_DIR/config.json,$MODEL_DIR/model.safetensors.index.json,$MODEL_DIR/model-00001-of-00004.safetensors,$MODEL_DIR/model-00002-of-00004.safetensors,$MODEL_DIR/model-00003-of-00004.safetensors,$MODEL_DIR/model-00004-of-00004.safetensors" \
    "$PYTHON_BIN" scripts/download_hf_snapshot.py \
      --repo-id "$MODEL_ID" \
      --local-dir "$MODEL_DIR" \
      --resume-download \
      --max-attempts 240 \
      --retry-delay-s 10
}

run_direct() {
  local dataset_name="$1"
  local benchmark_path="$2"
  local output_prefix="outputs/coder_llm_${dataset_name}_hard_v3"
  # Explicit step name: coder_llm_${dataset_name}_hard_v3
  wait_for_llm_slot
  run_step \
    "coder_llm_${dataset_name}_hard_v3" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --workspace "$WORKSPACE" \
      --model-dir "$MODEL_DIR" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 96 \
      --device cuda \
      --prompt-style strict_json || true
}

run_structured() {
  local dataset_name="$1"
  local benchmark_path="$2"
  local output_prefix="outputs/coder_structured_model_${dataset_name}_hard_v3"
  # Explicit step name: coder_structured_model_${dataset_name}_hard_v3
  wait_for_llm_slot
  run_step \
    "coder_structured_model_${dataset_name}_hard_v3" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_structured_verifier_eval.py \
      --workspace "$WORKSPACE" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --parser-mode model_evidence \
      --model-dir "$MODEL_DIR" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 96 \
      --device cuda || true
}

echo "EMNLP coder queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step \
  "pytest_coder_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_emnlp_coder_queue.py tests/test_qwen_llm_eval.py tests/test_structured_verifier_eval.py -q || true

run_download || exit "$FAILURES"

run_direct ucihar data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl
run_structured ucihar data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl
run_direct wisdm data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl
run_structured wisdm data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl
run_direct mhealth data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl
run_structured mhealth data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl

run_step \
  "emnlp_report_after_coder_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP CODER QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
