#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
EMBED_MODEL="${EMBED_MODEL:-models/Qwen_Qwen3-Embedding-0.6B}"
LLM_MODEL="${LLM_MODEL:-models/Qwen_Qwen3-4B-Instruct-2507}"
SAMPLE_SIZE="${SAMPLE_SIZE:-3072}"
SEED="${SEED:-2026}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-64}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_EMBED="${STEP_TIMEOUT_EMBED:-7200}"
STEP_TIMEOUT_BUILD="${STEP_TIMEOUT_BUILD:-3600}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/samples

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/experiment_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/experiment_queue_status.tsv"
LOCK_PATH="outputs/queue_logs/experiment_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Queue already running with pid=$old_pid"
    exit 0
  fi
fi
echo "$$" > "$LOCK_PATH"
trap 'rm -f "$LOCK_PATH"' EXIT

log_status() {
  local name="$1"
  local status="$2"
  local detail="$3"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$name" "$status" "$detail" >> "$STATUS_PATH"
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

echo "Queue run id: $RUN_ID"
echo "Workspace: $WORKSPACE"
echo "Python: $PYTHON_BIN"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step "pytest" 1800 "" "$PYTHON_BIN" -m pytest -q || exit "$FAILURES"

run_step \
  "build_ucihar_hard_v3" \
  "$STEP_TIMEOUT_BUILD" \
  "data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" \
  "$PYTHON_BIN" scripts/build_hard_benchmark.py \
    --variant v3 \
    --input data/benchmark/ucihar_sensorfact_test.jsonl \
    --output data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl \
    --max-records -1 || exit "$FAILURES"

run_step \
  "build_wisdm_hard" \
  "$STEP_TIMEOUT_BUILD" \
  "data/benchmark/wisdm_sensorfact_hard_v2_test.jsonl,data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" \
  "$PYTHON_BIN" scripts/build_wisdm_benchmark.py \
    --download \
    --hard-variants v2,v3 \
    --max-train-windows -1 \
    --max-test-windows -1 || exit "$FAILURES"

BENCHMARKS=(
  "ucihar_hard_v2:data/benchmark/ucihar_sensorfact_hard_v2_test.jsonl"
  "ucihar_hard_v3:data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl"
  "wisdm_hard_v2:data/benchmark/wisdm_sensorfact_hard_v2_test.jsonl"
  "wisdm_hard_v3:data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl"
)

for item in "${BENCHMARKS[@]}"; do
  name="${item%%:*}"
  path="${item#*:}"
  sample_path="data/benchmark/samples/${name}_sample_${SAMPLE_SIZE}_seed${SEED}.jsonl"
  run_step \
    "sample_${name}" \
    900 \
    "$sample_path" \
    "$PYTHON_BIN" scripts/sample_jsonl.py \
      --input "$path" \
      --output "$sample_path" \
      --sample-size "$SAMPLE_SIZE" \
      --seed "$SEED" || continue

  run_step \
    "embedding_${name}_sample" \
    "$STEP_TIMEOUT_EMBED" \
    "outputs/qwen_embedding_${name}_sample_metrics.json" \
    "$PYTHON_BIN" scripts/run_qwen_embedding_eval.py \
      --benchmark-path "$sample_path" \
      --output-metrics "outputs/qwen_embedding_${name}_sample_metrics.json" \
      --output-scores "outputs/qwen_embedding_${name}_sample_scores.jsonl" \
      --max-records -1 \
      --batch-size 64 \
      --device cuda \
      --pairwise-margin 0.02

  run_step \
    "llm_${name}_sample" \
    "$STEP_TIMEOUT_LLM" \
    "outputs/qwen_llm_${name}_sample_metrics.json" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --benchmark-path "$sample_path" \
      --output-metrics "outputs/qwen_llm_${name}_sample_metrics.json" \
      --output-rows "outputs/qwen_llm_${name}_sample_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens "$LLM_MAX_NEW_TOKENS" \
      --device cuda

  run_step \
    "embedding_${name}_full" \
    "$STEP_TIMEOUT_EMBED" \
    "outputs/qwen_embedding_${name}_metrics.json" \
    "$PYTHON_BIN" scripts/run_qwen_embedding_eval.py \
      --benchmark-path "$path" \
      --output-metrics "outputs/qwen_embedding_${name}_metrics.json" \
      --output-scores "outputs/qwen_embedding_${name}_scores.jsonl" \
      --max-records -1 \
      --batch-size 64 \
      --device cuda \
      --pairwise-margin 0.02

  run_step \
    "llm_${name}_full" \
    "$STEP_TIMEOUT_LLM" \
    "outputs/qwen_llm_${name}_metrics.json" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --benchmark-path "$path" \
      --output-metrics "outputs/qwen_llm_${name}_metrics.json" \
      --output-rows "outputs/qwen_llm_${name}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens "$LLM_MAX_NEW_TOKENS" \
      --device cuda

  run_step \
    "error_analysis_${name}" \
    900 \
    "outputs/qwen_llm_${name}_error_by_field.csv" \
    bash -lc "$PYTHON_BIN scripts/analyze_hard_llm_errors.py --records '$path' --rows 'outputs/qwen_llm_${name}_rows.jsonl' > 'outputs/qwen_llm_${name}_error_by_field.csv'"

  run_step \
    "audit_after_${name}" \
    900 \
    "" \
    "$PYTHON_BIN" scripts/audit_experiment_outputs.py \
      --workspace "$WORKSPACE" \
      --output outputs/experiment_audit_report.md
done

run_step \
  "final_audit" \
  900 \
  "" \
  "$PYTHON_BIN" scripts/audit_experiment_outputs.py \
    --workspace "$WORKSPACE" \
    --output outputs/experiment_audit_report.md

echo "[$(date -Is)] QUEUE FINISHED run_id=$RUN_ID"
nvidia-smi || true
exit "$FAILURES"
