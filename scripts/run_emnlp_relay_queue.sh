#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1024}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/stress data/benchmark/samples

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_relay_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_relay_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_relay_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "EMNLP relay queue already running with pid=$old_pid"
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

wait_for_llm_slot() {
  local own_pid="$$"
  while pgrep -af "scripts/run_qwen_llm_eval.py" | awk -v own="$own_pid" '$1 != own {found=1} END {exit found ? 0 : 1}'; do
    echo "[$(date -Is)] Waiting for active run_qwen_llm_eval.py before next relay job"
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

run_llm_step() {
  wait_for_llm_slot
  run_step "$@"
}

echo "EMNLP relay queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step "pytest_relay" "$STEP_TIMEOUT_SHORT" "" "$PYTHON_BIN" -m pytest -q || true

DATASETS=(
  "ucihar_hard_v3:data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl"
  "wisdm_hard_v3:data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl"
)
PROMPT_STYLES=(terse chain_then_json)
STRESS_VARIANTS=(numeric_mask shuffled_evidence hidden_evidence)

for item in "${DATASETS[@]}"; do
  dataset_name="${item%%:*}"
  dataset_path="${item#*:}"

  for prompt_style in "${PROMPT_STYLES[@]}"; do
    run_llm_step \
      "llm_${dataset_name}_full_prompt_${prompt_style}" \
      "$STEP_TIMEOUT_LLM" \
      "outputs/qwen_llm_${dataset_name}_full_prompt_${prompt_style}_metrics.json,outputs/qwen_llm_${dataset_name}_full_prompt_${prompt_style}_rows.jsonl" \
      "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
        --benchmark-path "$dataset_path" \
        --output-metrics "outputs/qwen_llm_${dataset_name}_full_prompt_${prompt_style}_metrics.json" \
        --output-rows "outputs/qwen_llm_${dataset_name}_full_prompt_${prompt_style}_rows.jsonl" \
        --max-records -1 \
        --batch-size "$LLM_BATCH_SIZE" \
        --max-new-tokens 128 \
        --device cuda \
        --prompt-style "$prompt_style" || true
  done

  sample_path="data/benchmark/samples/${dataset_name}_sample_${SAMPLE_SIZE}_seed3031.jsonl"
  run_step \
    "sample_${dataset_name}_stress_seed3031" \
    "$STEP_TIMEOUT_SHORT" \
    "$sample_path" \
    "$PYTHON_BIN" scripts/sample_jsonl.py \
      --input "$dataset_path" \
      --output "$sample_path" \
      --sample-size "$SAMPLE_SIZE" \
      --seed 3031 || true

  for variant in "${STRESS_VARIANTS[@]}"; do
    stress_path="data/benchmark/stress/${dataset_name}_${variant}_sample_${SAMPLE_SIZE}_seed3031.jsonl"
    run_step \
      "build_${dataset_name}_${variant}" \
      "$STEP_TIMEOUT_SHORT" \
      "$stress_path" \
      "$PYTHON_BIN" scripts/build_stress_benchmark.py \
        --input "$sample_path" \
        --output "$stress_path" \
        --variant "$variant" \
        --seed 3031 || true

    run_llm_step \
      "llm_${dataset_name}_${variant}_sample" \
      "$STEP_TIMEOUT_LLM" \
      "outputs/qwen_llm_${dataset_name}_${variant}_sample_metrics.json,outputs/qwen_llm_${dataset_name}_${variant}_sample_rows.jsonl" \
      "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
        --benchmark-path "$stress_path" \
        --output-metrics "outputs/qwen_llm_${dataset_name}_${variant}_sample_metrics.json" \
        --output-rows "outputs/qwen_llm_${dataset_name}_${variant}_sample_rows.jsonl" \
        --max-records -1 \
        --batch-size "$LLM_BATCH_SIZE" \
        --max-new-tokens 96 \
        --device cuda \
        --prompt-style strict_json || true
  done
done

run_step \
  "emnlp_experiment_report_after_relay" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP RELAY QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
