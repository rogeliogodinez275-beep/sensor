#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_EMBED="${STEP_TIMEOUT_EMBED:-7200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/stress

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_baseline_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_baseline_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_baseline_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Baseline queue already running with pid=$old_pid"
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

wait_for_gpu_eval_slot() {
  while pgrep -af "scripts/run_qwen_(llm|embedding)_eval.py" >/dev/null; do
    echo "[$(date -Is)] Waiting for active Qwen eval process"
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

run_gpu_step() {
  wait_for_gpu_eval_slot
  run_step "$@"
}

run_embedding_full_stress() {
  local dataset_name="$1"
  local variant="$2"
  local input_path="data/benchmark/stress/${dataset_name}_${variant}_full.jsonl"
  local output_prefix="outputs/qwen_embedding_${dataset_name}_${variant}_full"
  run_gpu_step \
    "embedding_${dataset_name}_${variant}_full" \
    "$STEP_TIMEOUT_EMBED" \
    "${output_prefix}_metrics.json,${output_prefix}_scores.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_embedding_eval.py \
      --benchmark-path "$input_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-scores "${output_prefix}_scores.jsonl" \
      --max-records -1 \
      --batch-size "$EMBED_BATCH_SIZE" \
      --device cuda || true
}

run_hard_v2_prompt() {
  local dataset_name="$1"
  local benchmark_path="data/benchmark/${dataset_name}_sensorfact_hard_v2_test.jsonl"
  local prompt_style="$2"
  local output_prefix="outputs/qwen_llm_${dataset_name}_hard_v2_prompt_${prompt_style}"
  run_gpu_step \
    "llm_${dataset_name}_hard_v2_prompt_${prompt_style}" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 128 \
      --device cuda \
      --prompt-style "$prompt_style" || true
}

run_hard_v2_stress() {
  local dataset_name="$1"
  local variant="$2"
  local input_path="data/benchmark/${dataset_name}_sensorfact_hard_v2_test.jsonl"
  local stress_path="data/benchmark/stress/${dataset_name}_hard_v2_${variant}_full.jsonl"
  local output_prefix="outputs/qwen_llm_${dataset_name}_hard_v2_${variant}_full"
  run_step \
    "build_${dataset_name}_hard_v2_${variant}_full" \
    "$STEP_TIMEOUT_SHORT" \
    "$stress_path" \
    "$PYTHON_BIN" scripts/build_stress_benchmark.py \
      --input "$input_path" \
      --output "$stress_path" \
      --variant "$variant" \
      --seed 4041 || true
  run_gpu_step \
    "llm_${dataset_name}_hard_v2_${variant}_full" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --benchmark-path "$stress_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 96 \
      --device cuda || true
}

echo "EMNLP baseline queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

DATASETS=(ucihar wisdm)
HARD_V3_DATASETS=(ucihar_hard_v3 wisdm_hard_v3)
VARIANTS=(numeric_mask shuffled_evidence hidden_evidence)
PROMPT_STYLES=(terse chain_then_json)

for dataset_name in "${HARD_V3_DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_embedding_full_stress "$dataset_name" "$variant"
  done
done

for dataset_name in "${DATASETS[@]}"; do
  for prompt_style in "${PROMPT_STYLES[@]}"; do
    run_hard_v2_prompt "$dataset_name" "$prompt_style"
  done
done

for dataset_name in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_hard_v2_stress "$dataset_name" "$variant"
  done
done

run_step \
  "emnlp_report_after_baseline_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP BASELINE QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
