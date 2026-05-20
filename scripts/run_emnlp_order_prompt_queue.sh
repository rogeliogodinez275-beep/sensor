#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/order

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_order_prompt_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_order_prompt_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_order_prompt_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Order prompt queue already running with pid=$old_pid"
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
  while pgrep -af "scripts/run_qwen_llm_eval.py" >/dev/null; do
    echo "[$(date -Is)] Waiting for active run_qwen_llm_eval.py"
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

ensure_order_file() {
  local dataset_name="$1"
  local hard_version="$2"
  local seed="$3"
  local input_path
  case "${dataset_name}:${hard_version}" in
    ucihar:v3) input_path="data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" ;;
    wisdm:v3) input_path="data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" ;;
    *)
      echo "Unknown dataset/version for order prompt queue: ${dataset_name}:${hard_version}" >&2
      return 2
      ;;
  esac
  local order_path="data/benchmark/order/${dataset_name}_hard_${hard_version}_caption_order_seed${seed}.jsonl"
  run_step \
    "build_${dataset_name}_hard_${hard_version}_caption_order_seed${seed}" \
    "$STEP_TIMEOUT_SHORT" \
    "$order_path" \
    "$PYTHON_BIN" scripts/build_caption_order_benchmark.py \
      --input "$input_path" \
      --output "$order_path" \
      --seed "$seed" || true
}

run_order_prompt_eval() {
  local dataset_name="$1"
  local hard_version="$2"
  local seed="$3"
  local prompt_style="$4"
  local order_path="data/benchmark/order/${dataset_name}_hard_${hard_version}_caption_order_seed${seed}.jsonl"
  local output_prefix="outputs/qwen_llm_${dataset_name}_hard_${hard_version}_caption_order_seed${seed}_prompt_${prompt_style}"
  run_llm_step \
    "llm_${dataset_name}_hard_${hard_version}_caption_order_seed${seed}_prompt_${prompt_style}" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --benchmark-path "$order_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 128 \
      --device cuda \
      --prompt-style "$prompt_style" || true
}

echo "EMNLP candidate-order prompt queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

# Output families: ucihar_hard_v3_caption_order_seed and wisdm_hard_v3_caption_order_seed.
DATASETS=(ucihar wisdm)
SEEDS=(5151 5152)
PROMPT_STYLES=(terse chain_then_json)

for dataset_name in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    ensure_order_file "$dataset_name" "v3" "$seed"
    for prompt_style in "${PROMPT_STYLES[@]}"; do
      run_order_prompt_eval "$dataset_name" "v3" "$seed" "$prompt_style"
    done
  done
done

run_step \
  "emnlp_report_after_order_prompt_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP ORDER PROMPT QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
