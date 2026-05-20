#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"
STEP_TIMEOUT_BUILD="${STEP_TIMEOUT_BUILD:-3600}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/stress data/benchmark/order

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_mhealth_extra_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_mhealth_extra_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_mhealth_extra_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_mhealth_extra_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "MHEALTH extra queue already running with pid=$old_pid"
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
    echo "[$(date -Is)] Waiting for active run_qwen_llm_eval.py before next MHEALTH extra job"
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
  local benchmark_path="$2"
  local output_prefix="$3"
  local prompt_style="${4:-strict_json}"
  local max_new_tokens="96"
  if [[ "$prompt_style" != "strict_json" ]]; then
    max_new_tokens="128"
  fi
  wait_for_llm_slot
  run_step \
    "$name" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --workspace "$WORKSPACE" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens "$max_new_tokens" \
      --device cuda \
      --prompt-style "$prompt_style" || true
}

run_embedding() {
  local name="$1"
  local benchmark_path="$2"
  local output_prefix="$3"
  run_step \
    "$name" \
    "$STEP_TIMEOUT_LLM" \
    "${output_prefix}_metrics.json,${output_prefix}_scores.jsonl" \
    "$PYTHON_BIN" scripts/run_qwen_embedding_eval.py \
      --workspace "$WORKSPACE" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-scores "${output_prefix}_scores.jsonl" \
      --max-records -1 \
      --batch-size "$EMBED_BATCH_SIZE" \
      --device cuda || true
}

echo "EMNLP MHEALTH extra queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step \
  "pytest_mhealth_extra" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_mhealth_data.py tests/test_emnlp_mhealth_extra_queue.py -q || true

run_step \
  "build_mhealth_hard_for_extra" \
  "$STEP_TIMEOUT_BUILD" \
  "data/benchmark/mhealth_sensorfact_hard_v2_test.jsonl,data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" \
  "$PYTHON_BIN" scripts/build_mhealth_benchmark.py \
    --workspace "$WORKSPACE" \
    --download \
    --hard-variants v2,v3 \
    --max-train-windows -1 \
    --max-test-windows -1 || true

run_embedding \
  "embedding_mhealth_hard_v2" \
  "data/benchmark/mhealth_sensorfact_hard_v2_test.jsonl" \
  "outputs/qwen_embedding_mhealth_hard_v2"

run_embedding \
  "embedding_mhealth_hard_v3" \
  "data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" \
  "outputs/qwen_embedding_mhealth_hard_v3"

run_llm \
  "llm_mhealth_hard_v2" \
  "data/benchmark/mhealth_sensorfact_hard_v2_test.jsonl" \
  "outputs/qwen_llm_mhealth_hard_v2"

run_llm \
  "llm_mhealth_hard_v3" \
  "data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" \
  "outputs/qwen_llm_mhealth_hard_v3"
# Explicit outputs: outputs/qwen_llm_mhealth_hard_v3_metrics.json and outputs/qwen_llm_mhealth_hard_v3_rows.jsonl.

PROMPT_STYLES=(terse chain_then_json)
for prompt_style in "${PROMPT_STYLES[@]}"; do
  run_llm \
    "llm_mhealth_hard_v3_prompt_${prompt_style}" \
    "data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" \
    "outputs/qwen_llm_mhealth_hard_v3_prompt_${prompt_style}" \
    "$prompt_style"
done
# Explicit prompt outputs include outputs/qwen_llm_mhealth_hard_v3_prompt_terse_metrics.json.

STRESS_VARIANTS=(numeric_mask shuffled_evidence hidden_evidence)
for variant in "${STRESS_VARIANTS[@]}"; do
  stress_path="data/benchmark/stress/mhealth_hard_v3_${variant}_full.jsonl"
  run_step \
    "build_mhealth_hard_v3_${variant}_full" \
    "$STEP_TIMEOUT_SHORT" \
    "$stress_path" \
    "$PYTHON_BIN" scripts/build_stress_benchmark.py \
      --input data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
      --output "$stress_path" \
      --variant "$variant" \
      --seed 4041 || true

  run_llm \
    "llm_mhealth_hard_v3_${variant}_full" \
    "$stress_path" \
    "outputs/qwen_llm_mhealth_hard_v3_${variant}_full"
# Explicit stress outputs include outputs/qwen_llm_mhealth_hard_v3_numeric_mask_full_metrics.json.

  for prompt_style in "${PROMPT_STYLES[@]}"; do
    run_llm \
      "llm_mhealth_hard_v3_${variant}_full_prompt_${prompt_style}" \
      "$stress_path" \
      "outputs/qwen_llm_mhealth_hard_v3_${variant}_full_prompt_${prompt_style}" \
      "$prompt_style"
  done
done
# Explicit stress-prompt outputs include outputs/qwen_llm_mhealth_hard_v3_numeric_mask_full_prompt_terse_metrics.json.

SEEDS=(5151 5152)
for seed in "${SEEDS[@]}"; do
  order_path="data/benchmark/order/mhealth_hard_v3_caption_order_seed${seed}.jsonl"
  run_step \
    "build_mhealth_hard_v3_caption_order_seed${seed}" \
    "$STEP_TIMEOUT_SHORT" \
    "$order_path" \
    "$PYTHON_BIN" scripts/build_caption_order_benchmark.py \
      --input data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl \
      --output "$order_path" \
      --seed "$seed" || true

  run_llm \
    "llm_mhealth_hard_v3_caption_order_seed${seed}" \
    "$order_path" \
    "outputs/qwen_llm_mhealth_hard_v3_caption_order_seed${seed}"
# Explicit order outputs include outputs/qwen_llm_mhealth_hard_v3_caption_order_seed5151_metrics.json.

  for prompt_style in "${PROMPT_STYLES[@]}"; do
    run_llm \
      "llm_mhealth_hard_v3_caption_order_seed${seed}_prompt_${prompt_style}" \
      "$order_path" \
      "outputs/qwen_llm_mhealth_hard_v3_caption_order_seed${seed}_prompt_${prompt_style}" \
      "$prompt_style"
  done
done
# Explicit order-prompt outputs include outputs/qwen_llm_mhealth_hard_v3_caption_order_seed5151_prompt_terse_metrics.json.

run_step \
  "emnlp_report_after_mhealth_extra_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP MHEALTH EXTRA QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
