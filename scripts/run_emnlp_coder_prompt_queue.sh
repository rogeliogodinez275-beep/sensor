#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
MODEL_DIR="${MODEL_DIR:-models/Qwen_Qwen2.5-Coder-7B-Instruct}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/order

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_coder_prompt_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_coder_prompt_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_coder_prompt_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_coder_prompt_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Coder prompt queue already running with pid=$old_pid"
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
    echo "[$(date -Is)] Waiting for active LLM job before coder-prompt step"
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
  local prompt_style="$3"
  local support_seed="$4"
  local negative_count="$5"
  local output_prefix="outputs/${name}"
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
      --model-dir "$MODEL_DIR" \
      --benchmark-path "$benchmark_path" \
      --output-metrics "${output_prefix}_metrics.json" \
      --output-rows "${output_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --max-new-tokens 128 \
      --device cuda \
      --prompt-style "$prompt_style" \
      "${extra_args[@]}" || true
}

ensure_order_file() {
  local dataset_name="$1"
  local seed="$2"
  local input_path
  case "$dataset_name" in
    ucihar) input_path="data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" ;;
    wisdm) input_path="data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" ;;
    mhealth) input_path="data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" ;;
    *) echo "Unknown dataset: $dataset_name" >&2; return 2 ;;
  esac
  local order_path="data/benchmark/order/${dataset_name}_hard_v3_caption_order_seed${seed}.jsonl"
  run_step \
    "build_${dataset_name}_hard_v3_caption_order_seed${seed}" \
    "$STEP_TIMEOUT_SHORT" \
    "$order_path" \
    "$PYTHON_BIN" scripts/build_caption_order_benchmark.py \
      --input "$input_path" \
      --output "$order_path" \
      --seed "$seed" || true
}

echo "EMNLP coder prompt queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step \
  "pytest_coder_prompt_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_qwen_llm_eval.py tests/test_caption_order_benchmark.py tests/test_emnlp_coder_prompt_queue.py -q || true

run_llm coder_llm_ucihar_hard_v3_prompt_terse data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl terse 42 all
run_llm coder_llm_wisdm_hard_v3_prompt_terse data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl terse 42 all
run_llm coder_llm_mhealth_hard_v3_prompt_terse data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl terse 42 all

for dataset_name in ucihar wisdm mhealth; do
  ensure_order_file "$dataset_name" 5153
  run_llm "coder_llm_${dataset_name}_hard_v3_caption_order_seed5153" "data/benchmark/order/${dataset_name}_hard_v3_caption_order_seed5153.jsonl" strict_json 42 all
done

run_llm coder_llm_ucihar_hard_v3_support_order_seed7101 data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl strict_json 7101 all
run_llm coder_llm_wisdm_hard_v3_support_order_seed7101 data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl strict_json 7101 all
run_llm coder_llm_mhealth_hard_v3_support_order_seed7101 data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl strict_json 7101 all

run_llm coder_llm_ucihar_hard_v3_support_balanced_neg3 data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl strict_json 42 3
run_llm coder_llm_wisdm_hard_v3_support_balanced_neg3 data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl strict_json 42 3
run_llm coder_llm_mhealth_hard_v3_support_balanced_neg3 data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl strict_json 42 3

run_step \
  "emnlp_report_after_coder_prompt_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP CODER PROMPT QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
