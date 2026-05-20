#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1536}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-128}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-7200}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs data/benchmark/samples

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_extra_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_extra_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_extra_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "EMNLP extra queue already running with pid=$old_pid"
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

echo "EMNLP extra queue run id: $RUN_ID"
df -h /root/autodl-tmp || true
nvidia-smi || true

run_step "pytest_extra" 1800 "" "$PYTHON_BIN" -m pytest -q || exit "$FAILURES"

DATASETS=(
  "ucihar_hard_v3:data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl"
  "wisdm_hard_v3:data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl"
)
SEEDS=(2027 2028 2029)
PROMPT_STYLES=(terse chain_then_json)

for item in "${DATASETS[@]}"; do
  name="${item%%:*}"
  path="${item#*:}"
  for seed in "${SEEDS[@]}"; do
    sample_path="data/benchmark/samples/${name}_sample_${SAMPLE_SIZE}_seed${seed}.jsonl"
    run_step \
      "sample_${name}_seed${seed}" \
      900 \
      "$sample_path" \
      "$PYTHON_BIN" scripts/sample_jsonl.py \
        --input "$path" \
        --output "$sample_path" \
        --sample-size "$SAMPLE_SIZE" \
        --seed "$seed" || continue

    run_step \
      "llm_${name}_seed${seed}" \
      "$STEP_TIMEOUT_LLM" \
      "outputs/qwen_llm_${name}_seed${seed}_metrics.json" \
      "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
        --benchmark-path "$sample_path" \
        --output-metrics "outputs/qwen_llm_${name}_seed${seed}_metrics.json" \
        --output-rows "outputs/qwen_llm_${name}_seed${seed}_rows.jsonl" \
        --max-records -1 \
        --batch-size "$LLM_BATCH_SIZE" \
        --max-new-tokens 64 \
        --device cuda \
        --prompt-style strict_json
  done

  for prompt_style in "${PROMPT_STYLES[@]}"; do
    sample_path="data/benchmark/samples/${name}_sample_${SAMPLE_SIZE}_seed2027.jsonl"
    run_step \
      "llm_${name}_prompt_${prompt_style}" \
      "$STEP_TIMEOUT_LLM" \
      "outputs/qwen_llm_${name}_prompt_${prompt_style}_metrics.json" \
      "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
        --benchmark-path "$sample_path" \
        --output-metrics "outputs/qwen_llm_${name}_prompt_${prompt_style}_metrics.json" \
        --output-rows "outputs/qwen_llm_${name}_prompt_${prompt_style}_rows.jsonl" \
        --max-records -1 \
        --batch-size "$LLM_BATCH_SIZE" \
        --max-new-tokens "$LLM_MAX_NEW_TOKENS" \
        --device cuda \
        --prompt-style "$prompt_style"
  done
done

run_step \
  "emnlp_experiment_report" \
  900 \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md

echo "[$(date -Is)] EMNLP EXTRA QUEUE FINISHED run_id=$RUN_ID"
nvidia-smi || true
exit "$FAILURES"
