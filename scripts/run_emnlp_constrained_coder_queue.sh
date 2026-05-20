#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-$(pwd)}"
cd "$WORKSPACE"

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="outputs/queue_logs/emnlp_constrained_coder_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_constrained_coder_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_constrained_coder_queue.lock"

mkdir -p outputs/queue_logs
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_constrained_coder_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" || true)"
  if [[ -n "${old_pid:-}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Constrained coder queue already running with pid=$old_pid"
    exit 0
  fi
fi
echo $$ > "$LOCK_PATH"
trap 'rm -f "$LOCK_PATH"' EXIT

log_status() {
  printf "%s\t%s\t%s\t%s\n" "$(date -Is)" "$1" "$2" "$3" | tee -a "$STATUS_PATH" >> "$LOG_PATH"
}

run_step() {
  local step_name="$1"
  local timeout_s="$2"
  shift 2
  log_status "$step_name" START "timeout=${timeout_s}s"
  if timeout "${timeout_s}" "$@" >> "$LOG_PATH" 2>&1; then
    log_status "$step_name" DONE "exit=0"
  else
    local code=$?
    if [[ $code -eq 124 ]]; then
      log_status "$step_name" TIMEOUT "exit=$code"
    else
      log_status "$step_name" FAIL "exit=$code"
    fi
    return $code
  fi
}

build_subset() {
  local dataset_name="$1"
  run_step \
    "build_constrained_subset_${dataset_name}" \
    1800 \
    "$PYTHON_BIN" scripts/build_constrained_caption_subset.py \
      --benchmark-path "data/benchmark/${dataset_name}_sensorfact_hard_v3_test.jsonl" \
      --structured-rows "outputs/axisfix_structured_regex_${dataset_name}_hard_v3_rows.jsonl" \
      --output-path "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" \
      --top-k 3
}

run_direct() {
  local dataset_name="$1"
  run_step \
    "coder_llm_${dataset_name}_hard_v3_constrained_prompt_fewshot_json" \
    34200 \
    "$PYTHON_BIN" scripts/run_qwen_llm_eval.py \
      --workspace "$WORKSPACE" \
      --model-dir models/Qwen_Qwen2.5-Coder-7B-Instruct \
      --benchmark-path "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" \
      --output-metrics "outputs/coder_llm_${dataset_name}_hard_v3_constrained_prompt_fewshot_json_metrics.json" \
      --output-rows "outputs/coder_llm_${dataset_name}_hard_v3_constrained_prompt_fewshot_json_rows.jsonl" \
      --max-records -1 \
      --batch-size 4 \
      --max-new-tokens 128 \
      --device cuda \
      --prompt-style fewshot_json \
      --support-seed 42
}

run_hybrid() {
  local dataset_name="$1"
  run_step \
    "hybrid_regex_coder_${dataset_name}_hard_v3_constrained_prompt_fewshot_json" \
    1800 \
    "$PYTHON_BIN" scripts/run_hybrid_verifier_eval.py \
      --workspace "$WORKSPACE" \
      --structured-rows "outputs/axisfix_structured_regex_${dataset_name}_hard_v3_rows.jsonl" \
      --direct-rows "outputs/coder_llm_${dataset_name}_hard_v3_constrained_prompt_fewshot_json_rows.jsonl" \
      --output-metrics "outputs/hybrid_regex_coder_${dataset_name}_hard_v3_constrained_prompt_fewshot_json_metrics.json" \
      --output-rows "outputs/hybrid_regex_coder_${dataset_name}_hard_v3_constrained_prompt_fewshot_json_rows.jsonl" \
      --structured-threshold 0.5 \
      --system-name "hybrid_regex_coder_${dataset_name}_hard_v3_constrained_prompt_fewshot_json"
}

echo "EMNLP constrained coder queue run id: $RUN_ID" | tee "$LOG_PATH"

run_step \
  "pytest_constrained_coder_queue" \
  1800 \
  "$PYTHON_BIN" -m pytest tests/test_hybrid_verifier_eval.py tests/test_emnlp_constrained_coder_queue.py -q

for dataset in ucihar wisdm mhealth; do
  build_subset "$dataset"
  run_direct "$dataset"
  run_hybrid "$dataset"
done

run_step \
  "emnlp_report_after_constrained_coder_queue" \
  1800 \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py --workspace "$WORKSPACE" --output outputs/emnlp_experiment_report.md
