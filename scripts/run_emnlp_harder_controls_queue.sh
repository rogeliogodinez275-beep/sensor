#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
MODEL_DIR="${MODEL_DIR:-models/Qwen_Qwen2.5-Coder-7B-Instruct}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
GATE_MARGIN="${GATE_MARGIN:-2.0}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs outputs/risk_bounded_next/controls docs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_harder_controls_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_harder_controls_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_harder_controls_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_harder_controls_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Harder controls queue already running with pid=$old_pid"
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
  [[ -z "$expected_outputs" ]] && return 1
  IFS=',' read -r -a paths <<< "$expected_outputs"
  local candidate_path
  for candidate_path in "${paths[@]}"; do
    [[ ! -s "$candidate_path" ]] && return 1
  done
  return 0
}

wait_for_llm_slot() {
  while ps -ef | grep -E "scripts/run_(qwen_llm_eval|logprob_reranker|structured_verifier_eval)\.py" | grep -v grep >/dev/null; do
    echo "[$(date -Is)] Waiting for active LLM job before harder-control step"
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

tag_for_mode() {
  case "$1" in
    numeric-swap) echo "numeric_swap" ;;
    axis-permutation) echo "axis_permutation" ;;
    trend-flip) echo "trend_flip" ;;
    *) echo "$1" ;;
  esac
}

run_harder_control_for_dataset() {
  local dataset_name="$1"
  local mode="$2"
  local tag
  tag="$(tag_for_mode "$mode")"
  local benchmark="outputs/risk_bounded_next/controls/${dataset_name}_hard_v3_constrained_${tag}.jsonl"
  local metadata="outputs/risk_bounded_next/controls/${dataset_name}_hard_v3_constrained_${tag}_metadata.json"
  local choice_prefix="outputs/coder_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}"
  local gated_prefix="outputs/coder_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}_margin${GATE_MARGIN//./p}"
  local nogate_prefix="outputs/coder_nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}"
  local primary_rows="outputs/coder_llm_${dataset_name}_hard_v3_constrained_caption_order_vote5_prompt_fewshot_json_caption_only_rows.jsonl"

  run_step \
    "build_${dataset_name}_${tag}_control_benchmark" \
    "$STEP_TIMEOUT_SHORT" \
    "$benchmark,$metadata" \
    "$PYTHON_BIN" scripts/build_evidence_control_benchmark.py \
      --input "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" \
      --output "$benchmark" \
      --mode "$mode" \
      --seed 20260520 \
      --metadata "$metadata" || true

  wait_for_llm_slot
  run_step \
    "coder_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_LLM" \
    "${choice_prefix}_metrics.json,${choice_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_logprob_reranker.py \
      --workspace "$WORKSPACE" \
      --model-dir "$MODEL_DIR" \
      --benchmark-path "$benchmark" \
      --output-metrics "${choice_prefix}_metrics.json" \
      --output-rows "${choice_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --device cuda \
      --mode choice || true

  run_step \
    "gate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_SHORT" \
    "${gated_prefix}_metrics.json,${gated_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/gate_caption_rows.py \
      --primary-rows "$primary_rows" \
      --alternate-rows "${choice_prefix}_rows.jsonl" \
      --output-metrics "${gated_prefix}_metrics.json" \
      --output-rows "${gated_prefix}_rows.jsonl" \
      --min-alternate-margin "$GATE_MARGIN" \
      --system-name "coder_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}_margin${GATE_MARGIN//./p}" || true

  run_step \
    "nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_SHORT" \
    "${nogate_prefix}_metrics.json,${nogate_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/gate_caption_rows.py \
      --primary-rows "$primary_rows" \
      --alternate-rows "${choice_prefix}_rows.jsonl" \
      --output-metrics "${nogate_prefix}_metrics.json" \
      --output-rows "${nogate_prefix}_rows.jsonl" \
      --min-alternate-margin -1000000 \
      --system-name "coder_nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" || true
}

echo "EMNLP harder controls queue run id: $RUN_ID"
nvidia-smi || true

run_step \
  "pytest_harder_controls_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest \
    tests/test_harder_evidence_controls.py \
    tests/test_emnlp_harder_controls_queue.py \
    tests/test_gate_caption_rows.py -q || true

for dataset in ucihar wisdm mhealth; do
  for mode in numeric-swap axis-permutation trend-flip; do
    run_harder_control_for_dataset "$dataset" "$mode"
  done
done

echo "[$(date -Is)] Harder controls queue finished with failures=$FAILURES"
log_status "harder_controls_queue" "FINISH" "failures=$FAILURES"
exit "$FAILURES"
