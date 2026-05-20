#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
MODEL_DIR="${MODEL_DIR:-models/Qwen_Qwen3-4B-Instruct-2507}"
MODEL_TAG="${MODEL_TAG:-qwen3_4b}"
LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
GATE_MARGIN="${GATE_MARGIN:-2.0}"
STEP_TIMEOUT_LLM="${STEP_TIMEOUT_LLM:-34200}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs docs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_external_qwen3_controls_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_external_qwen3_controls_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_external_qwen3_controls_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_external_qwen3_controls_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "External Qwen3 controls queue already running with pid=$old_pid"
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
  while ps -ef | grep -E "scripts/run_(qwen_llm_eval|logprob_reranker|structured_verifier_eval)\.py" | grep -v grep >/dev/null; do
    echo "[$(date -Is)] Waiting for active LLM job before external Qwen3 control step"
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
    shuffled) echo "shuffled_evidence" ;;
    numeric-mask) echo "numeric_mask" ;;
    hidden) echo "hidden_evidence" ;;
    *) echo "$1" ;;
  esac
}

primary_rows_for_dataset() {
  local dataset_name="$1"
  echo "outputs/coder_llm_${dataset_name}_hard_v3_constrained_caption_order_vote5_prompt_fewshot_json_caption_only_rows.jsonl"
}

run_control_for_dataset() {
  local dataset_name="$1"
  local mode="$2"
  local tag
  tag="$(tag_for_mode "$mode")"
  local control_benchmark="outputs/constrained_${dataset_name}_hard_v3_${tag}.jsonl"
  local control_meta="outputs/constrained_${dataset_name}_hard_v3_${tag}_metadata.json"
  local primary_rows
  primary_rows="$(primary_rows_for_dataset "$dataset_name")"
  local choice_prefix="outputs/${MODEL_TAG}_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}"
  local gated_prefix="outputs/${MODEL_TAG}_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}_margin${GATE_MARGIN//./p}"
  local nogate_prefix="outputs/${MODEL_TAG}_nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}"

  run_step \
    "build_${dataset_name}_${tag}_control_benchmark" \
    "$STEP_TIMEOUT_SHORT" \
    "$control_benchmark,$control_meta" \
    "$PYTHON_BIN" scripts/build_evidence_control_benchmark.py \
      --input "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" \
      --output "$control_benchmark" \
      --mode "$mode" \
      --seed 20260520 \
      --metadata "$control_meta" || true

  wait_for_llm_slot
  run_step \
    "${MODEL_TAG}_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_LLM" \
    "${choice_prefix}_metrics.json,${choice_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/run_logprob_reranker.py \
      --workspace "$WORKSPACE" \
      --model-dir "$MODEL_DIR" \
      --benchmark-path "$control_benchmark" \
      --output-metrics "${choice_prefix}_metrics.json" \
      --output-rows "${choice_prefix}_rows.jsonl" \
      --max-records -1 \
      --batch-size "$LLM_BATCH_SIZE" \
      --device cuda \
      --mode choice || true

  run_step \
    "${MODEL_TAG}_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_SHORT" \
    "${gated_prefix}_metrics.json,${gated_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/gate_caption_rows.py \
      --primary-rows "$primary_rows" \
      --alternate-rows "${choice_prefix}_rows.jsonl" \
      --output-metrics "${gated_prefix}_metrics.json" \
      --output-rows "${gated_prefix}_rows.jsonl" \
      --min-alternate-margin "$GATE_MARGIN" \
      --system-name "${MODEL_TAG}_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}_margin${GATE_MARGIN//./p}" || true

  run_step \
    "${MODEL_TAG}_nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" \
    "$STEP_TIMEOUT_SHORT" \
    "${nogate_prefix}_metrics.json,${nogate_prefix}_rows.jsonl" \
    "$PYTHON_BIN" scripts/gate_caption_rows.py \
      --primary-rows "$primary_rows" \
      --alternate-rows "${choice_prefix}_rows.jsonl" \
      --output-metrics "${nogate_prefix}_metrics.json" \
      --output-rows "${nogate_prefix}_rows.jsonl" \
      --min-alternate-margin -1000000 \
      --system-name "${MODEL_TAG}_nogate_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" || true
}

echo "EMNLP external Qwen3 controls queue run id: $RUN_ID"
nvidia-smi || true

run_step \
  "pytest_external_qwen3_controls_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_logprob_reranker.py tests/test_evidence_control_benchmark.py -q || true

for dataset in ucihar wisdm mhealth; do
  for mode in shuffled numeric-mask hidden; do
    run_control_for_dataset "$dataset" "$mode"
  done
done

run_step \
  "summarize_external_qwen3_controls" \
  "$STEP_TIMEOUT_SHORT" \
  "docs/external_qwen3_controls_summary_2026-05-20.md,outputs/external_qwen3_controls_summary_2026-05-20.json" \
  "$PYTHON_BIN" scripts/summarize_external_controls.py \
    --outputs-dir outputs \
    --model-tag "$MODEL_TAG" \
    --output-md docs/external_qwen3_controls_summary_2026-05-20.md \
    --output-json outputs/external_qwen3_controls_summary_2026-05-20.json || true

echo "[$(date -Is)] EMNLP EXTERNAL QWEN3 CONTROLS QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
exit "$FAILURES"
