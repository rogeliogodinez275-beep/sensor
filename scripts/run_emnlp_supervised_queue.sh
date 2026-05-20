#!/usr/bin/env bash
set -u -o pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
STEP_TIMEOUT_SHORT="${STEP_TIMEOUT_SHORT:-1800}"

cd "$WORKSPACE" || exit 2
mkdir -p outputs/queue_logs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_supervised_queue_${RUN_ID}.log"
STATUS_PATH="outputs/queue_logs/emnlp_supervised_status.tsv"
LOCK_PATH="outputs/queue_logs/emnlp_supervised_queue.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Supervised queue already running with pid=$old_pid"
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

run_supervised() {
  local dataset_name="$1"
  local feature_mode="$2"
  local output_prefix
  local metrics_path
  local rows_path
  local train_path
  local eval_path
  case "$dataset_name" in
    ucihar)
      train_path="data/benchmark/ucihar_sensorfact_train.jsonl"
      eval_path="data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl"
      if [[ "$feature_mode" == "numeric_only" ]]; then
        output_prefix="outputs/supervised_numeric_ucihar_hard_v3"
        metrics_path="outputs/supervised_numeric_ucihar_hard_v3_metrics.json"
        rows_path="outputs/supervised_numeric_ucihar_hard_v3_rows.jsonl"
      else
        output_prefix="outputs/supervised_ucihar_hard_v3"
        metrics_path="outputs/supervised_ucihar_hard_v3_metrics.json"
        rows_path="outputs/supervised_ucihar_hard_v3_rows.jsonl"
      fi
      ;;
    wisdm)
      train_path="data/benchmark/wisdm_sensorfact_train.jsonl"
      eval_path="data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl"
      if [[ "$feature_mode" == "numeric_only" ]]; then
        output_prefix="outputs/supervised_numeric_wisdm_hard_v3"
        metrics_path="outputs/supervised_numeric_wisdm_hard_v3_metrics.json"
        rows_path="outputs/supervised_numeric_wisdm_hard_v3_rows.jsonl"
      else
        output_prefix="outputs/supervised_wisdm_hard_v3"
        metrics_path="outputs/supervised_wisdm_hard_v3_metrics.json"
        rows_path="outputs/supervised_wisdm_hard_v3_rows.jsonl"
      fi
      ;;
    *)
      echo "Unknown dataset_name=$dataset_name"
      return 2
      ;;
  esac
  run_step \
    "supervised_${feature_mode}_${dataset_name}_hard_v3" \
    "$STEP_TIMEOUT_SHORT" \
    "${metrics_path},${rows_path}" \
    "$PYTHON_BIN" scripts/run_supervised_baseline.py \
      --workspace "$WORKSPACE" \
      --train-path "$train_path" \
      --eval-path "$eval_path" \
      --output-metrics "$metrics_path" \
      --output-rows "$rows_path" \
      --model-type logistic_regression \
      --max-train-records -1 \
      --max-eval-records -1 \
      --seed 42 \
      --hard-variant v3 \
      --feature-mode "$feature_mode" || true
}

echo "EMNLP supervised queue run id: $RUN_ID"
df -h /root/autodl-tmp || true

# The second pass is the numeric-only control: scripts/run_supervised_baseline.py --feature-mode numeric_only.
for dataset_name in ucihar wisdm; do
  run_supervised "$dataset_name" oracle_fields
  run_supervised "$dataset_name" numeric_only
done

run_step \
  "emnlp_report_after_supervised_queue" \
  "$STEP_TIMEOUT_SHORT" \
  "" \
  "$PYTHON_BIN" scripts/make_emnlp_experiment_report.py \
    --workspace "$WORKSPACE" \
    --output outputs/emnlp_experiment_report.md || true

echo "[$(date -Is)] EMNLP SUPERVISED QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
exit "$FAILURES"
