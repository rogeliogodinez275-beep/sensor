#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/emnlp_raw_sensor_clean}"
STATUS_PATH="${STATUS_PATH:-outputs/queue_logs/emnlp_raw_sensor_gpu_status.tsv}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
MAX_CALIBRATION_RECORDS="${MAX_CALIBRATION_RECORDS:-1024}"
STEP_TIMEOUT="${STEP_TIMEOUT:-14400}"

cd "$WORKSPACE" || exit 2
mkdir -p "$OUT_ROOT/models" outputs/queue_logs docs

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="outputs/queue_logs/emnlp_raw_sensor_gpu_${RUN_ID}.log"
LOCK_PATH="outputs/queue_logs/emnlp_raw_sensor_gpu.lock"
FAILURES=0

exec > >(tee -a "$LOG_PATH") 2>&1
ln -sfn "$(basename "$LOG_PATH")" outputs/queue_logs/emnlp_raw_sensor_gpu_launcher.log

if [[ -f "$LOCK_PATH" ]]; then
  old_pid="$(cat "$LOCK_PATH" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "raw sensor GPU queue already running with pid=$old_pid"
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

processed_train_for_dataset() {
  case "$1" in
    ucihar) echo "data/processed/ucihar_train.jsonl" ;;
    wisdm) echo "data/processed/wisdm_train.jsonl" ;;
    mhealth) echo "data/processed/mhealth_train.jsonl" ;;
    *) return 2 ;;
  esac
}

processed_test_for_dataset() {
  case "$1" in
    ucihar) echo "data/processed/ucihar_test.jsonl" ;;
    wisdm) echo "data/processed/wisdm_test.jsonl" ;;
    mhealth) echo "data/processed/mhealth_test.jsonl" ;;
    *) return 2 ;;
  esac
}

benchmark_train_for_dataset() {
  case "$1" in
    ucihar) echo "data/benchmark/ucihar_sensorfact_train.jsonl" ;;
    wisdm) echo "data/benchmark/wisdm_sensorfact_train.jsonl" ;;
    mhealth) echo "data/benchmark/mhealth_sensorfact_train.jsonl" ;;
    *) return 2 ;;
  esac
}

benchmark_eval_for_dataset() {
  local hard="data/benchmark/${1}_sensorfact_hard_v3_test.jsonl"
  local plain="data/benchmark/${1}_sensorfact_test.jsonl"
  if [[ -s "$hard" ]]; then
    echo "$hard"
  else
    echo "$plain"
  fi
}

echo "EMNLP raw-sensor GPU queue run id: $RUN_ID"
echo "WORKSPACE=$WORKSPACE"
echo "OUT_ROOT=$OUT_ROOT"
nvidia-smi || true
"$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY

run_step \
  "pytest_raw_sensor_baseline" \
  600 \
  "" \
  "$PYTHON_BIN" -m pytest tests/test_raw_sensor_baseline.py -q || true

"$PYTHON_BIN" - <<'PY'
from sensorfact.models.raw_sensor_alignment import write_leakage_audit
write_leakage_audit("outputs/emnlp_raw_sensor_clean/leakage_audit.json")
PY

for dataset in ucihar wisdm mhealth; do
  train_windows="$(processed_train_for_dataset "$dataset")"
  test_windows="$(processed_test_for_dataset "$dataset")"
  train_records="$(benchmark_train_for_dataset "$dataset")"
  eval_records="$(benchmark_eval_for_dataset "$dataset")"
  metrics="$OUT_ROOT/raw_sensor_${dataset}_metrics.json"
  rows="$OUT_ROOT/raw_sensor_${dataset}_rows.jsonl"
  model="$OUT_ROOT/models/raw_sensor_${dataset}.pt"
  run_step \
    "raw_sensor_${dataset}" \
    "$STEP_TIMEOUT" \
    "${metrics},${rows},${model}" \
    "$PYTHON_BIN" scripts/run_raw_sensor_baseline.py \
      --workspace "$WORKSPACE" \
      --train-windows "$train_windows" \
      --train-records "$train_records" \
      --eval-windows "$test_windows" \
      --eval-records "$eval_records" \
      --metrics "$metrics" \
      --rows "$rows" \
      --model "$model" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --seed 20260521 \
      --device cuda \
      --max-calibration-records "$MAX_CALIBRATION_RECORDS" || true
done

run_step \
  "summarize_raw_sensor_results" \
  600 \
  "$OUT_ROOT/raw_sensor_result_lock.md,$OUT_ROOT/raw_sensor_result_lock.json,docs/raw_sensor_leakage_audit.md" \
  "$PYTHON_BIN" scripts/summarize_raw_sensor_results.py \
    --root "$OUT_ROOT" \
    --docs-dir docs || true

echo "[$(date -Is)] RAW SENSOR GPU QUEUE FINISHED run_id=$RUN_ID failures=$FAILURES"
nvidia-smi || true
log_status "emnlp_raw_sensor_gpu_queue" "FINISH" "failures=$FAILURES"
exit "$FAILURES"
