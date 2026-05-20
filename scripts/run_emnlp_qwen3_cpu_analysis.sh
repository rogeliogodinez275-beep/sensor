#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/root/autodl-tmp/emnlp2026}"
PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/conda_envs/sensorfact/bin/python}"

cd "$WORKSPACE"
mkdir -p docs outputs/queue_logs

for dataset in ucihar wisdm mhealth; do
  primary="outputs/coder_llm_${dataset}_hard_v3_constrained_caption_order_vote5_prompt_fewshot_json_caption_only_rows.jsonl"
  qwen3_rows="outputs/qwen3_4b_choice_logprob_${dataset}_hard_v3_constrained_full_rows.jsonl"

  "$PYTHON_BIN" scripts/calibrate_gate_threshold.py \
    --primary-rows "$primary" \
    --alternate-rows "$qwen3_rows" \
    --output-json "outputs/dev_threshold_heldout_${dataset}_qwen3_choice_logprob.json" \
    --dev-modulus 5 \
    --dev-remainders 0 \
    --system-name "qwen3_dev_calibrated_caption_gate_${dataset}"

  "$PYTHON_BIN" scripts/analyze_margin_curve.py \
    --primary-rows "$primary" \
    --alternate-rows "$qwen3_rows" \
    --output-json "outputs/qwen3_choice_logprob_margin_curve_${dataset}.json" \
    --output-threshold-csv "outputs/qwen3_choice_logprob_margin_curve_${dataset}.csv" \
    --output-bucket-csv "outputs/qwen3_choice_logprob_margin_buckets_${dataset}.csv"

  args=(
    --benchmark "outputs/constrained_${dataset}_hard_v3_subset.jsonl"
    --target-fraction 0.50
    --output-json "outputs/balanced_candidate_subset_${dataset}.json"
    --output-md "docs/balanced_candidate_subset_${dataset}.md"
  )
  for spec in \
    "vote5=outputs/coder_llm_${dataset}_hard_v3_constrained_caption_order_vote5_prompt_fewshot_json_caption_only_rows.jsonl" \
    "visible_choice=outputs/coder_choice_logprob_${dataset}_hard_v3_constrained_full_rows.jsonl" \
    "visible_gated=outputs/coder_gated_vote5_choice_logprob_${dataset}_hard_v3_constrained_margin2_rows.jsonl" \
    "hidden_choice=outputs/coder_choice_logprob_${dataset}_hard_v3_constrained_hidden_evidence_rows.jsonl" \
    "hidden_gated=outputs/coder_gated_vote5_choice_logprob_${dataset}_hard_v3_constrained_hidden_evidence_margin2p0_rows.jsonl" \
    "position_prior=outputs/coder_position_prior_${dataset}_hard_v3_constrained_rows.jsonl" \
    "qwen3_choice=outputs/qwen3_4b_choice_logprob_${dataset}_hard_v3_constrained_full_rows.jsonl" \
    "qwen3_gated=outputs/qwen3_4b_gated_vote5_choice_logprob_${dataset}_hard_v3_constrained_margin2p0_rows.jsonl" \
    "qwen3_position_prior=outputs/qwen3_4b_position_prior_${dataset}_hard_v3_constrained_rows.jsonl"; do
    path="${spec#*=}"
    if [[ -s "$path" ]]; then
      args+=(--row-set "$spec")
    else
      echo "SKIP missing $spec"
    fi
  done
  "$PYTHON_BIN" scripts/analyze_balanced_candidate_subset.py "${args[@]}"
done

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

lines = [
    "# Qwen3 Gate Calibration Summary",
    "",
    "| Dataset | Best Dev Threshold | Dev Acc. | Heldout Acc. | Heldout Coverage |",
    "|---|---:|---:|---:|---:|",
]
for dataset in ["ucihar", "wisdm", "mhealth"]:
    path = Path(f"outputs/dev_threshold_heldout_{dataset}_qwen3_choice_logprob.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    heldout = data.get("heldout_eval") or {}
    lines.append(
        "| {dataset} | {threshold} | {dev_acc:.4f} | {heldout_acc:.4f} | {coverage:.4f} |".format(
            dataset=dataset,
            threshold=data.get("best_threshold"),
            dev_acc=float(data["best"]["caption_selection_accuracy"]),
            heldout_acc=float(heldout.get("caption_selection_accuracy", float("nan"))),
            coverage=float(heldout.get("coverage", float("nan"))),
        )
    )
lines.extend(
    [
        "",
        "Interpretation: this checks whether the fixed margin=2 gate transfers to the external Qwen3 reranker. Large threshold shifts or low heldout accuracy indicate model-specific calibration risk.",
    ]
)
Path("docs/qwen3_gate_calibration_summary_2026-05-20.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)
print("\n".join(lines))
PY

ls -lt \
  docs/qwen3_gate_calibration_summary_2026-05-20.md \
  docs/balanced_candidate_subset_*.md \
  outputs/dev_threshold_heldout_*_qwen3_choice_logprob.json
