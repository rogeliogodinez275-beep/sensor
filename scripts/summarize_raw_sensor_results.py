from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASETS = {
    "ucihar": "UCI HAR",
    "wisdm": "WISDM",
    "mhealth": "MHEALTH",
}
DEFAULT_DOCS_MD = "docs/raw_sensor_leakage_audit.md"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "missing"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _collect(root: Path) -> dict:
    rows = []
    for dataset_id, label in DATASETS.items():
        metrics_path = root / f"raw_sensor_{dataset_id}_metrics.json"
        rows_path = root / f"raw_sensor_{dataset_id}_rows.jsonl"
        model_path = root / "models" / f"raw_sensor_{dataset_id}.pt"
        metrics = _read_json(metrics_path)
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": label,
                "metrics_path": _path_text(metrics_path),
                "rows_path": _path_text(rows_path),
                "model_path": _path_text(model_path),
                "status": "done" if metrics else "missing",
                "caption_accuracy": None if metrics is None else metrics.get("caption_selection_accuracy"),
                "caption_macro_f1": None if metrics is None else metrics.get("caption_macro_f1"),
                "cf_reject_f1": None if metrics is None else metrics.get("cf_reject_f1"),
                "support_ece": None if metrics is None else metrics.get("support_ece"),
                "support_brier": None if metrics is None else metrics.get("support_brier"),
                "evidence_field_accuracy": None if metrics is None else metrics.get("evidence_field_accuracy"),
                "device": None if metrics is None else metrics.get("device"),
                "n_train_records": None if metrics is None else metrics.get("n_train_records"),
                "n_eval_records": None if metrics is None else metrics.get("n_eval_records"),
                "threshold": None if metrics is None else metrics.get("support_score_threshold"),
            }
        )
    return {
        "raw_sensor": rows,
        "claim_boundary": {
            "fair_main_candidate": "raw_sensor_field_aligner",
            "oracle_upper_bound": [
                "supervised oracle_fields",
                "supervised axis_drop",
                "supervised numeric_drop",
                "aligner base_full/distill_full/riskcal_full and drop ablations",
            ],
            "not_clean_enough": [
                "numeric_only modes that read numeric summaries from eval evidence cards",
            ],
        },
    }


def _render(payload: dict) -> str:
    lines = [
        "# Raw-Sensor Leakage Audit and Corrected Baseline",
        "",
        "## Why We Re-ran",
        "",
        "The previous near-1.0 supervised and aligner results are not fair main baselines. They read structured evidence-card fields, or numeric summaries stored inside the eval evidence card, during test-time scoring. Those numbers should be treated as oracle/upper-bound diagnostics, not as evidence that the model learned sensor grounding.",
        "",
        "The corrected baseline below trains on raw sensor windows with train evidence fields as supervision. At test time it reads only the raw sensor window and candidate text. Eval evidence is used only after prediction to compute diagnostic evidence-field accuracy.",
        "",
        "## Corrected Raw-Sensor Baseline",
        "",
        "| Dataset | Caption Acc. | Caption Macro-F1 | CF Reject F1 | Support ECE | Support Brier | Evidence Field Acc. | Device | Train N | Eval N | Threshold | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["raw_sensor"]:
        lines.append(
            f"| {row['dataset']} | {_fmt(row['caption_accuracy'])} | {_fmt(row['caption_macro_f1'])} | {_fmt(row['cf_reject_f1'])} | {_fmt(row['support_ece'])} | {_fmt(row['support_brier'])} | {_fmt(row['evidence_field_accuracy'])} | {_fmt(row['device'])} | {_fmt(row['n_train_records'], 0)} | {_fmt(row['n_eval_records'], 0)} | {_fmt(row['threshold'])} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Fair main baseline candidate: `raw_sensor_field_aligner`.",
            "- `oracle_fields`, `axis_drop`, `numeric_drop`, and full/distilled aligners are upper bounds or ablations because they read structured evidence at test time.",
            "- Numeric-only modes that read numeric summaries from eval evidence cards are not clean enough for the main result; they are at best semi-oracle diagnostics.",
            "- Reranker results should remain caption-only residual ambiguity results and must not be credited for support F1 or CF rejection F1.",
            "",
            "## Files",
            "",
        ]
    )
    for row in payload["raw_sensor"]:
        lines.append(
            f"- {row['dataset']}: metrics `{row['metrics_path']}`; rows/model are archived locally or on the remote server, not in the lightweight GitHub package."
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw-sensor-only SensorFact baseline results.")
    parser.add_argument("--root", default="outputs/emnlp_raw_sensor_clean")
    parser.add_argument("--docs-dir", default="docs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    docs_dir = Path(args.docs_dir)
    root.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    payload = _collect(root)
    result_json = root / "raw_sensor_result_lock.json"
    result_md = root / "raw_sensor_result_lock.md"
    docs_md = docs_dir / "raw_sensor_leakage_audit.md"
    text = _render(payload)
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result_md.write_text(text, encoding="utf-8", newline="\n")
    docs_md.write_text(text, encoding="utf-8", newline="\n")
    print("Raw-sensor summary written.")
    print(f"result_json: {result_json}")
    print(f"result_md: {result_md}")
    print(f"docs_md: {docs_md}")


if __name__ == "__main__":
    main()
