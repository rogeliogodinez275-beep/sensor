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

CONDITIONS = {
    "visible": "full",
    "shuffled-evidence": "shuffled_evidence",
    "numeric-mask": "numeric_mask",
    "hidden-evidence": "hidden_evidence",
}


def _read_json(path: Path) -> dict | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(metrics: dict | None, key: str) -> float | None:
    if metrics is None or key not in metrics:
        return None
    return float(metrics[key])


def _fmt(value: float | None) -> str:
    return "信息不足" if value is None else f"{value:.4f}"


def collect_control_rows(outputs_dir: str | Path) -> list[dict]:
    root = Path(outputs_dir)
    rows: list[dict] = []
    for dataset, label in DATASETS.items():
        for condition, tag in CONDITIONS.items():
            if condition == "visible":
                choice_path = root / f"coder_choice_logprob_{dataset}_hard_v3_constrained_full_metrics.json"
                gated_path = root / f"coder_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained_margin2_metrics.json"
            else:
                choice_path = root / f"coder_choice_logprob_{dataset}_hard_v3_constrained_{tag}_metrics.json"
                gated_path = root / f"coder_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained_{tag}_margin2p0_metrics.json"
            choice = _read_json(choice_path)
            gated = _read_json(gated_path)
            alternate_count = _metric(gated, "caption_gate_alternate_count")
            n_records = _metric(gated, "n_eval_records")
            coverage = None
            if alternate_count is not None and n_records:
                coverage = alternate_count / n_records
            rows.append(
                {
                    "dataset": label,
                    "dataset_id": dataset,
                    "condition": condition,
                    "choice_acc": _metric(choice, "caption_selection_accuracy"),
                    "gated_acc": _metric(gated, "caption_selection_accuracy"),
                    "gated_coverage": coverage,
                }
            )
    return rows


def _risk_notes(rows: list[dict]) -> list[str]:
    notes: list[str] = []
    by_dataset: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["condition"]] = row
    for dataset, condition_rows in by_dataset.items():
        visible = condition_rows.get("visible", {})
        visible_choice = visible.get("choice_acc")
        visible_gated = visible.get("gated_acc")
        for condition in ["shuffled-evidence", "numeric-mask", "hidden-evidence"]:
            control = condition_rows.get(condition, {})
            control_choice = control.get("choice_acc")
            control_gated = control.get("gated_acc")
            if visible_choice is not None and control_choice is not None and visible_choice - control_choice < 0.03:
                notes.append(
                    f"- {dataset} 的 {condition} choice 与 visible 差距小于 0.03，需要收紧 grounding claim。"
                )
            if visible_gated is not None and control_gated is not None and visible_gated - control_gated < 0.03:
                notes.append(
                    f"- {dataset} 的 {condition} gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。"
                )
    if not notes:
        notes.append("- 当前 control 与 visible 存在清晰差距，可作为 evidence dependence 的支持证据。")
    return notes


def build_summary_markdown(rows: list[dict]) -> str:
    table = [
        "| Dataset | Condition | Choice Acc | Gated Acc | Gated Coverage |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {dataset} | {condition} | {choice} | {gated} | {coverage} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                choice=_fmt(row.get("choice_acc")),
                gated=_fmt(row.get("gated_acc")),
                coverage=_fmt(row.get("gated_coverage")),
            )
        )
    notes = "\n".join(_risk_notes(rows))
    return f"""# Evidence-Control Summary

## Control Table

{chr(10).join(table)}

## Interpretation Guardrails

{notes}

## Paper Use

- visible > control: can support partial evidence dependence.
- control close to visible: write as language-prior / candidate-bias risk, not as grounded understanding.
- CF F1 remains structured-only and should be narrated separately from caption controls.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize visible/shuffled/masked/hidden evidence controls.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_control_rows(args.outputs_dir)
    md = build_summary_markdown(rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md, encoding="utf-8", newline="\n")
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Evidence-control summary written.")
    print(f"output_md: {output_md}")


if __name__ == "__main__":
    main()

