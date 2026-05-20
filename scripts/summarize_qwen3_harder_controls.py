from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = [
    ("ucihar", "UCI HAR"),
    ("wisdm", "WISDM"),
    ("mhealth", "MHEALTH"),
]

CONTROLS = [
    ("numeric_swap", "numeric_swap"),
    ("axis_permutation", "axis_permutation"),
    ("trend_flip", "trend_flip"),
]

KINDS = [
    (
        "choice",
        "outputs/qwen3_4b_harder_choice_logprob_{dataset}_hard_v3_constrained_{control}_metrics.json",
    ),
    (
        "gated",
        "outputs/qwen3_4b_harder_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained_{control}_margin2p0_metrics.json",
    ),
    (
        "nogate",
        "outputs/qwen3_4b_harder_nogate_vote5_choice_logprob_{dataset}_hard_v3_constrained_{control}_metrics.json",
    ),
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect(root: Path) -> dict:
    rows = []
    missing = []
    for dataset_id, dataset_label in DATASETS:
        for control_id, control_label in CONTROLS:
            metrics = {}
            paths = {}
            for kind, template in KINDS:
                rel = template.format(dataset=dataset_id, control=control_id)
                path = root / rel
                paths[kind] = rel
                if not path.exists():
                    missing.append(rel)
                    continue
                metrics[kind] = read_json(path)
            if len(metrics) == len(KINDS):
                rows.append(
                    {
                        "dataset": dataset_label,
                        "dataset_id": dataset_id,
                        "control": control_label,
                        "choice_accuracy": metrics["choice"]["caption_selection_accuracy"],
                        "gated_accuracy": metrics["gated"]["caption_selection_accuracy"],
                        "nogate_accuracy": metrics["nogate"]["caption_selection_accuracy"],
                        "gate_overrides": metrics["gated"].get("caption_gate_alternate_count"),
                        "n_eval_records": metrics["choice"].get("n_eval_records"),
                        "paths": paths,
                    }
                )
    return {"rows": rows, "missing": missing}


def render_markdown(payload: dict) -> str:
    lines = [
        "# Qwen3 Harder Evidence Controls Summary",
        "",
        "This table reports Qwen3-4B cross-model confirmation for the harder evidence controls.",
        "",
        "| Dataset | Control | Forced-choice Acc | Gated Acc | No-gate Acc | Gate overrides | N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {dataset} | `{control}` | {choice:.4f} | {gated:.4f} | {nogate:.4f} | {overrides} | {n} |".format(
                dataset=row["dataset"],
                control=row["control"],
                choice=row["choice_accuracy"],
                gated=row["gated_accuracy"],
                nogate=row["nogate_accuracy"],
                overrides=row["gate_overrides"],
                n=row["n_eval_records"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `axis_permutation` produces the largest degradation across all datasets, which independently confirms that axis/channel evidence is a major decision signal.",
            "- `numeric_swap` and `trend_flip` preserve high accuracy on MHEALTH and remain much less damaging than axis permutation, so the grounding claim should remain bounded.",
            "- This supports cross-model evidence-dependence for axis/channel fields, while still exposing language-prior or template-dependence risk for numeric magnitude and trend fields.",
        ]
    )
    if payload["missing"]:
        lines.extend(["", "## Missing Files", ""])
        lines.extend(f"- `{path}`" for path in payload["missing"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3 harder evidence-control metrics.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    payload = collect(root)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8", newline="\n")
    print("Qwen3 harder-control summary written.")
    print(f"rows: {len(payload['rows'])}")
    print(f"missing: {len(payload['missing'])}")
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")


if __name__ == "__main__":
    main()

