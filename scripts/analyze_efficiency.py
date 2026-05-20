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


def collect_efficiency_rows(outputs_dir: str | Path) -> list[dict]:
    root = Path(outputs_dir)
    rows = []
    for dataset, label in DATASETS.items():
        gated = _read_json(root / f"coder_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained_margin2_metrics.json")
        choice = _read_json(root / f"coder_choice_logprob_{dataset}_hard_v3_constrained_full_metrics.json")
        n_records = _metric(gated, "n_eval_records")
        alternate_count = _metric(gated, "caption_gate_alternate_count")
        coverage = None if n_records in (None, 0.0) or alternate_count is None else alternate_count / n_records
        rows.append(
            {
                "dataset": label,
                "dataset_id": dataset,
                "n_eval_records": n_records,
                "full_choice_prompts": _metric(choice, "n_scoring_prompts"),
                "effective_override_prompts": alternate_count,
                "gate_coverage": coverage,
            }
        )
    return rows


def build_efficiency_markdown(rows: list[dict]) -> str:
    table = [
        "| Dataset | N | Full Choice Prompts | Override Prompts | Gate Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| {dataset} | {n} | {full} | {override} | {coverage} |".format(
                dataset=row["dataset"],
                n=_fmt(row.get("n_eval_records")),
                full=_fmt(row.get("full_choice_prompts")),
                override=_fmt(row.get("effective_override_prompts")),
                coverage=_fmt(row.get("gate_coverage")),
            )
        )
    return f"""# Efficiency Analysis

## Reranker Cost Proxy

{chr(10).join(table)}

## Interpretation

The full forced-choice scorer requires one prompt per evaluated window. The deployed gated method should be discussed by override coverage: only rows above the margin threshold change the vote5 decision, even though the current offline analysis scores all rows to estimate the gate.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize reranker efficiency and gate coverage.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_efficiency_rows(args.outputs_dir)
    md = build_efficiency_markdown(rows)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8", newline="\n")
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Efficiency analysis written.")
    print(f"output_md: {out_md}")


if __name__ == "__main__":
    main()

