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


def collect_position_prior_rows(
    outputs_dir: str | Path,
    *,
    position_prefix: str = "coder_position_prior",
    choice_prefix: str = "coder_choice_logprob",
) -> list[dict]:
    root = Path(outputs_dir)
    rows = []
    for dataset, label in DATASETS.items():
        prior = _read_json(root / f"{position_prefix}_{dataset}_hard_v3_constrained_metrics.json")
        choice = _read_json(root / f"{choice_prefix}_{dataset}_hard_v3_constrained_full_metrics.json")
        prior_acc = _metric(prior, "caption_selection_accuracy")
        choice_acc = _metric(choice, "caption_selection_accuracy")
        rows.append(
            {
                "dataset": label,
                "dataset_id": dataset,
                "position_prior_acc": prior_acc,
                "choice_acc": choice_acc,
                "choice_minus_position": None if prior_acc is None or choice_acc is None else choice_acc - prior_acc,
                "n_eval_records": _metric(prior, "n_eval_records"),
            }
        )
    return rows


def build_position_prior_markdown(rows: list[dict]) -> str:
    table = [
        "| Dataset | Position-Prior Acc | Full Choice Acc | Choice - Position | N |",
        "|---|---:|---:|---:|---:|",
    ]
    high_risk = []
    for row in rows:
        table.append(
            "| {dataset} | {prior} | {choice} | {delta} | {n} |".format(
                dataset=row["dataset"],
                prior=_fmt(row.get("position_prior_acc")),
                choice=_fmt(row.get("choice_acc")),
                delta=_fmt(row.get("choice_minus_position")),
                n=_fmt(row.get("n_eval_records")),
            )
        )
        prior_acc = row.get("position_prior_acc")
        if prior_acc is not None and prior_acc >= 0.35:
            high_risk.append(f"- {row['dataset']} 的位置先验较高，候选顺序/答案索引偏置必须在论文中控制。")
    if not high_risk:
        high_risk.append("- 位置先验未显示明显偏置，但仍应作为 control 汇报。")
    return f"""# Position-Prior Baseline Summary

## Results

{chr(10).join(table)}

## Interpretation

{chr(10).join(high_risk)}

This baseline sees only option labels, not evidence or candidate text. It should be reported as an answer-index / position-prior control, not as a sensor-language model.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize position-prior answer-index baseline.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--position-prefix", default="coder_position_prior")
    parser.add_argument("--choice-prefix", default="coder_choice_logprob")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_position_prior_rows(
        args.outputs_dir,
        position_prefix=args.position_prefix,
        choice_prefix=args.choice_prefix,
    )
    md = build_position_prior_markdown(rows)
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8", newline="\n")
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Position-prior summary written.")
    print(f"output_md: {out_md}")


if __name__ == "__main__":
    main()
