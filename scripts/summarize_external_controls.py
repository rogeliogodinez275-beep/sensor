from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ["ucihar", "wisdm", "mhealth"]
CONDITIONS = [
    {
        "condition": "visible",
        "choice_suffix": "full",
        "gated_suffix": "",
        "nogate_suffix": "",
    },
    {
        "condition": "shuffled",
        "choice_suffix": "shuffled_evidence",
        "gated_suffix": "shuffled_evidence",
        "nogate_suffix": "shuffled_evidence",
    },
    {
        "condition": "numeric-mask",
        "choice_suffix": "numeric_mask",
        "gated_suffix": "numeric_mask",
        "nogate_suffix": "numeric_mask",
    },
    {
        "condition": "hidden",
        "choice_suffix": "hidden_evidence",
        "gated_suffix": "hidden_evidence",
        "nogate_suffix": "hidden_evidence",
    },
]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def acc(payload: dict | None) -> float | None:
    if not payload or payload.get("caption_selection_accuracy") is None:
        return None
    return float(payload["caption_selection_accuracy"])


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def collect_rows(outputs_dir: Path, model_tag: str) -> list[dict]:
    rows: list[dict] = []
    for dataset in DATASETS:
        visible_acc = None
        for item in CONDITIONS:
            condition = item["condition"]
            choice_suffix = item["choice_suffix"]
            gated_suffix = item["gated_suffix"]
            nogate_suffix = item["nogate_suffix"]
            gated_mid = f"_{gated_suffix}" if gated_suffix else ""
            nogate_mid = f"_{nogate_suffix}" if nogate_suffix else ""
            choice = read_json(
                outputs_dir
                / f"{model_tag}_choice_logprob_{dataset}_hard_v3_constrained_{choice_suffix}_metrics.json"
            )
            gated = read_json(
                outputs_dir
                / f"{model_tag}_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained{gated_mid}_margin2p0_metrics.json"
            )
            nogate = read_json(
                outputs_dir
                / f"{model_tag}_nogate_vote5_choice_logprob_{dataset}_hard_v3_constrained{nogate_mid}_metrics.json"
            )
            choice_acc = acc(choice)
            if condition == "visible":
                visible_acc = choice_acc
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "choice_accuracy": choice_acc,
                    "gated_accuracy": acc(gated),
                    "nogate_accuracy": acc(nogate),
                    "choice_delta_vs_visible": (
                        choice_acc - visible_acc
                        if choice_acc is not None and visible_acc is not None
                        else None
                    ),
                    "n_eval_records": (
                        int(choice["n_eval_records"])
                        if choice and choice.get("n_eval_records") is not None
                        else None
                    ),
                }
            )
    return rows


def write_markdown(payload: dict, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External Evidence-Control Summary",
        "",
        f"Model tag: `{payload['model_tag']}`",
        "",
        "| Dataset | Condition | Choice Acc. | Gated Acc. | No-Gate Acc. | Choice Delta vs Visible | N |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {dataset} | {condition} | {choice} | {gated} | {nogate} | {delta} | {n} |".format(
                dataset=row["dataset"],
                condition=row["condition"],
                choice=fmt(row["choice_accuracy"]),
                gated=fmt(row["gated_accuracy"]),
                nogate=fmt(row["nogate_accuracy"]),
                delta=fmt(row["choice_delta_vs_visible"]),
                n=row["n_eval_records"] if row["n_eval_records"] is not None else "NA",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: visible should be read against shuffled, numeric-mask, and hidden controls. Small visible-control gaps indicate language or candidate priors and require a bounded grounding claim.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external model evidence-control outputs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--model-tag", default="qwen3_4b")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {"model_tag": args.model_tag, "rows": collect_rows(Path(args.outputs_dir), args.model_tag)}
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.output_md))
    print("External evidence-control summary written.")
    print(f"output_md: {args.output_md}")


if __name__ == "__main__":
    main()
