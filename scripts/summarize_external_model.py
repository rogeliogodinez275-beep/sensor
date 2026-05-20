from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = ["ucihar", "wisdm", "mhealth"]


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_value(payload: dict | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def first_metric_value(payload: dict | None, keys: list[str]) -> float | None:
    for key in keys:
        value = metric_value(payload, key)
        if value is not None:
            return value
    return None


def mcnemar_midp(payload: dict | None) -> float | None:
    value = first_metric_value(payload, ["mcnemar_midp", "mcnemar_mid_p"])
    if value is not None:
        return value
    if payload and isinstance(payload.get("mcnemar"), dict):
        nested = payload["mcnemar"].get("midp_value")
        if nested is not None:
            return float(nested)
    return None


def format_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def summarize(outputs_dir: Path, model_tag: str) -> dict:
    rows: list[dict] = []
    for dataset in DATASETS:
        choice = read_json(
            outputs_dir
            / f"{model_tag}_choice_logprob_{dataset}_hard_v3_constrained_full_metrics.json"
        )
        gated = read_json(
            outputs_dir
            / f"{model_tag}_gated_vote5_choice_logprob_{dataset}_hard_v3_constrained_margin2p0_metrics.json"
        )
        nogate = read_json(
            outputs_dir
            / f"{model_tag}_nogate_vote5_choice_logprob_{dataset}_hard_v3_constrained_metrics.json"
        )
        paired = read_json(
            outputs_dir
            / f"paired_label_significance_{dataset}_{model_tag}_vote5_vs_gated_choice_logprob.json"
        )
        choice_acc = metric_value(choice, "caption_selection_accuracy")
        gated_acc = metric_value(gated, "caption_selection_accuracy")
        nogate_acc = metric_value(nogate, "caption_selection_accuracy")
        gate_count = metric_value(gated, "caption_gate_alternate_count")
        n_records = metric_value(gated, "n_eval_records") or metric_value(choice, "n_eval_records")
        delta = None
        p_value = None
        if paired:
            delta = first_metric_value(paired, ["delta_accuracy", "delta"])
            p_value = mcnemar_midp(paired)
        rows.append(
            {
                "dataset": dataset,
                "choice_accuracy": choice_acc,
                "gated_accuracy": gated_acc,
                "nogate_accuracy": nogate_acc,
                "delta_vs_vote5": delta,
                "mcnemar_midp": p_value,
                "gate_alternate_count": int(gate_count) if gate_count is not None else None,
                "n_eval_records": int(n_records) if n_records is not None else None,
            }
        )
    return {"model_tag": model_tag, "rows": rows}


def write_markdown(payload: dict, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External Cross-Model Reranker Summary",
        "",
        f"Model tag: `{payload['model_tag']}`",
        "",
        "| Dataset | Choice Acc. | Gated Acc. | No-Gate Acc. | Delta vs Vote5 | McNemar mid-p | Gate Count | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            "| {dataset} | {choice} | {gated} | {nogate} | {delta} | {p} | {gate} | {n} |".format(
                dataset=row["dataset"],
                choice=format_float(row["choice_accuracy"]),
                gated=format_float(row["gated_accuracy"]),
                nogate=format_float(row["nogate_accuracy"]),
                delta=format_float(row["delta_vs_vote5"]),
                p=format_float(row["mcnemar_midp"]),
                gate=row["gate_alternate_count"] if row["gate_alternate_count"] is not None else "NA",
                n=row["n_eval_records"] if row["n_eval_records"] is not None else "NA",
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this table is a cross-model robustness check for the selective reranking claim. It should not be used to attribute CF support or rejection gains to the reranker; those remain owned by the structured verifier.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize external cross-model reranker outputs.")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--model-tag", default="qwen3_4b")
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(Path(args.outputs_dir), args.model_tag)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.output_md))
    print("External model summary written.")
    print(f"output_md: {args.output_md}")


if __name__ == "__main__":
    main()
