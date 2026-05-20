from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def candidate_balance_features(row: dict) -> dict:
    selection = row.get("caption_selection") or {}
    candidates = selection.get("candidates") or []
    evidence_tokens = tokenize(str(row.get("evidence_text") or ""))
    lengths: list[float] = []
    overlaps: list[float] = []
    for candidate in candidates:
        tokens = tokenize(str(candidate.get("text") or ""))
        lengths.append(float(len(tokens)))
        if tokens:
            overlaps.append(len(tokens & evidence_tokens) / len(tokens))
        else:
            overlaps.append(0.0)
    length_range = max(lengths) - min(lengths) if lengths else 0.0
    overlap_range = max(overlaps) - min(overlaps) if overlaps else 0.0
    return {
        "window_id": str(row["window_id"]),
        "candidate_count": len(candidates),
        "length_range": length_range,
        "length_std": _std(lengths),
        "evidence_overlap_range": overlap_range,
        "evidence_overlap_std": _std(overlaps),
        "balance_score": length_range / 20.0 + overlap_range,
    }


def select_balanced_features(
    features: list[dict],
    *,
    target_fraction: float,
    max_length_range: float | None,
    max_overlap_range: float | None,
) -> list[dict]:
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    filtered = []
    for row in features:
        if max_length_range is not None and row["length_range"] > max_length_range:
            continue
        if max_overlap_range is not None and row["evidence_overlap_range"] > max_overlap_range:
            continue
        filtered.append(row)
    if not filtered:
        return []
    n_select = max(1, int(round(len(filtered) * target_fraction)))
    return sorted(
        filtered,
        key=lambda row: (
            float(row["balance_score"]),
            float(row["length_range"]),
            float(row["evidence_overlap_range"]),
            str(row["window_id"]),
        ),
    )[:n_select]


def evaluate_rows(rows: list[dict], selected_ids: set[str]) -> dict:
    subset = [row for row in rows if str(row.get("window_id")) in selected_ids]
    correct = sum(
        1
        for row in subset
        if row.get("caption_prediction") is not None
        and row.get("caption_answer_index") is not None
        and int(row["caption_prediction"]) == int(row["caption_answer_index"])
    )
    return {
        "n_eval_records": len(subset),
        "caption_selection_accuracy": correct / len(subset) if subset else None,
    }


def parse_row_set(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError("--row-set must use NAME=PATH")
    name, path = text.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError("--row-set must use non-empty NAME=PATH")
    return name.strip(), Path(path.strip())


def build_summary(
    *,
    benchmark_path: Path,
    row_sets: list[tuple[str, Path]],
    target_fraction: float,
    max_length_range: float | None,
    max_overlap_range: float | None,
) -> dict:
    benchmark_rows = read_jsonl(benchmark_path)
    features = [candidate_balance_features(row) for row in benchmark_rows]
    selected_features = select_balanced_features(
        features,
        target_fraction=target_fraction,
        max_length_range=max_length_range,
        max_overlap_range=max_overlap_range,
    )
    selected_ids = {row["window_id"] for row in selected_features}
    systems = []
    for name, path in row_sets:
        rows = read_jsonl(path)
        full_metrics = evaluate_rows(rows, {str(row.get("window_id")) for row in rows})
        subset_metrics = evaluate_rows(rows, selected_ids)
        full_acc = full_metrics["caption_selection_accuracy"]
        subset_acc = subset_metrics["caption_selection_accuracy"]
        systems.append(
            {
                "name": name,
                "path": str(path),
                "full_accuracy": full_acc,
                "balanced_accuracy": subset_acc,
                "delta_balanced_minus_full": (
                    subset_acc - full_acc if subset_acc is not None and full_acc is not None else None
                ),
                "balanced_n": subset_metrics["n_eval_records"],
                "full_n": full_metrics["n_eval_records"],
            }
        )
    return {
        "benchmark_path": str(benchmark_path),
        "selection": {
            "total_records": len(features),
            "selected_records": len(selected_features),
            "target_fraction": target_fraction,
            "max_length_range": max_length_range,
            "max_overlap_range": max_overlap_range,
            "mean_selected_length_range": (
                sum(row["length_range"] for row in selected_features) / len(selected_features)
                if selected_features
                else None
            ),
            "mean_selected_overlap_range": (
                sum(row["evidence_overlap_range"] for row in selected_features) / len(selected_features)
                if selected_features
                else None
            ),
        },
        "systems": systems,
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def write_markdown(payload: dict, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    selection = payload["selection"]
    lines = [
        "# Balanced Candidate Subset Analysis",
        "",
        f"Benchmark: `{payload['benchmark_path']}`",
        "",
        "| Total | Selected | Target Fraction | Mean Length Range | Mean Evidence-Overlap Range |",
        "|---:|---:|---:|---:|---:|",
        "| {total} | {selected} | {fraction:.2f} | {len_range} | {overlap_range} |".format(
            total=selection["total_records"],
            selected=selection["selected_records"],
            fraction=selection["target_fraction"],
            len_range=fmt(selection["mean_selected_length_range"]),
            overlap_range=fmt(selection["mean_selected_overlap_range"]),
        ),
        "",
        "| System | Full Acc. | Balanced Acc. | Delta | Balanced N | Full N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["systems"]:
        lines.append(
            "| {name} | {full} | {balanced} | {delta} | {balanced_n} | {full_n} |".format(
                name=row["name"],
                full=fmt(row["full_accuracy"]),
                balanced=fmt(row["balanced_accuracy"]),
                delta=fmt(row["delta_balanced_minus_full"]),
                balanced_n=row["balanced_n"],
                full_n=row["full_n"],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: this subset reduces obvious candidate-length and evidence-overlap imbalance. It is a diagnostic control, not a replacement for the main benchmark.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze accuracy on a balanced candidate subset.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--row-set", action="append", default=[], help="NAME=PATH; can be repeated.")
    parser.add_argument("--target-fraction", type=float, default=0.5)
    parser.add_argument("--max-length-range", type=float, default=None)
    parser.add_argument("--max-overlap-range", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row_sets = [parse_row_set(item) for item in args.row_set]
    payload = build_summary(
        benchmark_path=Path(args.benchmark),
        row_sets=row_sets,
        target_fraction=args.target_fraction,
        max_length_range=args.max_length_range,
        max_overlap_range=args.max_overlap_range,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(payload, Path(args.output_md))
    print("Balanced candidate subset analysis written.")
    print(json.dumps(payload["selection"], ensure_ascii=False))


if __name__ == "__main__":
    main()
