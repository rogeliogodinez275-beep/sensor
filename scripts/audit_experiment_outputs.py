from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.audit import (
    answer_position_counts,
    changed_fact_counts,
    forbidden_shortcut_hits,
    subject_overlap,
)
from sensorfact.io import read_jsonl


SHORTCUT_PHRASES = [
    "supported description",
    "description claims",
    "this description",
    "supported by",
    "unsupported",
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return list(read_jsonl(path))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_dict(data: dict) -> str:
    if not data:
        return "{}"
    return ", ".join(f"{key}:{value}" for key, value in sorted(data.items()))


def metric_row(name: str, metrics: dict) -> str:
    if not metrics:
        return f"| {name} | missing |  |  |  |  |"
    return (
        f"| {name} | {metrics.get('n_eval_records', 'N/A')} | "
        f"{metrics.get('caption_selection_accuracy', 'N/A')} | "
        f"{metrics.get('cf_reject_accuracy', 'N/A')} | "
        f"{metrics.get('cf_reject_f1', 'N/A')} | "
        f"{metrics.get('parse_success_rate', metrics.get('cf_pairwise_reject_accuracy', 'N/A'))} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit benchmark leakage and Qwen result files.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="outputs/experiment_audit_report.md")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    benchmark_dir = workspace / "data" / "benchmark"
    processed_dir = workspace / "data" / "processed"
    outputs = workspace / "outputs"
    benchmark_names = [
        "ucihar_sensorfact_hard_v2_test",
        "ucihar_sensorfact_hard_v3_test",
        "wisdm_sensorfact_hard_v2_test",
        "wisdm_sensorfact_hard_v3_test",
    ]
    lines = ["# Experiment Audit Report", ""]
    lines.append("## Benchmark Leakage Checks")
    lines.append("")
    lines.append(
        "| Benchmark | Records | Answer Positions | Shortcut Hits | Changed Facts |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for name in benchmark_names:
        records = read_rows(benchmark_dir / f"{name}.jsonl")
        lines.append(
            f"| {name} | {len(records)} | {fmt_dict(answer_position_counts(records))} | "
            f"{fmt_dict(forbidden_shortcut_hits(records, SHORTCUT_PHRASES))} | "
            f"{fmt_dict(changed_fact_counts(records))} |"
        )

    lines.extend(["", "## Subject Split Checks", ""])
    lines.append("| Dataset | Train | Test | Overlap |")
    lines.append("| --- | --- | --- | --- |")
    for dataset in ["ucihar", "wisdm"]:
        train = read_rows(processed_dir / f"{dataset}_train.jsonl")
        test = read_rows(processed_dir / f"{dataset}_test.jsonl")
        overlap = subject_overlap(train, test)
        lines.append(f"| {dataset} | {len(train)} | {len(test)} | {len(overlap)} |")

    metric_files = [
        ("UCI hard v2 embedding", "qwen_embedding_ucihar_hard_v2_metrics.json"),
        ("UCI hard v2 LLM", "qwen_llm_ucihar_hard_v2_metrics.json"),
        ("UCI hard v3 embedding", "qwen_embedding_ucihar_hard_v3_metrics.json"),
        ("UCI hard v3 LLM", "qwen_llm_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v2 embedding", "qwen_embedding_wisdm_hard_v2_metrics.json"),
        ("WISDM hard v2 LLM", "qwen_llm_wisdm_hard_v2_metrics.json"),
        ("WISDM hard v3 embedding", "qwen_embedding_wisdm_hard_v3_metrics.json"),
        ("WISDM hard v3 LLM", "qwen_llm_wisdm_hard_v3_metrics.json"),
    ]
    lines.extend(["", "## Qwen Metrics", ""])
    lines.append("| Run | N | Caption Acc | CF Acc | CF F1 | Parse/Pairwise |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for title, filename in metric_files:
        lines.append(metric_row(title, read_json(outputs / filename)))

    output_path = workspace / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
