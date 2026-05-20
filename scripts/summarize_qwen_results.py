from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen evaluation metrics.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--output-md", default="outputs/qwen_results_summary.md")
    parser.add_argument("--output-csv", default="outputs/qwen_results_summary.csv")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.outputs)
    rows = [
        ("Easy", "Qwen Embedding", read_json(output_dir / "qwen_embedding_metrics.json")),
        ("Easy", "Qwen 4B LLM", read_json(output_dir / "qwen_llm_metrics.json")),
        ("Hard v1", "Qwen Embedding", read_json(output_dir / "qwen_embedding_hard_metrics.json")),
        ("Hard v1", "Qwen 4B LLM", read_json(output_dir / "qwen_llm_hard_metrics.json")),
        ("Hard v2", "Qwen Embedding", read_json(output_dir / "qwen_embedding_hard_v2_metrics.json")),
        ("Hard v2", "Qwen 4B LLM", read_json(output_dir / "qwen_llm_hard_v2_metrics.json")),
    ]
    headers = [
        "Benchmark",
        "System",
        "N",
        "Caption Acc",
        "CF Acc",
        "CF F1",
        "Pairwise CF Acc",
        "Parse Success",
    ]
    table_rows = []
    for benchmark, system, metrics in rows:
        if not metrics:
            continue
        table_rows.append(
            [
                benchmark,
                system,
                fmt(metrics.get("n_eval_records")),
                fmt(metrics.get("caption_selection_accuracy")),
                fmt(metrics.get("cf_reject_accuracy")),
                fmt(metrics.get("cf_reject_f1")),
                fmt(metrics.get("cf_pairwise_reject_accuracy")),
                fmt(metrics.get("parse_success_rate")),
            ]
        )
    markdown = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown.extend("| " + " | ".join(row) + " |" for row in table_rows)
    Path(args.output_md).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    csv_lines = [",".join(headers)]
    csv_lines.extend(",".join(row) for row in table_rows)
    Path(args.output_csv).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(Path(args.output_md).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
