from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_json


def _candidate_fact(record: dict | None, index: int | None) -> str:
    if record is None or index is None:
        return "unknown"
    candidates = record.get("caption_selection", {}).get("candidates", [])
    if index < 0 or index >= len(candidates):
        return "out_of_range"
    candidate = candidates[index]
    facts = candidate.get("changed_facts")
    if isinstance(facts, list) and facts:
        return "+".join(str(item) for item in facts)
    fact = candidate.get("changed_fact")
    return "supported" if fact is None else str(fact)


def _summarize(groups: dict[str, list[bool]]) -> dict[str, dict]:
    return {
        key: {"n": len(values), "accuracy": sum(values) / max(1, len(values))}
        for key, values in sorted(groups.items())
    }


def analyze_per_fact_breakdown(rows: list[dict], benchmark_records: list[dict]) -> dict:
    benchmark_by_window = {str(record["window_id"]): record for record in benchmark_records}
    by_gold: dict[str, list[bool]] = defaultdict(list)
    by_pred_wrong: dict[str, list[bool]] = defaultdict(list)
    correct_flags: list[bool] = []

    for row in rows:
        window_id = str(row["window_id"])
        benchmark = benchmark_by_window.get(window_id)
        answer = int(row["caption_answer_index"])
        prediction = None if row.get("caption_prediction") is None else int(row["caption_prediction"])
        correct = prediction == answer
        correct_flags.append(correct)
        by_gold[_candidate_fact(benchmark, answer)].append(correct)
        if not correct:
            by_pred_wrong[_candidate_fact(benchmark, prediction)].append(correct)

    return {
        "overall": {
            "n": len(correct_flags),
            "accuracy": sum(correct_flags) / max(1, len(correct_flags)),
        },
        "by_gold_changed_fact": _summarize(by_gold),
        "by_predicted_wrong_fact": _summarize(by_pred_wrong),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Per-Fact Breakdown",
        "",
        f"Overall accuracy: {report['overall']['accuracy']:.4f} over {report['overall']['n']} rows.",
        "",
        "## By Gold Fact",
        "",
        "| Fact | N | Accuracy |",
        "|---|---:|---:|",
    ]
    for fact, item in report["by_gold_changed_fact"].items():
        lines.append(f"| {fact} | {item['n']} | {item['accuracy']:.4f} |")
    lines.extend(["", "## By Predicted Wrong Fact", "", "| Fact | N | Accuracy |", "|---|---:|---:|"])
    for fact, item in report["by_predicted_wrong_fact"].items():
        lines.append(f"| {fact} | {item['n']} | {item['accuracy']:.4f} |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze caption accuracy by candidate changed_fact.")
    parser.add_argument("--rows", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.rows))
    benchmark = list(read_jsonl(args.benchmark))
    report = analyze_per_fact_breakdown(rows, benchmark)
    report["rows"] = args.rows
    report["benchmark"] = args.benchmark
    write_json(args.output_json, report)
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print("Per-fact breakdown written.")
    print(json.dumps(report["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()

