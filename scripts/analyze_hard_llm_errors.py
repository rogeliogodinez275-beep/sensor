from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize hard LLM counterfactual errors by changed fact.")
    parser.add_argument("--records", default="data/benchmark/ucihar_sensorfact_hard_test.jsonl")
    parser.add_argument("--rows", default="outputs/qwen_llm_hard_rows.jsonl")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _legacy_support_items(record: dict) -> list[dict]:
    items = [{"kind": "positive", "changed_fact": None, "changed_facts": []}]
    for item in record["counterfactuals"]:
        changed_facts = item.get("changed_facts")
        if changed_facts is None:
            changed_facts = [item["changed_fact"]] if item.get("changed_fact") else []
        items.append(
            {
                "kind": "counterfactual",
                "changed_fact": item.get("changed_fact"),
                "changed_facts": list(changed_facts),
            }
        )
    return items


def summarize_errors(records: list[dict], rows: list[dict]) -> dict:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    positive_total = 0
    positive_supported = 0
    counterfactual_total = 0
    counterfactual_rejected = 0
    for record, row in zip(records, rows):
        support_items = row.get("support_items") or _legacy_support_items(record)
        predictions = row["support_predictions"]
        for item, prediction in zip(support_items, predictions):
            if item.get("kind") == "positive":
                positive_total += 1
                positive_supported += int(prediction is True)
                continue
            changed_facts = item.get("changed_facts")
            if not changed_facts:
                changed_facts = [item.get("changed_fact") or "unknown"]
            for field in changed_facts:
                stats[str(field)][0] += 1
                stats[str(field)][1] += int(prediction is False)
            counterfactual_total += 1
            counterfactual_rejected += int(prediction is False)
    return {
        "field_stats": stats,
        "positive_total": positive_total,
        "positive_supported": positive_supported,
        "counterfactual_total": counterfactual_total,
        "counterfactual_rejected": counterfactual_rejected,
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.records)
    rows = read_jsonl(args.rows)
    summary = summarize_errors(records, rows)
    stats = summary["field_stats"]
    positive_total = summary["positive_total"]
    positive_supported = summary["positive_supported"]
    counterfactual_total = summary["counterfactual_total"]
    counterfactual_rejected = summary["counterfactual_rejected"]

    print("summary,value")
    print(f"positive_support_rate,{positive_supported / max(1, positive_total):.4f}")
    print(f"counterfactual_rejection_rate,{counterfactual_rejected / max(1, counterfactual_total):.4f}")
    print(
        "statement_accuracy,"
        f"{(positive_supported + counterfactual_rejected) / max(1, positive_total + counterfactual_total):.4f}"
    )
    print()
    print("field,total,rejected,recall")
    for field in sorted(stats):
        total, rejected = stats[field]
        print(f"{field},{total},{rejected},{rejected / max(1, total):.4f}")


if __name__ == "__main__":
    main()
