from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_json


def _candidate_fact(record: dict | None, prediction: int | None) -> str:
    if record is None or prediction is None:
        return "unknown"
    candidates = record.get("caption_selection", {}).get("candidates", [])
    if prediction < 0 or prediction >= len(candidates):
        return "out_of_range"
    candidate = candidates[prediction]
    facts = candidate.get("changed_facts")
    if isinstance(facts, list) and facts:
        return "+".join(str(item) for item in facts)
    fact = candidate.get("changed_fact")
    return "supported" if fact is None else str(fact)


def _row_by_window(rows: list[dict]) -> dict[str, dict]:
    return {str(row["window_id"]): row for row in rows}


def analyze_failure_taxonomy(
    primary_rows: list[dict],
    gated_rows: list[dict],
    *,
    benchmark_records: list[dict] | None = None,
    max_examples: int = 8,
) -> dict:
    gated_by_window = _row_by_window(gated_rows)
    benchmark_by_window = _row_by_window(benchmark_records or [])
    corrected = []
    regressed = []
    unchanged_wrong = 0
    unchanged_correct = 0
    corrected_facts: Counter[str] = Counter()
    regressed_facts: Counter[str] = Counter()

    for primary in primary_rows:
        window_id = str(primary["window_id"])
        gated = gated_by_window.get(window_id)
        if gated is None:
            continue
        answer = int(primary["caption_answer_index"])
        primary_pred = None if primary.get("caption_prediction") is None else int(primary["caption_prediction"])
        gated_pred = None if gated.get("caption_prediction") is None else int(gated["caption_prediction"])
        primary_correct = primary_pred == answer
        gated_correct = gated_pred == answer
        benchmark = benchmark_by_window.get(window_id)

        if not primary_correct and gated_correct:
            fact = _candidate_fact(benchmark, primary_pred)
            corrected_facts[fact] += 1
            if len(corrected) < max_examples:
                corrected.append(
                    {
                        "window_id": window_id,
                        "answer_index": answer,
                        "primary_prediction": primary_pred,
                        "gated_prediction": gated_pred,
                        "primary_wrong_fact": fact,
                        "caption_gate_source": gated.get("caption_gate_source"),
                    }
                )
        elif primary_correct and not gated_correct:
            fact = _candidate_fact(benchmark, gated_pred)
            regressed_facts[fact] += 1
            if len(regressed) < max_examples:
                regressed.append(
                    {
                        "window_id": window_id,
                        "answer_index": answer,
                        "primary_prediction": primary_pred,
                        "gated_prediction": gated_pred,
                        "gated_wrong_fact": fact,
                        "caption_gate_source": gated.get("caption_gate_source"),
                    }
                )
        elif primary_correct and gated_correct:
            unchanged_correct += 1
        else:
            unchanged_wrong += 1

    return {
        "summary": {
            "n_primary_rows": len(primary_rows),
            "n_gated_rows": len(gated_rows),
            "corrected_count": len(corrected) + sum(corrected_facts.values()) - len(corrected),
            "regressed_count": len(regressed) + sum(regressed_facts.values()) - len(regressed),
            "unchanged_correct_count": unchanged_correct,
            "unchanged_wrong_count": unchanged_wrong,
        },
        "corrected_by_primary_wrong_fact": dict(corrected_facts),
        "regressed_by_gated_wrong_fact": dict(regressed_facts),
        "examples": {
            "corrected": corrected,
            "regressed": regressed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze vote5 vs gated correction/failure taxonomy.")
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--gated-rows", required=True)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = list(read_jsonl(args.primary_rows))
    gated_rows = list(read_jsonl(args.gated_rows))
    benchmark_records = list(read_jsonl(args.benchmark)) if args.benchmark else None
    report = analyze_failure_taxonomy(
        primary_rows,
        gated_rows,
        benchmark_records=benchmark_records,
        max_examples=args.max_examples,
    )
    report["primary_rows"] = args.primary_rows
    report["gated_rows"] = args.gated_rows
    report["benchmark"] = args.benchmark
    write_json(args.output_json, report)
    print("Failure taxonomy analysis finished.")
    for key, value in report["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

