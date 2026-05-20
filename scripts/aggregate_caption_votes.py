from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy


def _map_prediction(row: dict) -> int | None:
    prediction = row.get("caption_prediction")
    if prediction is None:
        return None
    prediction = int(prediction)
    index_map = row.get("candidate_index_map")
    if isinstance(index_map, list):
        try:
            return int(index_map[prediction])
        except (IndexError, TypeError, ValueError):
            return prediction
    return prediction


def _map_answer(row: dict) -> int:
    answer = int(row["caption_answer_index"])
    index_map = row.get("candidate_index_map")
    if isinstance(index_map, list):
        try:
            return int(index_map[answer])
        except (IndexError, TypeError, ValueError):
            return answer
    return answer


def _candidate_count(row: dict) -> int:
    index_map = row.get("candidate_index_map")
    if isinstance(index_map, list) and index_map:
        return max(int(item) for item in index_map) + 1
    answer = row.get("caption_answer_index")
    pred = row.get("caption_prediction")
    max_index = max(int(answer or 0), int(pred or 0))
    return max_index + 1


def aggregate_caption_rows(rows_by_run: list[list[dict]], *, system_name: str) -> tuple[dict, list[dict]]:
    if not rows_by_run:
        return {
            "system": system_name,
            "caption_selection_accuracy": 0.0,
            "n_eval_records": 0,
            "vote_size": 0,
        }, []

    ordered_rows = rows_by_run[0]
    rows_by_window = [{str(row["window_id"]): row for row in run_rows} for run_rows in rows_by_run]
    aggregated_rows: list[dict] = []
    selection_examples: list[dict] = []

    for base_row in ordered_rows:
        window_id = str(base_row["window_id"])
        votes = []
        for run_rows in rows_by_window:
            row = run_rows.get(window_id)
            if row is None:
                continue
            prediction = _map_prediction(row)
            if prediction is not None:
                votes.append(prediction)
        if not votes:
            prediction = None
        else:
            counts = Counter(votes)
            top_count = max(counts.values())
            tied = {pred for pred, count in counts.items() if count == top_count}
            prediction = next(vote for vote in votes if vote in tied)

        candidate_count = _candidate_count(base_row)
        caption_scores = [0.0 for _ in range(candidate_count)]
        if prediction is not None and 0 <= prediction < candidate_count:
            caption_scores[prediction] = 1.0
        answer_index = _map_answer(base_row)
        selection_examples.append({"answer_index": answer_index, "scores": caption_scores})
        aggregated_rows.append(
            {
                "window_id": window_id,
                "candidate_index_map": None,
                "caption_prediction": prediction,
                "caption_answer_index": answer_index,
                "caption_scores": caption_scores,
                "vote_predictions": votes,
            }
        )

    metrics = {
        "system": system_name,
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "n_eval_records": len(aggregated_rows),
        "vote_size": len(rows_by_run),
        "support_evaluated": False,
    }
    return metrics, aggregated_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiple caption-only direct runs by majority vote.")
    parser.add_argument("--input-rows", nargs="+", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--system-name", default="caption_vote")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_by_run = [list(read_jsonl(path)) for path in args.input_rows]
    metrics, rows = aggregate_caption_rows(rows_by_run, system_name=args.system_name)
    write_json(args.output_metrics, metrics)
    write_jsonl(args.output_rows, rows)
    print("Caption vote aggregation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
