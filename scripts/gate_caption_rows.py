from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy


def _score_margin(row: dict) -> float:
    scores = sorted((float(score) for score in row.get("caption_scores", [])), reverse=True)
    if len(scores) < 2:
        return 0.0
    return float(scores[0] - scores[1])


def _map_prediction(row: dict, *, strict_map: bool = False) -> tuple[int | None, bool]:
    prediction = row.get("caption_prediction")
    if prediction is None:
        return None, False
    prediction = int(prediction)
    index_map = row.get("candidate_index_map")
    if isinstance(index_map, list):
        try:
            return int(index_map[prediction]), False
        except (IndexError, TypeError, ValueError):
            if strict_map:
                return None, True
            return prediction, True
    return prediction, False


def gate_caption_rows(
    primary_rows: list[dict],
    alternate_rows: list[dict],
    *,
    min_alternate_margin: float,
) -> list[dict]:
    alternate_by_window = {str(row["window_id"]): row for row in alternate_rows}
    rows: list[dict] = []
    for primary in primary_rows:
        alternate = alternate_by_window.get(str(primary["window_id"]))
        alternate_prediction, bad_alternate_map = (
            _map_prediction(alternate, strict_map=True) if alternate is not None else (None, False)
        )
        use_alternate = (
            alternate is not None
            and not bad_alternate_map
            and alternate_prediction is not None
            and _score_margin(alternate) > min_alternate_margin
        )
        chosen = alternate if use_alternate else primary
        row = json.loads(json.dumps(primary))
        primary_prediction, _ = _map_prediction(primary)
        row["caption_prediction"] = alternate_prediction if use_alternate else primary_prediction
        row["caption_scores"] = chosen.get("caption_scores", primary.get("caption_scores"))
        row["caption_gate_source"] = "alternate" if use_alternate else "primary"
        row["caption_gate_alternate_margin"] = None if alternate is None else _score_margin(alternate)
        row["caption_gate_bad_alternate_map"] = bad_alternate_map
        rows.append(row)
    return rows


def evaluate_gated_rows(rows: list[dict], system_name: str) -> dict:
    examples = [
        {
            "answer_index": int(row["caption_answer_index"]),
            "scores": [
                1.0 if idx == row.get("caption_prediction") else 0.0
                for idx in range(max(int(row["caption_answer_index"]), int(row.get("caption_prediction") or 0)) + 1)
            ],
        }
        for row in rows
    ]
    return {
        "system": system_name,
        "caption_selection_accuracy": caption_selection_accuracy(examples),
        "n_eval_records": len(rows),
        "caption_gate_alternate_count": sum(1 for row in rows if row.get("caption_gate_source") == "alternate"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate caption rows between a primary and alternate predictor.")
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--alternate-rows", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--min-alternate-margin", type=float, required=True)
    parser.add_argument("--system-name", default="gated_caption_rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = gate_caption_rows(
        list(read_jsonl(args.primary_rows)),
        list(read_jsonl(args.alternate_rows)),
        min_alternate_margin=args.min_alternate_margin,
    )
    write_jsonl(args.output_rows, rows)
    metrics = evaluate_gated_rows(rows, args.system_name)
    metrics["primary_rows"] = args.primary_rows
    metrics["alternate_rows"] = args.alternate_rows
    metrics["min_alternate_margin"] = args.min_alternate_margin
    write_json(args.output_metrics, metrics)
    print("Caption gating finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
