from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_jsonl
from sensorfact.qwen_llm_eval import record_evidence_text


def _row_index(rows: list[dict] | None) -> dict[str, dict]:
    return {str(row.get("window_id")): row for row in rows or []}


def _caption_prediction(row: dict | None) -> int | None:
    if not row:
        return None
    for key in ("caption_prediction", "prediction", "predicted_index"):
        if key in row and row[key] is not None:
            return int(row[key])
    return None


def _annotation_row(record: dict, dataset_id: str, reason: str) -> dict:
    caption = record.get("positive", {}).get("text", "")
    return {
        "dataset_id": dataset_id,
        "window_id": record.get("window_id"),
        "label": record.get("label"),
        "selection_reason": reason,
        "evidence_text": record_evidence_text(record),
        "model_caption": caption,
        "candidate_texts": " ||| ".join(
            item.get("text", "") for item in record.get("caption_selection", {}).get("candidates", [])
        ),
        "gold_answer_index": record.get("caption_selection", {}).get("answer_index"),
        "human_caption": "",
        "support_evidence": "",
        "counterfactual_validity": "",
        "annotator_notes": "",
    }


def build_annotation_subset(
    records: list[dict],
    dataset_id: str,
    target_count: int,
    seed: int = 42,
    vote_rows: list[dict] | None = None,
    gated_rows: list[dict] | None = None,
) -> list[dict]:
    vote_by_id = _row_index(vote_rows)
    gated_by_id = _row_index(gated_rows)
    disagreement: list[dict] = []
    fallback: list[dict] = []
    for record in records:
        window_id = str(record.get("window_id"))
        vote_pred = _caption_prediction(vote_by_id.get(window_id))
        gated_pred = _caption_prediction(gated_by_id.get(window_id))
        if vote_pred is not None and gated_pred is not None and vote_pred != gated_pred:
            disagreement.append(_annotation_row(record, dataset_id, "vote_gated_disagreement"))
        else:
            fallback.append(_annotation_row(record, dataset_id, "random_hard_sample"))
    rng = random.Random(seed)
    rng.shuffle(disagreement)
    rng.shuffle(fallback)
    return (disagreement + fallback)[:target_count]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "dataset_id",
        "window_id",
        "label",
        "selection_reason",
        "evidence_text",
        "model_caption",
        "candidate_texts",
        "gold_answer_index",
        "human_caption",
        "support_evidence",
        "counterfactual_validity",
        "annotator_notes",
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a high-value human annotation subset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--target-count", type=int, required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vote-rows", default=None)
    parser.add_argument("--gated-rows", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.input))
    vote_rows = list(read_jsonl(args.vote_rows)) if args.vote_rows else None
    gated_rows = list(read_jsonl(args.gated_rows)) if args.gated_rows else None
    rows = build_annotation_subset(
        records=records,
        dataset_id=args.dataset_id,
        target_count=args.target_count,
        seed=args.seed,
        vote_rows=vote_rows,
        gated_rows=gated_rows,
    )
    write_jsonl(args.output_jsonl, rows)
    write_csv(args.output_csv, rows)
    print(f"Wrote annotation subset: {len(rows)}")


if __name__ == "__main__":
    main()
