from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_jsonl


def _mapped_prediction(row: dict) -> int | None:
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


def build_vote_candidate_subset(base_records: list[dict], vote_runs: list[list[dict]]) -> list[dict]:
    vote_by_window = [{str(row["window_id"]): row for row in rows} for rows in vote_runs]
    output = []
    for record in base_records:
        window_id = str(record["window_id"])
        base_map = list(record.get("candidate_index_map", range(len(record["caption_selection"]["candidates"]))))
        candidates_by_original = {
            int(original_idx): candidate
            for original_idx, candidate in zip(base_map, record["caption_selection"]["candidates"])
        }
        votes: list[int] = []
        for run in vote_by_window:
            row = run.get(window_id)
            if row is None:
                continue
            prediction = _mapped_prediction(row)
            if prediction is not None and prediction in candidates_by_original:
                votes.append(prediction)

        selected = list(dict.fromkeys(votes))
        if not selected:
            selected = [int(idx) for idx in base_map]
        answer_index = int(record["caption_selection"]["answer_index"])
        answer_original = int(base_map[answer_index]) if 0 <= answer_index < len(base_map) else -1
        remapped_answer = selected.index(answer_original) if answer_original in selected else -1

        subset_record = json.loads(json.dumps(record))
        subset_record["caption_selection"]["candidates"] = [candidates_by_original[idx] for idx in selected]
        subset_record["caption_selection"]["answer_index"] = remapped_answer
        subset_record["candidate_index_map"] = selected
        subset_record["vote_candidate_subset"] = {
            "vote_count": len(votes),
            "unique_vote_count": len(selected),
            "gold_in_vote_candidates": answer_original in selected,
            "source": "caption_order_vote5",
        }
        output.append(subset_record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build candidate subset from order-vote predictions.")
    parser.add_argument("--base-benchmark", required=True)
    parser.add_argument("--vote-rows", nargs="+", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_records = list(read_jsonl(args.base_benchmark))
    vote_runs = [list(read_jsonl(path)) for path in args.vote_rows]
    output = build_vote_candidate_subset(base_records, vote_runs)
    count = write_jsonl(args.output_path, output)
    kept = sum(1 for row in output if row["vote_candidate_subset"]["gold_in_vote_candidates"])
    print(f"Wrote {count} vote-candidate records to {args.output_path}; gold_kept={kept}/{count}")


if __name__ == "__main__":
    main()
