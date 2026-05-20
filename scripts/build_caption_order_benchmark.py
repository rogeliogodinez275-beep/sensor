from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_jsonl


def _candidate_key(candidate: dict) -> str:
    return json.dumps(candidate, sort_keys=True, ensure_ascii=False)


def _shuffled_order(n_items: int, seed_text: str) -> list[int]:
    indices = list(range(n_items))
    if n_items <= 1:
        return indices
    rng = random.Random(seed_text)
    for _ in range(1000):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if shuffled != indices:
            return shuffled
    return indices[1:] + indices[:1]


def build_caption_order_rows(rows: list[dict], seed: int = 5151) -> list[dict]:
    shuffled_rows: list[dict] = []
    for row in rows:
        new_row = copy.deepcopy(row)
        selection = new_row["caption_selection"]
        candidates = list(selection["candidates"])
        original_answer_index = int(selection["answer_index"])
        order = _shuffled_order(len(candidates), f"{seed}:{row.get('window_id', '')}:caption")
        shuffled_candidates = [candidates[index] for index in order]
        selection["candidates"] = shuffled_candidates
        selection["answer_index"] = order.index(original_answer_index)
        if "candidate_index_map" in new_row:
            index_map = list(new_row["candidate_index_map"])
            new_row["candidate_index_map"] = [index_map[index] for index in order]
        new_row["caption_order_seed"] = seed
        new_row["caption_order_original_answer_index"] = original_answer_index
        new_row["caption_order_original_keys"] = [_candidate_key(candidate) for candidate in candidates]
        shuffled_rows.append(new_row)
    return shuffled_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shuffle caption candidate order while preserving labels.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=5151)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    count = write_jsonl(args.output, build_caption_order_rows(rows, seed=args.seed))
    print(f"Wrote {count} caption-order rows to {args.output}")


if __name__ == "__main__":
    main()
