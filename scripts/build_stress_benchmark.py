from __future__ import annotations

import argparse
import copy
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import read_jsonl, write_jsonl


NUMERIC_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def _mask_numbers(text: str) -> str:
    return NUMERIC_RE.sub("<num>", text)


def _deranged_indices(n_items: int, seed: int) -> list[int]:
    indices = list(range(n_items))
    if n_items <= 1:
        return indices
    rng = random.Random(seed)
    for _ in range(1000):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(idx != shuffled[idx] for idx in indices):
            return shuffled
    return indices[1:] + indices[:1]


def build_stress_rows(rows: list[dict], variant: str, seed: int = 42) -> list[dict]:
    stressed = [copy.deepcopy(row) for row in rows]
    if variant == "numeric_mask":
        for row in stressed:
            row["evidence_text"] = _mask_numbers(str(row.get("evidence_text", "")))
            row["stress_variant"] = variant
        return stressed

    if variant == "shuffled_evidence":
        source_indices = _deranged_indices(len(rows), seed)
        for row, source_index in zip(stressed, source_indices):
            row["evidence_text"] = str(rows[source_index].get("evidence_text", ""))
            row["stress_variant"] = variant
            row["stress_source_window_id"] = rows[source_index].get("window_id")
        return stressed

    if variant == "hidden_evidence":
        for row in stressed:
            row["evidence_text"] = (
                "Instance-specific sensor evidence is hidden for this control condition."
            )
            row["stress_variant"] = variant
        return stressed

    raise ValueError(f"Unknown stress variant: {variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stress-test SensorFact benchmark variants.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=["numeric_mask", "shuffled_evidence", "hidden_evidence"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(read_jsonl(args.input))
    count = write_jsonl(args.output, build_stress_rows(rows, args.variant, seed=args.seed))
    print(f"Wrote {count} {args.variant} rows to {args.output}")


if __name__ == "__main__":
    main()
