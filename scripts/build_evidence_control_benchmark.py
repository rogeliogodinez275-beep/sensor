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

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.qwen_llm_eval import record_evidence_text


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def mask_numeric_evidence_text(text: str) -> str:
    return NUMBER_RE.sub("<num>", text)


def _deranged_indices(n_records: int, seed: int) -> list[int]:
    if n_records < 2:
        raise ValueError("shuffled control requires at least two records")
    rng = random.Random(seed)
    indices = list(range(n_records))
    for _ in range(100):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(src != idx for idx, src in enumerate(shuffled)):
            return shuffled
    return indices[1:] + indices[:1]


def build_control_records(records: list[dict], *, mode: str, seed: int = 42) -> list[dict]:
    rows = [copy.deepcopy(record) for record in records]
    if mode == "shuffled":
        sources = _deranged_indices(len(rows), seed)
        for idx, source_idx in enumerate(sources):
            source = records[source_idx]
            rows[idx]["evidence"] = copy.deepcopy(source.get("evidence"))
            rows[idx]["evidence_text"] = record_evidence_text(source)
            rows[idx]["evidence_control"] = "shuffled"
            rows[idx]["evidence_source_window_id"] = source.get("window_id")
        return rows

    if mode == "numeric-mask":
        for row in rows:
            row["evidence_text"] = mask_numeric_evidence_text(record_evidence_text(row))
            row["evidence_control"] = "numeric-mask"
            row["evidence_source_window_id"] = row.get("window_id")
        return rows

    if mode == "hidden":
        for row in rows:
            row["evidence_text"] = "Evidence is hidden for this control condition."
            row["evidence_control"] = "hidden"
            row["evidence_source_window_id"] = None
        return rows

    raise ValueError(f"unknown evidence control mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evidence-control hard benchmarks.")
    parser.add_argument("--input", required=True, help="Input benchmark JSONL.")
    parser.add_argument("--output", required=True, help="Output control benchmark JSONL.")
    parser.add_argument("--mode", choices=["shuffled", "numeric-mask", "hidden"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.input))
    rows = build_control_records(records, mode=args.mode, seed=args.seed)
    n_written = write_jsonl(args.output, rows)
    metadata = {
        "input": args.input,
        "output": args.output,
        "mode": args.mode,
        "seed": args.seed,
        "n_records": n_written,
    }
    if args.metadata:
        write_json(args.metadata, metadata)
    print("Evidence control benchmark written.")
    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

