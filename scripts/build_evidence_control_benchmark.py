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
from sensorfact.hard_benchmark import hard_evidence_text
from sensorfact.qwen_llm_eval import record_evidence_text
from sensorfact.schemas import EvidenceCard
from sensorfact.axis_vocab import dataset_axis_options


NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
MODE_ALIASES = {
    "numeric_mask": "numeric-mask",
    "numeric-mask": "numeric-mask",
    "numeric_swap": "numeric-swap",
    "numeric-swap": "numeric-swap",
    "axis_permutation": "axis-permutation",
    "axis-permutation": "axis-permutation",
    "trend_flip": "trend-flip",
    "trend-flip": "trend-flip",
}


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


def _canonical_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def _rebuild_v3_evidence_text(evidence: dict) -> str:
    return hard_evidence_text(EvidenceCard.from_json_dict(evidence), variant="v3")


def _alternate_axis(evidence: dict) -> str:
    current = str(evidence.get("dominant_axis", "uncertain"))
    for option in dataset_axis_options(str(evidence.get("dataset_id", "unknown"))):
        if option != current:
            return option
    return "uncertain" if current != "uncertain" else "acc_x"


def _flip_trend_segments(values: list[str]) -> list[str]:
    mapping = {"rise": "fall", "fall": "rise", "stable": "stable"}
    return [mapping.get(str(item), str(item)) for item in values]


def build_control_records(records: list[dict], *, mode: str, seed: int = 42) -> list[dict]:
    mode = _canonical_mode(mode)
    rows = [copy.deepcopy(record) for record in records]
    if mode == "visible":
        for row in rows:
            row["evidence_text"] = record_evidence_text(row)
            row["evidence_control"] = "visible"
            row["evidence_source_window_id"] = row.get("window_id")
        return rows

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

    if mode == "numeric-swap":
        sources = _deranged_indices(len(rows), seed)
        for idx, source_idx in enumerate(sources):
            source = records[source_idx]
            row = rows[idx]
            row["evidence"] = copy.deepcopy(row.get("evidence"))
            row["evidence"]["numeric"] = copy.deepcopy(source.get("evidence", {}).get("numeric", {}))
            row["evidence_text"] = _rebuild_v3_evidence_text(row["evidence"])
            row["evidence_control"] = "numeric-swap"
            row["evidence_source_window_id"] = source.get("window_id")
        return rows

    if mode == "axis-permutation":
        for row in rows:
            row["evidence"] = copy.deepcopy(row.get("evidence"))
            row["evidence"]["dominant_axis"] = _alternate_axis(row["evidence"])
            row["evidence_text"] = _rebuild_v3_evidence_text(row["evidence"])
            row["evidence_control"] = "axis-permutation"
            row["evidence_source_window_id"] = row.get("window_id")
        return rows

    if mode == "trend-flip":
        for row in rows:
            row["evidence"] = copy.deepcopy(row.get("evidence"))
            row["evidence"]["trend_segments"] = _flip_trend_segments(
                list(row["evidence"].get("trend_segments", []))
            )
            row["evidence_text"] = _rebuild_v3_evidence_text(row["evidence"])
            row["evidence_control"] = "trend-flip"
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
    parser.add_argument(
        "--mode",
        choices=[
            "visible",
            "shuffled",
            "numeric-mask",
            "numeric_mask",
            "numeric-swap",
            "numeric_swap",
            "axis-permutation",
            "axis_permutation",
            "trend-flip",
            "trend_flip",
            "hidden",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", default=None, help="Optional JSON metadata output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(read_jsonl(args.input))
    mode = _canonical_mode(args.mode)
    rows = build_control_records(records, mode=mode, seed=args.seed)
    n_written = write_jsonl(args.output, rows)
    metadata = {
        "input": args.input,
        "output": args.output,
        "mode": mode,
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
