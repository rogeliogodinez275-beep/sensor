from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.calibrate_gate_threshold import DEFAULT_THRESHOLDS, sweep_thresholds
from scripts.gate_caption_rows import _score_margin, _map_prediction
from sensorfact.io import read_jsonl, write_json


DEFAULT_BUCKETS = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]


def margin_buckets(alternate_rows: list[dict], *, buckets: list[float] | None = None) -> list[dict]:
    buckets = buckets or DEFAULT_BUCKETS
    rows = []
    for lower, upper in zip(buckets, buckets[1:]):
        selected = [row for row in alternate_rows if lower < _score_margin(row) <= upper]
        rows.append(_summarize_bucket(selected, label=f"({lower:g}, {upper:g}]", lower=lower, upper=upper))
    selected = [row for row in alternate_rows if _score_margin(row) > buckets[-1]]
    rows.append(_summarize_bucket(selected, label=f"({buckets[-1]:g}, inf)", lower=buckets[-1], upper=None))
    selected = [row for row in alternate_rows if _score_margin(row) <= buckets[0]]
    rows.insert(0, _summarize_bucket(selected, label=f"(-inf, {buckets[0]:g}]", lower=None, upper=buckets[0]))
    return rows


def _summarize_bucket(rows: list[dict], *, label: str, lower: float | None, upper: float | None) -> dict:
    correct = 0
    valid = 0
    for row in rows:
        prediction, bad_map = _map_prediction(row, strict_map=True)
        if prediction is None or bad_map:
            continue
        valid += 1
        correct += int(prediction == int(row["caption_answer_index"]))
    return {
        "bucket": label,
        "lower": lower,
        "upper": upper,
        "count": len(rows),
        "valid_count": valid,
        "alternate_accuracy": correct / max(1, valid),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze caption gate margin coverage/accuracy curves.")
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--alternate-rows", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-threshold-csv", default=None)
    parser.add_argument("--output-bucket-csv", default=None)
    parser.add_argument("--thresholds", default=None)
    return parser.parse_args()


def _parse_thresholds(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_THRESHOLDS
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    primary_rows = list(read_jsonl(args.primary_rows))
    alternate_rows = list(read_jsonl(args.alternate_rows))
    threshold_rows = sweep_thresholds(
        primary_rows,
        alternate_rows,
        thresholds=_parse_thresholds(args.thresholds),
    )
    bucket_rows = margin_buckets(alternate_rows)
    payload = {
        "primary_rows": args.primary_rows,
        "alternate_rows": args.alternate_rows,
        "threshold_curve": threshold_rows,
        "margin_buckets": bucket_rows,
    }
    write_json(args.output_json, payload)
    if args.output_threshold_csv:
        write_csv(args.output_threshold_csv, threshold_rows)
    if args.output_bucket_csv:
        write_csv(args.output_bucket_csv, bucket_rows)
    print("Margin curve analysis finished.")
    print(f"threshold_points: {len(threshold_rows)}")
    print(f"bucket_count: {len(bucket_rows)}")


if __name__ == "__main__":
    main()

