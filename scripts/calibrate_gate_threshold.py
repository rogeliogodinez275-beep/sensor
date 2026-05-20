from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gate_caption_rows import gate_caption_rows, evaluate_gated_rows
from sensorfact.io import read_jsonl, write_json


DEFAULT_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]


def _primary_accuracy(primary_rows: list[dict]) -> float:
    metrics = evaluate_gated_rows(primary_rows, system_name="primary")
    return float(metrics["caption_selection_accuracy"])


def sweep_thresholds(
    primary_rows: list[dict],
    alternate_rows: list[dict],
    *,
    thresholds: list[float],
) -> list[dict]:
    primary_acc = _primary_accuracy(primary_rows)
    swept = []
    n_records = len(primary_rows)
    for threshold in thresholds:
        gated_rows = gate_caption_rows(
            primary_rows,
            alternate_rows,
            min_alternate_margin=float(threshold),
        )
        metrics = evaluate_gated_rows(
            gated_rows,
            system_name=f"gated_margin_{threshold:g}",
        )
        alternate_count = int(metrics["caption_gate_alternate_count"])
        acc = float(metrics["caption_selection_accuracy"])
        swept.append(
            {
                "threshold": float(threshold),
                "caption_selection_accuracy": acc,
                "delta_vs_primary": acc - primary_acc,
                "alternate_count": alternate_count,
                "coverage": alternate_count / max(1, n_records),
                "n_eval_records": n_records,
            }
        )
    return swept


def choose_threshold(swept_rows: list[dict]) -> dict:
    if not swept_rows:
        raise ValueError("no threshold rows to choose from")
    return max(
        swept_rows,
        key=lambda row: (
            float(row["caption_selection_accuracy"]),
            -int(row["alternate_count"]),
            -float(row["threshold"]),
        ),
    )


def split_rows_by_window(
    primary_rows: list[dict],
    alternate_rows: list[dict],
    *,
    dev_modulus: int,
    dev_remainders: set[int],
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    if dev_modulus <= 1:
        raise ValueError("dev_modulus must be greater than 1")
    alternate_by_window = {str(row["window_id"]): row for row in alternate_rows}
    dev_primary: list[dict] = []
    dev_alternate: list[dict] = []
    eval_primary: list[dict] = []
    eval_alternate: list[dict] = []
    for primary in primary_rows:
        window_id = str(primary["window_id"])
        alternate = alternate_by_window.get(window_id)
        if alternate is None:
            continue
        digest = hashlib.sha256(window_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:12], 16) % dev_modulus
        if bucket in dev_remainders:
            dev_primary.append(primary)
            dev_alternate.append(alternate)
        else:
            eval_primary.append(primary)
            eval_alternate.append(alternate)
    if not dev_primary:
        raise ValueError("dev split is empty")
    if not eval_primary:
        raise ValueError("eval split is empty")
    return dev_primary, dev_alternate, eval_primary, eval_alternate


def evaluate_threshold(
    primary_rows: list[dict],
    alternate_rows: list[dict],
    *,
    threshold: float,
    system_name: str,
) -> dict:
    gated_rows = gate_caption_rows(
        primary_rows,
        alternate_rows,
        min_alternate_margin=threshold,
    )
    metrics = evaluate_gated_rows(gated_rows, system_name=system_name)
    metrics["threshold"] = threshold
    metrics["coverage"] = metrics["caption_gate_alternate_count"] / max(1, len(primary_rows))
    return metrics


def _parse_thresholds(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_THRESHOLDS
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate caption gate threshold on a dev split.")
    parser.add_argument("--primary-rows", required=True)
    parser.add_argument("--alternate-rows", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--thresholds", default=None, help="Comma-separated margin thresholds.")
    parser.add_argument("--system-name", default="dev_calibrated_caption_gate")
    parser.add_argument(
        "--dev-modulus",
        type=int,
        default=None,
        help="If set, choose threshold on hash buckets and evaluate on the remaining rows.",
    )
    parser.add_argument(
        "--dev-remainders",
        default="0",
        help="Comma-separated hash remainders used as dev buckets when --dev-modulus is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primary_rows = list(read_jsonl(args.primary_rows))
    alternate_rows = list(read_jsonl(args.alternate_rows))
    eval_metrics = None
    split = None
    if args.dev_modulus:
        dev_remainders = {int(item.strip()) for item in args.dev_remainders.split(",") if item.strip()}
        dev_primary, dev_alternate, eval_primary, eval_alternate = split_rows_by_window(
            primary_rows,
            alternate_rows,
            dev_modulus=args.dev_modulus,
            dev_remainders=dev_remainders,
        )
        split = {
            "dev_modulus": args.dev_modulus,
            "dev_remainders": sorted(dev_remainders),
            "n_dev_records": len(dev_primary),
            "n_eval_records": len(eval_primary),
        }
        primary_rows = dev_primary
        alternate_rows = dev_alternate
    else:
        eval_primary = []
        eval_alternate = []
    swept = sweep_thresholds(
        primary_rows,
        alternate_rows,
        thresholds=_parse_thresholds(args.thresholds),
    )
    best = choose_threshold(swept)
    if split:
        eval_metrics = evaluate_threshold(
            eval_primary,
            eval_alternate,
            threshold=float(best["threshold"]),
            system_name=f"{args.system_name}_heldout_eval",
        )
    payload = {
        "system": args.system_name,
        "primary_rows": args.primary_rows,
        "alternate_rows": args.alternate_rows,
        "selection_rule": "maximize dev accuracy; break ties by lower alternate coverage, then higher threshold",
        "best_threshold": best["threshold"],
        "best": best,
        "sweep": swept,
        "split": split,
        "heldout_eval": eval_metrics,
    }
    write_json(args.output_json, payload)
    print("Gate threshold calibration finished.")
    print(f"best_threshold: {best['threshold']}")
    print(f"dev_accuracy: {best['caption_selection_accuracy']}")
    print(f"coverage: {best['coverage']}")


if __name__ == "__main__":
    main()
