from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.hybrid_verifier_eval import evaluate_hybrid_grounding
from sensorfact.io import read_jsonl


def sweep_thresholds(
    structured_rows: list[dict],
    direct_rows: list[dict],
    *,
    thresholds: list[float],
) -> tuple[list[dict], dict]:
    summaries: list[dict] = []
    best: dict | None = None
    for threshold in thresholds:
        metrics, _ = evaluate_hybrid_grounding(
            structured_rows,
            direct_rows,
            structured_threshold=threshold,
            system_name="hybrid_threshold_sweep",
        )
        row = {
            "threshold": float(threshold),
            "caption_selection_accuracy": float(metrics["caption_selection_accuracy"]),
            "cf_reject_f1": float(metrics["cf_reject_f1"]),
            "caption_fallback_rate": float(metrics["caption_fallback_rate"]),
            "structured_decisive_rate": float(metrics["structured_decisive_rate"]),
        }
        summaries.append(row)
        if best is None:
            best = row
            continue
        candidate_key = (
            row["caption_selection_accuracy"],
            row["cf_reject_f1"],
            -abs(row["threshold"] - 0.5),
        )
        best_key = (
            best["caption_selection_accuracy"],
            best["cf_reject_f1"],
            -abs(best["threshold"] - 0.5),
        )
        if candidate_key > best_key:
            best = row
    if best is None:
        raise ValueError("threshold sweep requires at least one threshold")
    return summaries, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep hybrid structured thresholds.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--structured-rows", required=True)
    parser.add_argument("--direct-rows", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    structured_rows = list(read_jsonl(workspace / args.structured_rows))
    direct_rows = list(read_jsonl(workspace / args.direct_rows))
    step = float(args.step)
    if step <= 0 or step > 1:
      raise ValueError("--step must be in (0, 1].")
    count = int(round(1.0 / step))
    thresholds = [round(i * step, 10) for i in range(count + 1)]
    summaries, best = sweep_thresholds(structured_rows, direct_rows, thresholds=thresholds)
    result = {"thresholds": summaries, "best": best}
    if args.output_json:
        output_path = workspace / args.output_json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
