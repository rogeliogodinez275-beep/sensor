from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.supervised_baseline import run_supervised_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trainable supervised SensorFact grounding baseline.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--model-type", choices=["logistic_regression", "random_forest"], default="logistic_regression")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-eval-records", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-variant", default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["oracle_fields", "numeric_only"],
        default="oracle_fields",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_supervised_baseline(
        train_path=workspace / args.train_path,
        eval_path=workspace / args.eval_path,
        output_metrics_path=workspace / args.output_metrics,
        output_rows_path=workspace / args.output_rows,
        model_type=args.model_type,
        threshold=args.threshold,
        max_train_records=args.max_train_records,
        max_eval_records=args.max_eval_records,
        seed=args.seed,
        hard_variant=args.hard_variant,
        feature_mode=args.feature_mode,
    )
    print("Supervised baseline evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
