from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.models.raw_sensor_alignment import train_raw_sensor_field_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a raw-sensor-only SensorFact field alignment baseline.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--train-records", required=True)
    parser.add_argument("--eval-windows", required=True)
    parser.add_argument("--eval-records", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-eval-records", type=int, default=-1)
    parser.add_argument("--max-calibration-records", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics, _ = train_raw_sensor_field_model(
        train_windows_path=workspace / args.train_windows,
        train_records_path=workspace / args.train_records,
        eval_windows_path=workspace / args.eval_windows,
        eval_records_path=workspace / args.eval_records,
        metrics_path=workspace / args.metrics,
        rows_path=workspace / args.rows,
        model_path=workspace / args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_train_records=args.max_train_records,
        max_eval_records=args.max_eval_records,
        max_calibration_records=args.max_calibration_records,
    )
    print("Raw-sensor baseline finished.")
    for key in [
        "input_source",
        "device",
        "caption_selection_accuracy",
        "caption_macro_f1",
        "cf_reject_f1",
        "support_ece",
        "evidence_field_accuracy",
        "n_train_records",
        "n_eval_records",
    ]:
        print(f"{key}: {metrics.get(key)}")


if __name__ == "__main__":
    main()
