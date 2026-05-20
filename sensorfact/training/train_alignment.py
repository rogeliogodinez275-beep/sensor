from __future__ import annotations

import argparse
from pathlib import Path

from sensorfact.eval.eval_alignment import evaluate_alignment
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.models.alignment import LightweightSensorTextAligner


def train_alignment_model(
    train_path: str | Path,
    eval_path: str | Path,
    model_path: str | Path,
    metrics_path: str | Path,
    rows_path: str | Path | None = None,
    epochs: int = 200,
    learning_rate: float = 0.05,
    seed: int = 42,
    evidence_loss_weight: float = 0.5,
) -> dict:
    train_records = list(read_jsonl(train_path))
    eval_records = list(read_jsonl(eval_path))
    model = LightweightSensorTextAligner.fit(
        train_records,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        evidence_loss_weight=evidence_loss_weight,
    )
    model.save(model_path)
    metrics, rows = evaluate_alignment(model, eval_records)
    metrics.update(
        {
            "system": "lightweight_sensor_text_aligner",
            "train_path": str(train_path),
            "eval_path": str(eval_path),
            "model_path": str(model_path),
            "n_train_records": len(train_records),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "evidence_loss_weight": evidence_loss_weight,
        }
    )
    write_json(metrics_path, metrics)
    if rows_path is not None:
        write_jsonl(rows_path, rows)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the lightweight SensorFact alignment baseline.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--rows", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evidence-loss-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_alignment_model(
        train_path=args.train,
        eval_path=args.eval,
        model_path=args.model,
        metrics_path=args.metrics,
        rows_path=args.rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        evidence_loss_weight=args.evidence_loss_weight,
    )
    print("Lightweight alignment training complete.")
    print(f"caption_selection_accuracy: {metrics['caption_selection_accuracy']:.4f}")
    print(f"evidence_field_accuracy: {metrics['evidence_field_accuracy']:.4f}")


if __name__ == "__main__":
    main()
