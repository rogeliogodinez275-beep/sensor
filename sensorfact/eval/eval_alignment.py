from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import brier_score, caption_selection_accuracy, expected_calibration_error, risk_coverage_auc
from sensorfact.models.alignment import FIELD_VOCABS, LightweightSensorTextAligner
from sensorfact.supervised_baseline import extract_statement_fields


def _candidate_compatibility_scores(
    model: LightweightSensorTextAligner,
    field_probs: dict[str, np.ndarray],
    row_idx: int,
    candidates: list[dict],
) -> list[float]:
    scores: list[float] = []
    for candidate in candidates:
        claims = extract_statement_fields(str(candidate.get("text", "")))
        score = 0.0
        claimed = 0
        for field, value in claims.items():
            vocab = model.field_vocabs.get(field)
            probs = field_probs.get(field)
            if vocab is None or probs is None or value not in vocab:
                continue
            score += float(probs[row_idx, vocab.index(value)])
            claimed += 1
        scores.append(score / max(1, claimed))
    return scores


def evaluate_alignment(
    model: LightweightSensorTextAligner,
    records: list[dict],
) -> tuple[dict, list[dict]]:
    if not records:
        return {"caption_selection_accuracy": 0.0, "evidence_field_accuracy": 0.0, "n_eval_records": 0}, []
    caption_probs = model.predict_caption_proba(records)
    field_probs = model.predict_field_proba(records)
    rows: list[dict] = []
    selection_examples: list[dict] = []
    field_correct = 0
    field_total = 0
    for idx, record in enumerate(records):
        candidates = record["caption_selection"]["candidates"]
        answer_index = int(record["caption_selection"]["answer_index"])
        candidate_scores = _candidate_compatibility_scores(model, field_probs, idx, candidates)
        caption_pred = int(np.argmax(candidate_scores)) if candidate_scores else 0
        selection_examples.append({"answer_index": answer_index, "scores": candidate_scores})
        evidence_predictions = {}
        evidence_probabilities = {}
        for field, probs in field_probs.items():
            pred_idx = int(np.argmax(probs[idx]))
            pred_value = model.field_vocabs[field][pred_idx]
            truth = str(record["evidence"].get(field))
            evidence_predictions[field] = pred_value
            evidence_probabilities[field] = [float(x) for x in probs[idx]]
            field_correct += int(pred_value == truth)
            field_total += 1
        rows.append(
            {
                "window_id": record.get("window_id"),
                "caption_prediction": caption_pred,
                "caption_answer_index": answer_index,
                "caption_scores": candidate_scores,
                "caption_label_prediction": model.caption_vocab[caption_pred],
                "evidence_predictions": evidence_predictions,
                "evidence_probabilities": evidence_probabilities,
            }
        )
    y_true = [1 if row["caption_prediction"] == row["caption_answer_index"] else 0 for row in rows]
    y_score = [
        float(max(row["caption_scores"]) if row["caption_scores"] else 0.0)
        for row in rows
    ]
    metrics = {
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "evidence_field_accuracy": float(field_correct / max(1, field_total)),
        "caption_brier": brier_score(y_true, y_score),
        "caption_ece": expected_calibration_error(y_true, y_score),
        "caption_risk_coverage_auc": risk_coverage_auc(y_true, y_score),
        "n_eval_records": len(records),
        "field_names": list(FIELD_VOCABS),
    }
    return metrics, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight SensorFact alignment model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--rows", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = LightweightSensorTextAligner.load(args.model)
    records = list(read_jsonl(args.eval))
    metrics, rows = evaluate_alignment(model, records)
    write_json(args.metrics, metrics)
    if args.rows:
        write_jsonl(args.rows, rows)
    print("Lightweight alignment evaluation complete.")
    print(f"caption_selection_accuracy: {metrics['caption_selection_accuracy']:.4f}")
    print(f"evidence_field_accuracy: {metrics['evidence_field_accuracy']:.4f}")


if __name__ == "__main__":
    main()
