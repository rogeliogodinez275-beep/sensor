from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from sensorfact.axis_vocab import AXIS_TEXT_TO_CODE, AXIS_VALUES
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy, counterfactual_rejection_metrics


FIELDS = ["intensity", "periodicity", "dominant_axis", "dominant_frequency", "burstiness"]
INTENSITY = ["low", "medium", "high"]
PERIODICITY = ["none", "weak", "strong"]
AXES = AXIS_VALUES
FREQUENCY = ["low", "mid", "high", "uncertain"]
BURSTINESS = ["smooth", "bursty"]
TRENDS = ["rise", "fall", "stable"]


PHRASE_MAP: dict[str, list[tuple[str, str]]] = {
    "intensity": [
        ("small-amplitude movement", "low"),
        ("subtle energy", "low"),
        ("contains low motion", "low"),
        ("motion intensity is low", "low"),
        ("moderate-amplitude movement", "medium"),
        ("moderate energy", "medium"),
        ("contains medium motion", "medium"),
        ("motion intensity is medium", "medium"),
        ("large-amplitude movement", "high"),
        ("forceful energy", "high"),
        ("contains high motion", "high"),
        ("motion intensity is high", "high"),
    ],
    "periodicity": [
        ("little repeatable rhythm", "none"),
        ("none periodicity", "none"),
        ("an irregular but noticeable rhythm", "weak"),
        ("a loose rhythm", "weak"),
        ("loose rhythm", "weak"),
        ("weak periodicity", "weak"),
        ("strong rhythm", "strong"),
        ("a repeatable rhythm", "strong"),
        ("repeatable rhythm", "strong"),
        ("clearly repeated rhythm", "strong"),
        ("strong periodicity", "strong"),
    ],
    "dominant_axis": [
        *[
            (phrase, code)
            for phrase, code in sorted(
                AXIS_TEXT_TO_CODE.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        ],
        ("strongest activity on uncertain", "uncertain"),
    ],
    "dominant_frequency": [
        ("a slow cadence", "low"),
        ("slow cadence", "low"),
        ("a mid-paced cadence", "mid"),
        ("mid-paced cadence", "mid"),
        ("middle-rate cadence", "mid"),
        ("a quick cadence", "high"),
        ("quick cadence", "high"),
        ("fast cadence", "high"),
        ("no stable cadence", "uncertain"),
    ],
    "burstiness": [
        ("changes remain fairly even rather than spiky", "smooth"),
        ("an even profile", "smooth"),
        ("even profile", "smooth"),
        ("smooth movement pattern", "smooth"),
        ("movement is smooth", "smooth"),
        ("short spikes", "bursty"),
        ("spiky bursts", "bursty"),
        ("movement is bursty", "bursty"),
    ],
}


def _one_hot(value: str | None, vocab: list[str]) -> np.ndarray:
    out = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        out[vocab.index(value)] = 1.0
    return out


def extract_statement_fields(text: str) -> dict[str, str]:
    lower = text.lower()
    fields: dict[str, str] = {}
    for field, phrases in PHRASE_MAP.items():
        for phrase, value in phrases:
            if phrase in lower:
                fields[field] = value
                break

    axis_match = re.search(r"dominant movement axis is\s+([a-z0-9_]+)", lower)
    if axis_match and axis_match.group(1) in AXES:
        fields["dominant_axis"] = axis_match.group(1)
    strongest_match = re.search(r"strongest activity on\s+([a-z0-9_]+)", lower)
    if strongest_match and strongest_match.group(1) in AXES:
        fields["dominant_axis"] = strongest_match.group(1)
    freq_match = re.search(r"dominant frequency is\s+([a-z0-9_]+)", lower)
    if freq_match and freq_match.group(1) in FREQUENCY:
        fields["dominant_frequency"] = freq_match.group(1)
    return fields


def _card_vector(card: dict) -> np.ndarray:
    numeric = card.get("numeric", {})
    base = np.asarray(
        [
            float(numeric.get("rms_energy", 0.0)),
            float(numeric.get("autocorr_peak", 0.0)),
            float(numeric.get("fft_dominant_ratio", 0.0)),
            float(numeric.get("dominant_frequency_hz", 0.0)),
            float(numeric.get("peak_count", 0.0)),
            float(card.get("confidence", 1.0)),
        ],
        dtype=np.float32,
    )
    trend_counts = np.asarray(
        [sum(1 for item in card.get("trend_segments", []) if item == trend) for trend in TRENDS],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            base,
            _one_hot(str(card.get("intensity")), INTENSITY),
            _one_hot(str(card.get("periodicity")), PERIODICITY),
            _one_hot(str(card.get("dominant_axis")), AXES),
            _one_hot(str(card.get("dominant_frequency")), FREQUENCY),
            _one_hot(str(card.get("burstiness")), BURSTINESS),
            trend_counts,
        ]
    )


def statement_feature_vector(record: dict, text: str, feature_mode: str = "oracle_fields") -> np.ndarray:
    if feature_mode not in {"oracle_fields", "numeric_only"}:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    card = record["evidence"]
    claims = extract_statement_fields(text)
    missing = np.asarray([1.0 if field not in claims else 0.0 for field in FIELDS], dtype=np.float32)
    matches = np.asarray(
        [
            1.0 if field in claims and str(card.get(field)) == claims[field] else 0.0
            for field in FIELDS
        ],
        dtype=np.float32,
    )
    mismatches = np.asarray(
        [
            1.0 if field in claims and str(card.get(field)) != claims[field] else 0.0
            for field in FIELDS
        ],
        dtype=np.float32,
    )
    if feature_mode == "numeric_only":
        numeric = card.get("numeric", {})
        evidence_features = np.asarray(
            [
                float(numeric.get("rms_energy", 0.0)),
                float(numeric.get("autocorr_peak", 0.0)),
                float(numeric.get("fft_dominant_ratio", 0.0)),
                float(numeric.get("dominant_frequency_hz", 0.0)),
                float(numeric.get("peak_count", 0.0)),
                float(card.get("confidence", 1.0)),
            ],
            dtype=np.float32,
        )
        matches = np.zeros_like(matches)
        mismatches = np.zeros_like(mismatches)
    else:
        evidence_features = _card_vector(card)
    return np.concatenate(
        [
            evidence_features,
            _one_hot(claims.get("intensity"), INTENSITY),
            _one_hot(claims.get("periodicity"), PERIODICITY),
            _one_hot(claims.get("dominant_axis"), AXES),
            _one_hot(claims.get("dominant_frequency"), FREQUENCY),
            _one_hot(claims.get("burstiness"), BURSTINESS),
            missing,
            matches,
            mismatches,
        ]
    )


def iter_statement_examples(records: Iterable[dict]) -> Iterable[tuple[dict, str, int]]:
    for record in records:
        yield record, record["positive"]["text"], 1
        for item in record.get("paraphrases", []):
            yield record, item["text"], 1
        for item in record["counterfactuals"]:
            yield record, item["text"], 0


@dataclass
class FieldMatchFallbackClassifier:
    threshold: float = 0.5

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "FieldMatchFallbackClassifier":
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        matches = values[:, -10:-5]
        mismatches = values[:, -5:]
        claimed = matches + mismatches
        denominator = np.maximum(np.sum(claimed, axis=1), 1.0)
        any_mismatch = np.sum(mismatches, axis=1) > 0.0
        match_fraction = np.sum(matches, axis=1) / denominator
        positive = np.where(any_mismatch, np.minimum(match_fraction, 0.1), match_fraction)
        return np.stack([1.0 - positive, positive], axis=1)


def _make_classifier(model_type: str, seed: int):
    try:
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
    except ImportError:
        return FieldMatchFallbackClassifier()


def _training_matrix(records: list[dict], feature_mode: str) -> tuple[np.ndarray, np.ndarray]:
    rows = list(iter_statement_examples(records))
    x = np.stack([statement_feature_vector(record, text, feature_mode=feature_mode) for record, text, _ in rows])
    y = np.asarray([label for _, _, label in rows], dtype=np.int64)
    return x, y


def evaluate_supervised_grounding(
    train_records: list[dict],
    eval_records: list[dict],
    model_type: str = "logistic_regression",
    threshold: float = 0.5,
    seed: int = 42,
    hard_variant: str | None = None,
    feature_mode: str = "oracle_fields",
) -> tuple[dict, list[dict]]:
    x_train, y_train = _training_matrix(train_records, feature_mode=feature_mode)
    classifier = _make_classifier(model_type, seed=seed)
    classifier.fit(x_train, y_train)

    selection_examples = []
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []

    for record in eval_records:
        candidates = record["caption_selection"]["candidates"]
        caption_scores = [
            float(
                classifier.predict_proba(
                    statement_feature_vector(record, item["text"], feature_mode=feature_mode).reshape(1, -1)
                )[0, 1]
            )
            for item in candidates
        ]
        caption_prediction = int(np.argmax(caption_scores)) if caption_scores else None
        selection_examples.append(
            {
                "answer_index": record["caption_selection"]["answer_index"],
                "scores": caption_scores,
            }
        )

        support_items = [
            {
                "kind": "positive",
                "text": record["positive"]["text"],
                "supported": True,
                "changed_fact": None,
                "changed_facts": [],
            }
        ]
        for item in record["counterfactuals"]:
            support_items.append(
                {
                    "kind": "counterfactual",
                    "text": item["text"],
                    "supported": False,
                    "changed_fact": item.get("changed_fact"),
                    "changed_facts": item.get("changed_facts", []),
                }
            )
        support_probabilities = [
            float(
                classifier.predict_proba(
                    statement_feature_vector(record, item["text"], feature_mode=feature_mode).reshape(1, -1)
                )[0, 1]
            )
            for item in support_items
        ]
        y_true.extend([1 if item["supported"] else 0 for item in support_items])
        y_score.extend(support_probabilities)
        rows.append(
            {
                "window_id": record["window_id"],
                "caption_prediction": caption_prediction,
                "caption_answer_index": record["caption_selection"]["answer_index"],
                "caption_scores": caption_scores,
                "support_probabilities": support_probabilities,
                "support_items": support_items,
            }
        )

    cf = counterfactual_rejection_metrics(y_true, y_score, threshold=threshold)
    metrics = {
        "system": "trainable_supervised_field_verifier",
        "model_type": model_type if classifier.__class__.__name__ != "FieldMatchFallbackClassifier" else "field_match_fallback",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": cf["accuracy"],
        "cf_reject_precision": cf["precision"],
        "cf_reject_recall": cf["recall"],
        "cf_reject_f1": cf["f1"],
        "n_train_records": len(train_records),
        "n_eval_records": len(eval_records),
        "support_score_threshold": threshold,
        "hard_variant": hard_variant,
        "feature_mode": feature_mode,
    }
    return metrics, rows


def run_supervised_baseline(
    train_path: str | Path,
    eval_path: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path | None = None,
    model_type: str = "logistic_regression",
    threshold: float = 0.5,
    max_train_records: int | None = None,
    max_eval_records: int | None = None,
    seed: int = 42,
    hard_variant: str | None = None,
    feature_mode: str = "oracle_fields",
) -> dict:
    train_records = list(read_jsonl(train_path))
    eval_records = list(read_jsonl(eval_path))
    if max_train_records is not None and max_train_records >= 0:
        train_records = train_records[:max_train_records]
    if max_eval_records is not None and max_eval_records >= 0:
        eval_records = eval_records[:max_eval_records]
    metrics, rows = evaluate_supervised_grounding(
        train_records=train_records,
        eval_records=eval_records,
        model_type=model_type,
        threshold=threshold,
        seed=seed,
        hard_variant=hard_variant,
        feature_mode=feature_mode,
    )
    metrics["train_path"] = str(train_path)
    metrics["eval_path"] = str(eval_path)
    metrics["max_train_records"] = max_train_records
    metrics["max_eval_records"] = max_eval_records
    write_json(output_metrics_path, metrics)
    if output_rows_path is not None:
        write_jsonl(output_rows_path, rows)
    return metrics
