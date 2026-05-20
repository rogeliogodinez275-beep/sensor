from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from sensorfact.benchmark import evidence_to_text
from sensorfact.features import apply_normalization, normalize_matrix, sensor_features
from sensorfact.metrics import accuracy, caption_selection_accuracy, counterfactual_rejection_metrics, macro_f1
from sensorfact.schemas import EvidenceCard, SensorWindow


@dataclass
class CentroidClassifier:
    labels: list[str]
    centroids: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, windows: list[SensorWindow]) -> "CentroidClassifier":
        features = np.stack([sensor_features(row) for row in windows])
        features, mean, std = normalize_matrix(features)
        labels = sorted({str(row.label) for row in windows})
        centroids = []
        for label in labels:
            subset = features[[str(row.label) == label for row in windows]]
            centroids.append(np.mean(subset, axis=0))
        return cls(labels=labels, centroids=np.stack(centroids), mean=mean, std=std)

    def predict(self, windows: list[SensorWindow]) -> list[str]:
        if not windows:
            return []
        features = np.stack([sensor_features(row) for row in windows])
        features = apply_normalization(features, self.mean, self.std)
        distances = np.linalg.norm(features[:, None, :] - self.centroids[None, :, :], axis=2)
        idx = np.argmin(distances, axis=1)
        return [self.labels[int(i)] for i in idx]

    def to_json_dict(self) -> dict:
        return {
            "labels": self.labels,
            "centroids": self.centroids.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


def evaluate_har(train: list[SensorWindow], test: list[SensorWindow]) -> tuple[dict, CentroidClassifier]:
    classifier = CentroidClassifier.fit(train)
    pred = classifier.predict(test)
    truth = [str(row.label) for row in test]
    return (
        {
            "har_accuracy": accuracy(truth, pred),
            "har_macro_f1": macro_f1(truth, pred),
            "n_train": len(train),
            "n_test": len(test),
        },
        classifier,
    )


def _field_score(card: EvidenceCard, text: str, fields: list[str], contradiction_penalty: float) -> float:
    lower = text.lower()
    checks: list[tuple[str, str]] = [
        ("intensity", card.intensity),
        ("periodicity", card.periodicity),
        ("dominant_axis", card.dominant_axis),
        ("dominant_frequency", card.dominant_frequency),
        ("burstiness", card.burstiness),
    ]
    hits = 0.0
    total = 0.0
    contradictions = 0
    for field, value in checks:
        if field not in fields:
            continue
        total += 1.0
        if str(value).lower() in lower:
            hits += 1.0
        elif field == "dominant_axis" and any(axis in lower for axis in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]):
            contradictions += 1
            hits -= contradiction_penalty
        elif field == "intensity" and any(word in lower for word in ["low", "medium", "high"]):
            contradictions += 1
            hits -= contradiction_penalty
        elif field == "periodicity" and any(word in lower for word in ["none", "weak", "strong", "non"]):
            contradictions += 1
            hits -= contradiction_penalty
        elif field == "dominant_frequency" and any(word in lower for word in ["low", "mid", "high"]):
            contradictions += 1
            hits -= contradiction_penalty
        elif field == "burstiness" and any(word in lower for word in ["smooth", "bursty"]):
            contradictions += 1
            hits -= contradiction_penalty
    if "trend" in fields:
        total += 1.0
        trend_hits = sum(1 for item in card.trend_segments if item.lower() in lower)
        hits += min(1.0, trend_hits / max(1, len(card.trend_segments)))
    if total == 0:
        return 0.0
    score = float(max(0.0, min(1.0, hits / total)))
    if contradiction_penalty >= 0.75 and contradictions:
        return min(score, 0.35)
    return score


def make_support_scorer(system_id: str) -> Callable[[EvidenceCard, str], float]:
    if system_id == "B2":
        fields = ["trend"]
        penalty = 0.0
    elif system_id == "B3":
        fields = ["intensity", "periodicity", "dominant_axis", "dominant_frequency", "burstiness", "trend"]
        penalty = 0.25
    elif system_id == "M1":
        fields = ["intensity", "periodicity", "dominant_axis", "dominant_frequency", "burstiness", "trend"]
        penalty = 0.75
    else:
        fields = ["intensity", "periodicity", "dominant_axis", "dominant_frequency", "burstiness"]
        penalty = 0.50
    return lambda card, text: _field_score(card, text, fields, contradiction_penalty=penalty)


def evaluate_grounding(records: list[dict], system_id: str) -> dict:
    scorer = make_support_scorer(system_id)
    selection_examples = []
    y_true = []
    y_score = []
    qa_correct = 0
    qa_total = 0

    for record in records:
        card = EvidenceCard.from_json_dict(record["evidence"])
        scores = [scorer(card, item["text"]) for item in record["caption_selection"]["candidates"]]
        selection_examples.append(
            {"answer_index": record["caption_selection"]["answer_index"], "scores": scores}
        )
        positive = record["positive"]
        y_true.append(1)
        y_score.append(scorer(card, positive["text"]))
        for item in record["counterfactuals"]:
            y_true.append(0)
            y_score.append(scorer(card, item["text"]))

        for item in record["qa"]:
            qa_total += 1
            pred = _answer_qa(card, item["fact"], system_id)
            qa_correct += int(str(pred).lower() == str(item["answer"]).lower())

    cf = counterfactual_rejection_metrics(y_true, y_score, threshold=0.5)
    return {
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": cf["accuracy"],
        "cf_reject_precision": cf["precision"],
        "cf_reject_recall": cf["recall"],
        "cf_reject_f1": cf["f1"],
        "qa_accuracy": float(qa_correct / max(1, qa_total)),
        "n_eval_records": len(records),
        "support_score_threshold": 0.5,
    }


def _answer_qa(card: EvidenceCard, fact: str, system_id: str) -> str:
    if system_id == "B2" and fact != "task_understanding":
        if fact == "periodicity":
            return "unknown"
        if fact == "dominant_axis":
            return "unknown"
        if fact == "intensity":
            return "unknown"
    if fact == "dominant_axis":
        return card.dominant_axis
    if fact == "periodicity":
        return card.periodicity
    if fact == "intensity":
        return card.intensity
    if fact == "task_understanding":
        return "counterfactual"
    return "unknown"


def write_checkpoint(path: str | Path, data: dict) -> None:
    from sensorfact.io import write_json

    write_json(path, data)
