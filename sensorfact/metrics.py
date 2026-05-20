from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def accuracy(y_true: Iterable, y_pred: Iterable) -> float:
    pairs = list(zip(y_true, y_pred))
    if not pairs:
        return 0.0
    return float(sum(1 for truth, pred in pairs if truth == pred) / len(pairs))


def caption_selection_accuracy(examples: list[dict]) -> float:
    if not examples:
        return 0.0
    correct = 0
    for example in examples:
        scores = list(example["scores"])
        pred = int(np.argmax(scores))
        correct += int(pred == int(example["answer_index"]))
    return float(correct / len(examples))


def counterfactual_rejection_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    truth = [int(x) for x in y_true]
    pred = [1 if float(score) >= threshold else 0 for score in y_score]
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    total = max(1, len(truth))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float((tp + tn) / total),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def macro_f1(y_true: Iterable, y_pred: Iterable) -> float:
    truth = list(y_true)
    pred = list(y_pred)
    labels = sorted(set(truth) | set(pred))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, pred) if t == label and p != label)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def brier_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    truth = np.asarray([int(x) for x in y_true], dtype=np.float32)
    score = np.asarray([float(x) for x in y_score], dtype=np.float32)
    if truth.size == 0:
        return 0.0
    return float(np.mean(np.square(score - truth)))


def expected_calibration_error(
    y_true: Iterable[int],
    y_score: Iterable[float],
    n_bins: int = 10,
) -> float:
    truth = np.asarray([int(x) for x in y_true], dtype=np.float32)
    score = np.asarray([float(x) for x in y_score], dtype=np.float32)
    if truth.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = truth.size
    ece = 0.0
    for idx in range(n_bins):
        left, right = bins[idx], bins[idx + 1]
        if idx == n_bins - 1:
            mask = (score >= left) & (score <= right)
        else:
            mask = (score >= left) & (score < right)
        if not np.any(mask):
            continue
        accuracy_value = float(np.mean(truth[mask]))
        confidence_value = float(np.mean(score[mask]))
        ece += float(np.sum(mask)) / total * abs(accuracy_value - confidence_value)
    return float(ece)


def risk_coverage_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    truth = np.asarray([int(x) for x in y_true], dtype=np.int64)
    score = np.asarray([float(x) for x in y_score], dtype=np.float32)
    if truth.size == 0:
        return 0.0
    pred = (score >= 0.5).astype(np.int64)
    confidence = np.maximum(score, 1.0 - score)
    order = np.argsort(-confidence)
    correct = (pred[order] == truth[order]).astype(np.float32)
    coverages = np.arange(1, truth.size + 1, dtype=np.float32) / truth.size
    risks = 1.0 - np.cumsum(correct) / np.arange(1, truth.size + 1, dtype=np.float32)
    risk_area = float(np.trapezoid(risks, coverages))
    return float(max(0.0, min(1.0, 1.0 - risk_area)))


def majority_baseline(labels: Iterable[str]) -> str | None:
    counts = Counter(labels)
    return counts.most_common(1)[0][0] if counts else None
