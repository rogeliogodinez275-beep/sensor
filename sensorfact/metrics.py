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


def majority_baseline(labels: Iterable[str]) -> str | None:
    counts = Counter(labels)
    return counts.most_common(1)[0][0] if counts else None
