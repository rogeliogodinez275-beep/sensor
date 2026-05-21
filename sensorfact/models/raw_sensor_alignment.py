from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import (
    brier_score,
    caption_selection_accuracy,
    counterfactual_rejection_metrics,
    expected_calibration_error,
    macro_f1,
    risk_coverage_auc,
)
from sensorfact.models.alignment import FIELD_VOCABS
from sensorfact.supervised_baseline import extract_statement_fields


class RawSensorFieldPredictor(Protocol):
    def predict_field_proba(self, sensors: list[list[list[float]]]) -> list[dict[str, list[float]]]:
        ...


def _as_sensor_list(row: dict) -> list[list[float]]:
    sensor = row.get("raw_sensor", row.get("sensor"))
    if sensor is None:
        raise KeyError(f"record {row.get('window_id')} is missing raw sensor data")
    return sensor


def join_benchmark_with_windows(records: list[dict], windows: list[dict]) -> list[dict]:
    by_window_id = {str(row["window_id"]): row for row in windows}
    joined: list[dict] = []
    for record in records:
        window_id = str(record["window_id"])
        if window_id not in by_window_id:
            raise KeyError(f"missing raw sensor window for window_id={window_id}")
        window = by_window_id[window_id]
        joined.append(
            {
                **record,
                "raw_sensor": window["sensor"],
                "raw_sensor_split": window.get("split"),
                "raw_sensor_label": window.get("label"),
                "raw_sensor_subject_id": window.get("subject_id"),
            }
        )
    return joined


def score_text_from_field_probs(text: str, field_probs: dict[str, list[float]]) -> float:
    claims = extract_statement_fields(text)
    score = 0.0
    claimed = 0
    for field, value in claims.items():
        vocab = FIELD_VOCABS.get(field)
        probs = field_probs.get(field)
        if vocab is None or probs is None or value not in vocab:
            continue
        score += float(probs[vocab.index(value)])
        claimed += 1
    return float(score / max(1, claimed))


def _field_predictions(field_probs: dict[str, list[float]]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for field, probs in field_probs.items():
        vocab = FIELD_VOCABS[field]
        predictions[field] = vocab[int(np.argmax(np.asarray(probs, dtype=np.float32)))]
    return predictions


def _calibrate_support_threshold(
    predictor: RawSensorFieldPredictor,
    joined_records: list[dict],
) -> float:
    if not joined_records:
        return 0.5
    _, rows = evaluate_raw_sensor_baseline(predictor, joined_records, support_threshold=0.5)
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        labels.extend([1 if item["supported"] else 0 for item in row["support_items"]])
        scores.extend(float(x) for x in row["support_probabilities"])
    if not labels:
        return 0.5
    candidates = sorted({0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, *scores})
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        f1 = counterfactual_rejection_metrics(labels, scores, threshold=threshold)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_raw_sensor_baseline(
    predictor: RawSensorFieldPredictor,
    records: list[dict],
    support_threshold: float = 0.5,
) -> tuple[dict, list[dict]]:
    sensors = [_as_sensor_list(record) for record in records]
    probs_by_record = predictor.predict_field_proba(sensors)
    selection_examples: list[dict] = []
    caption_truth: list[int] = []
    caption_pred: list[int] = []
    caption_correct: list[int] = []
    caption_confidence: list[float] = []
    support_true: list[int] = []
    support_scores: list[float] = []
    cf_reject_true: list[int] = []
    cf_reject_score: list[float] = []
    field_correct = 0
    field_total = 0
    rows: list[dict] = []

    for record, field_probs in zip(records, probs_by_record):
        candidates = record["caption_selection"]["candidates"]
        answer_index = int(record["caption_selection"]["answer_index"])
        caption_scores = [score_text_from_field_probs(str(item.get("text", "")), field_probs) for item in candidates]
        caption_prediction = int(np.argmax(caption_scores)) if caption_scores else 0
        selection_examples.append({"answer_index": answer_index, "scores": caption_scores})
        caption_truth.append(answer_index)
        caption_pred.append(caption_prediction)
        caption_correct.append(1 if caption_prediction == answer_index else 0)
        caption_confidence.append(float(max(caption_scores)) if caption_scores else 0.0)

        support_items = [
            {
                "kind": "positive",
                "text": record["positive"]["text"],
                "supported": True,
                "changed_fact": None,
                "changed_facts": [],
            }
        ]
        for item in record.get("counterfactuals", []):
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
            score_text_from_field_probs(str(item.get("text", "")), field_probs) for item in support_items
        ]
        support_true.extend([1 if item["supported"] else 0 for item in support_items])
        support_scores.extend(support_probabilities)
        for item, probability in zip(support_items, support_probabilities):
            if not item["supported"]:
                cf_reject_true.append(1)
                cf_reject_score.append(float(1.0 - probability))

        evidence_predictions = _field_predictions(field_probs)
        evidence_truth = {
            field: str(record.get("evidence", {}).get(field))
            for field in FIELD_VOCABS
            if record.get("evidence", {}).get(field) is not None
        }
        for field, truth in evidence_truth.items():
            field_correct += int(evidence_predictions.get(field) == truth)
            field_total += 1

        rows.append(
            {
                "window_id": record["window_id"],
                "caption_prediction": caption_prediction,
                "caption_answer_index": answer_index,
                "caption_scores": [float(x) for x in caption_scores],
                "support_probabilities": [float(x) for x in support_probabilities],
                "support_items": support_items,
                "evidence_predictions": evidence_predictions,
                "evidence_probabilities": {
                    field: [float(x) for x in values] for field, values in field_probs.items()
                },
                "evidence_truth_eval_only": evidence_truth,
            }
        )

    support_metrics = counterfactual_rejection_metrics(support_true, support_scores, threshold=support_threshold)
    metrics = {
        "system": "raw_sensor_field_aligner",
        "input_source": "raw_sensor_only",
        "test_time_evidence_access": False,
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "caption_macro_f1": macro_f1(caption_truth, caption_pred),
        "caption_brier": brier_score(caption_correct, caption_confidence),
        "caption_ece": expected_calibration_error(caption_correct, caption_confidence),
        "caption_risk_coverage_auc": risk_coverage_auc(caption_correct, caption_confidence),
        "cf_reject_accuracy": support_metrics["accuracy"],
        "cf_reject_precision": support_metrics["precision"],
        "cf_reject_recall": support_metrics["recall"],
        "cf_reject_f1": support_metrics["f1"],
        "support_brier": brier_score(support_true, support_scores),
        "support_ece": expected_calibration_error(support_true, support_scores),
        "support_risk_coverage_auc": risk_coverage_auc(support_true, support_scores),
        "cf_reject_brier": brier_score(cf_reject_true, cf_reject_score),
        "cf_reject_ece": expected_calibration_error(cf_reject_true, cf_reject_score),
        "cf_reject_risk_coverage_auc": risk_coverage_auc(cf_reject_true, cf_reject_score),
        "evidence_field_accuracy": float(field_correct / max(1, field_total)),
        "n_eval_records": len(records),
        "support_score_threshold": float(support_threshold),
        "field_names": list(FIELD_VOCABS),
        "leakage_audit": (
            "scoring uses raw_sensor -> predicted evidence-field probabilities and candidate text only; "
            "eval evidence is used only for evidence_field_accuracy diagnostics"
        ),
    }
    return metrics, rows


def _sensor_shape(windows: list[dict]) -> tuple[int, int]:
    lengths = []
    channels = []
    for window in windows:
        arr = np.asarray(window["sensor"], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"sensor window {window.get('window_id')} must be 2D [time, channels]")
        lengths.append(int(arr.shape[0]))
        channels.append(int(arr.shape[1]))
    if not lengths:
        raise ValueError("at least one raw sensor window is required")
    return int(max(lengths)), int(max(channels))


def _prepare_sensor(sensor: list[list[float]], length: int, channels: int) -> np.ndarray:
    arr = np.asarray(sensor, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("raw sensor must be a 2D array [time, channels]")
    out = np.zeros((length, channels), dtype=np.float32)
    time = min(length, arr.shape[0])
    chan = min(channels, arr.shape[1])
    out[:time, :chan] = arr[:time, :chan]
    return out


def _targets_from_records(records: list[dict]) -> dict[str, np.ndarray]:
    targets: dict[str, list[int]] = {field: [] for field in FIELD_VOCABS}
    for record in records:
        evidence = record["evidence"]
        for field, vocab in FIELD_VOCABS.items():
            value = str(evidence.get(field))
            targets[field].append(vocab.index(value) if value in vocab else 0)
    return {field: np.asarray(values, dtype=np.int64) for field, values in targets.items()}


@dataclass
class TorchRawSensorFieldPredictor:
    model_path: Path
    length: int
    channels: int
    mean: float
    std: float
    device: str = "cpu"

    def _load_model(self):
        import torch
        from torch import nn

        class _FieldNet(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(channels, 64, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=5, padding=2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Dropout(0.1),
                )
                self.heads = nn.ModuleDict(
                    {field: nn.Linear(128, len(vocab)) for field, vocab in FIELD_VOCABS.items()}
                )

            def forward(self, x):
                hidden = self.encoder(x)
                return {field: head(hidden) for field, head in self.heads.items()}

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = _FieldNet(int(checkpoint["channels"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)
        model.eval()
        self.length = int(checkpoint["length"])
        self.channels = int(checkpoint["channels"])
        self.mean = float(checkpoint["mean"])
        self.std = float(checkpoint["std"])
        return model, torch

    def predict_field_proba(self, sensors: list[list[list[float]]]) -> list[dict[str, list[float]]]:
        model, torch = self._load_model()
        x_np = np.stack([_prepare_sensor(sensor, self.length, self.channels) for sensor in sensors])
        x_np = (x_np - self.mean) / max(self.std, 1e-6)
        x_np = np.transpose(x_np, (0, 2, 1))
        with torch.no_grad():
            tensor = torch.from_numpy(x_np.astype(np.float32)).to(self.device)
            logits = model(tensor)
            probabilities = {field: torch.softmax(value, dim=1).cpu().numpy() for field, value in logits.items()}
        out: list[dict[str, list[float]]] = []
        for idx in range(len(sensors)):
            out.append({field: [float(x) for x in probs[idx]] for field, probs in probabilities.items()})
        return out


def train_raw_sensor_field_model(
    train_windows_path: str | Path,
    train_records_path: str | Path,
    eval_windows_path: str | Path,
    eval_records_path: str | Path,
    metrics_path: str | Path,
    rows_path: str | Path | None = None,
    model_path: str | Path | None = None,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "auto",
    max_train_records: int | None = None,
    max_eval_records: int | None = None,
    max_calibration_records: int = 1024,
) -> tuple[dict, list[dict]]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_windows = list(read_jsonl(train_windows_path))
    train_records = list(read_jsonl(train_records_path))
    eval_windows = list(read_jsonl(eval_windows_path))
    eval_records = list(read_jsonl(eval_records_path))
    if max_train_records is not None and max_train_records >= 0:
        train_records = train_records[:max_train_records]
    if max_eval_records is not None and max_eval_records >= 0:
        eval_records = eval_records[:max_eval_records]
    train_joined = join_benchmark_with_windows(train_records, train_windows)
    eval_joined = join_benchmark_with_windows(eval_records, eval_windows)
    train_windows_joined = [{"sensor": row["raw_sensor"], "window_id": row["window_id"]} for row in train_joined]
    length, channels = _sensor_shape(train_windows_joined)

    x_np = np.stack([_prepare_sensor(row["raw_sensor"], length, channels) for row in train_joined])
    mean = float(np.mean(x_np))
    std = float(np.std(x_np))
    std = std if std >= 1e-6 else 1.0
    x_np = (x_np - mean) / std
    x_np = np.transpose(x_np, (0, 2, 1)).astype(np.float32)
    targets_np = _targets_from_records(train_joined)

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"

    class FieldNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(channels, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.1),
            )
            self.heads = nn.ModuleDict({field: nn.Linear(128, len(vocab)) for field, vocab in FIELD_VOCABS.items()})

        def forward(self, x):
            hidden = self.encoder(x)
            return {field: head(hidden) for field, head in self.heads.items()}

    model = FieldNet(channels).to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x_tensor = torch.from_numpy(x_np)
    target_tensors = {field: torch.from_numpy(values) for field, values in targets_np.items()}
    dataset = TensorDataset(x_tensor, *[target_tensors[field] for field in FIELD_VOCABS])
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        generator=generator,
    )
    field_names = list(FIELD_VOCABS)
    last_loss = math.nan
    for _ in range(max(1, int(epochs))):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in loader:
            batch_x = batch[0].to(resolved_device)
            batch_targets = {field: batch[idx + 1].to(resolved_device) for idx, field in enumerate(field_names)}
            logits = model(batch_x)
            loss = sum(nn.functional.cross_entropy(logits[field], batch_targets[field]) for field in field_names)
            loss = loss / len(field_names)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_batches += 1
        last_loss = total_loss / max(1, total_batches)

    model_path = Path(model_path) if model_path is not None else Path(metrics_path).with_suffix(".pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "length": int(length),
            "channels": int(channels),
            "mean": float(mean),
            "std": float(std),
            "field_vocabs": FIELD_VOCABS,
            "seed": int(seed),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "device": resolved_device,
        },
        model_path,
    )

    predictor = TorchRawSensorFieldPredictor(
        model_path=model_path,
        length=length,
        channels=channels,
        mean=mean,
        std=std,
        device=resolved_device,
    )
    calibration_records = train_joined
    if max_calibration_records is not None and max_calibration_records > 0:
        calibration_records = train_joined[: min(len(train_joined), int(max_calibration_records))]
    support_threshold = _calibrate_support_threshold(predictor, calibration_records)
    metrics, rows = evaluate_raw_sensor_baseline(predictor, eval_joined, support_threshold=support_threshold)
    metrics.update(
        {
            "n_train_records": len(train_joined),
            "n_calibration_records": len(calibration_records),
            "n_eval_records": len(eval_joined),
            "train_windows_path": str(train_windows_path),
            "train_records_path": str(train_records_path),
            "eval_windows_path": str(eval_windows_path),
            "eval_records_path": str(eval_records_path),
            "model_path": str(model_path),
            "device": resolved_device,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "last_train_loss": float(last_loss),
            "support_score_threshold": float(support_threshold),
            "rng_probe": int(rng.integers(0, 2**31 - 1)),
        }
    )
    write_json(metrics_path, metrics)
    if rows_path is not None:
        write_jsonl(rows_path, rows)
    return metrics, rows


def write_leakage_audit(path: str | Path) -> None:
    payload = {
        "oracle_or_upper_bound": [
            "supervised oracle_fields",
            "supervised axis_drop",
            "supervised numeric_drop",
            "aligner base_full",
            "aligner distill_full",
            "aligner riskcal_full",
            "aligner distill_axis_drop",
            "aligner distill_numeric_drop",
        ],
        "not_clean_enough_for_main_claim": [
            "supervised numeric_only if numeric features are read from eval evidence cards",
            "aligner numeric_only if numeric features are read from eval evidence cards",
        ],
        "fair_main_candidate": "raw_sensor_field_aligner",
        "rule": "at test time the model may read raw sensor windows and candidate text only; eval evidence cards are diagnostics only",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
