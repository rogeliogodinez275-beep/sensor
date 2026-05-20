from pathlib import Path

from sensorfact.eval.eval_alignment import evaluate_alignment
from sensorfact.io import write_jsonl
from sensorfact.models.alignment import (
    LightweightSensorTextAligner,
    build_alignment_dataset,
)
from sensorfact.training.train_alignment import train_alignment_model


def make_record(window_id: str, label: str, rms: float, axis: str) -> dict:
    positive = f"The motion intensity is {'high' if rms > 1.0 else 'low'}. The dominant movement axis is {axis}."
    negative = f"The motion intensity is {'low' if rms > 1.0 else 'high'}. The dominant movement axis is {axis}."
    return {
        "window_id": window_id,
        "dataset_id": "toy",
        "label": label,
        "evidence": {
            "window_id": window_id,
            "dataset_id": "toy",
            "intensity": "high" if rms > 1.0 else "low",
            "periodicity": "strong",
            "dominant_axis": axis,
            "dominant_frequency": "low",
            "cross_channel_relation": "no clear relation",
            "lead_lag": "no clear lag",
            "burstiness": "smooth",
            "trend_segments": ["rise", "stable", "fall"],
            "confidence": 1.0,
            "numeric": {
                "rms_energy": rms,
                "autocorr_peak": 0.8,
                "fft_dominant_ratio": 0.5,
                "dominant_frequency_hz": 1.0,
                "peak_count": 4,
            },
        },
        "positive": {"text": positive, "supported": True, "changed_fact": None},
        "counterfactuals": [{"text": negative, "supported": False, "changed_fact": "intensity"}],
        "caption_selection": {
            "answer_index": 0,
            "candidates": [
                {"text": positive, "supported": True, "changed_fact": None},
                {"text": negative, "supported": False, "changed_fact": "intensity"},
            ],
        },
    }


def test_alignment_dataset_contains_caption_and_evidence_targets():
    records = [
        make_record("w1", "walk", 0.5, "acc_x"),
        make_record("w2", "run", 2.0, "gyro_y"),
    ]

    dataset = build_alignment_dataset(records)

    assert dataset.features.shape[0] == 2
    assert dataset.caption_labels.tolist() == [0, 1]
    assert "intensity" in dataset.field_names
    assert dataset.field_targets["intensity"].tolist() == [0, 2]


def test_lightweight_alignment_train_and_eval_smoke(tmp_path: Path):
    train_records = [
        make_record("w1", "walk", 0.5, "acc_x"),
        make_record("w2", "run", 2.0, "gyro_y"),
        make_record("w3", "walk", 0.4, "acc_x"),
        make_record("w4", "run", 2.2, "gyro_y"),
    ]
    eval_records = [make_record("w5", "walk", 0.6, "acc_x")]
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    metrics_path = tmp_path / "metrics.json"
    rows_path = tmp_path / "rows.jsonl"
    model_path = tmp_path / "alignment_model.json"
    write_jsonl(train_path, train_records)
    write_jsonl(eval_path, eval_records)

    metrics = train_alignment_model(
        train_path=train_path,
        eval_path=eval_path,
        model_path=model_path,
        metrics_path=metrics_path,
        rows_path=rows_path,
        epochs=60,
        learning_rate=0.2,
        seed=7,
    )
    model = LightweightSensorTextAligner.load(model_path)
    eval_metrics, rows = evaluate_alignment(model, eval_records)

    assert metrics["n_train_records"] == 4
    assert metrics["n_eval_records"] == 1
    assert metrics_path.exists()
    assert rows_path.exists()
    assert eval_metrics["caption_selection_accuracy"] >= 0.0
    assert rows[0]["caption_prediction"] in {0, 1}
