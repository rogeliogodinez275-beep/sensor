from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.io import write_jsonl
from sensorfact.models.raw_sensor_alignment import (
    FIELD_VOCABS,
    RawSensorFieldPredictor,
    evaluate_raw_sensor_baseline,
    join_benchmark_with_windows,
    score_text_from_field_probs,
)


def make_window(window_id: str, value: float, split: str = "train") -> dict:
    return {
        "window_id": window_id,
        "dataset_id": "toy",
        "split": split,
        "label": "walk" if value < 0.5 else "run",
        "subject_id": "s1",
        "sensor": [[value, value + 0.1], [value + 0.2, value + 0.3]],
    }


def make_record(window_id: str, intensity: str, answer_index: int = 0) -> dict:
    positive = (
        f"The motion intensity is {intensity}. The signal shows strong periodicity. "
        "The dominant movement axis is acc_x. The dominant frequency is low. "
        "The movement is smooth."
    )
    negative_intensity = "high" if intensity == "low" else "low"
    negative = (
        f"The motion intensity is {negative_intensity}. The signal shows strong periodicity. "
        "The dominant movement axis is acc_x. The dominant frequency is low. "
        "The movement is smooth."
    )
    return {
        "window_id": window_id,
        "dataset_id": "toy",
        "label": "walk" if intensity == "low" else "run",
        "evidence": {
            "window_id": window_id,
            "dataset_id": "toy",
            "intensity": intensity,
            "periodicity": "strong",
            "dominant_axis": "acc_x",
            "dominant_frequency": "low",
            "burstiness": "smooth",
            "trend_segments": ["stable"],
            "confidence": 1.0,
            "numeric": {"rms_energy": 99.0 if intensity == "high" else -99.0},
        },
        "positive": {"text": positive, "supported": True, "changed_fact": None},
        "counterfactuals": [{"text": negative, "supported": False, "changed_fact": "intensity"}],
        "caption_selection": {
            "answer_index": answer_index,
            "candidates": [
                {"text": positive, "supported": True, "changed_fact": None},
                {"text": negative, "supported": False, "changed_fact": "intensity"},
            ],
        },
    }


def test_join_benchmark_with_windows_keeps_mapping_and_raw_sensor():
    records = [make_record("w1", "low"), make_record("w2", "high")]
    windows = [make_window("w1", 0.1), make_window("w2", 1.0)]

    joined = join_benchmark_with_windows(records, windows)

    assert [row["window_id"] for row in joined] == ["w1", "w2"]
    assert joined[0]["raw_sensor"] == windows[0]["sensor"]
    assert joined[1]["caption_selection"]["candidates"][0]["text"] == records[1]["caption_selection"]["candidates"][0]["text"]


def test_join_benchmark_with_windows_fails_on_missing_window():
    with pytest.raises(KeyError, match="missing raw sensor window"):
        join_benchmark_with_windows([make_record("w_missing", "low")], [make_window("w1", 0.1)])


def test_score_text_from_field_probs_does_not_read_record_evidence():
    probs = {
        "intensity": [0.9, 0.1, 0.0],
        "periodicity": [0.0, 0.0, 1.0],
        "dominant_axis": [1.0] + [0.0] * (len(FIELD_VOCABS["dominant_axis"]) - 1),
        "dominant_frequency": [1.0, 0.0, 0.0, 0.0],
        "burstiness": [1.0, 0.0],
    }
    low_text = make_record("w1", "low")["positive"]["text"]
    high_text = make_record("w1", "high")["positive"]["text"]

    assert score_text_from_field_probs(low_text, probs) > score_text_from_field_probs(high_text, probs)


def test_eval_candidate_scores_are_unchanged_when_eval_evidence_is_mutated():
    class FixedPredictor(RawSensorFieldPredictor):
        def predict_field_proba(self, sensors):
            return [
                {
                    "intensity": [0.9, 0.1, 0.0],
                    "periodicity": [0.0, 0.0, 1.0],
                    "dominant_axis": [1.0] + [0.0] * (len(FIELD_VOCABS["dominant_axis"]) - 1),
                    "dominant_frequency": [1.0, 0.0, 0.0, 0.0],
                    "burstiness": [1.0, 0.0],
                }
                for _ in sensors
            ]

    clean = join_benchmark_with_windows([make_record("w1", "low")], [make_window("w1", 0.1)])
    mutated_record = make_record("w1", "low")
    mutated_record["evidence"] = {
        **mutated_record["evidence"],
        "intensity": "high",
        "periodicity": "none",
        "dominant_axis": "gyro_z",
        "dominant_frequency": "high",
        "burstiness": "bursty",
        "numeric": {"rms_energy": 12345.0},
    }
    mutated = join_benchmark_with_windows([mutated_record], [make_window("w1", 0.1)])

    _, clean_rows = evaluate_raw_sensor_baseline(FixedPredictor(), clean)
    _, mutated_rows = evaluate_raw_sensor_baseline(FixedPredictor(), mutated)

    assert clean_rows[0]["caption_scores"] == mutated_rows[0]["caption_scores"]
    assert clean_rows[0]["support_probabilities"] == mutated_rows[0]["support_probabilities"]
    assert clean_rows[0]["evidence_predictions"] == mutated_rows[0]["evidence_predictions"]


def test_raw_sensor_training_smoke_when_torch_is_available(tmp_path: Path):
    pytest.importorskip("torch")
    from sensorfact.models.raw_sensor_alignment import train_raw_sensor_field_model

    train_windows = [make_window(f"w{i}", 0.1 if i % 2 == 0 else 1.0) for i in range(8)]
    train_records = [make_record(f"w{i}", "low" if i % 2 == 0 else "high") for i in range(8)]
    eval_windows = [make_window("e1", 0.1, split="test")]
    eval_records = [make_record("e1", "low")]

    train_window_path = tmp_path / "train_windows.jsonl"
    train_record_path = tmp_path / "train_records.jsonl"
    eval_window_path = tmp_path / "eval_windows.jsonl"
    eval_record_path = tmp_path / "eval_records.jsonl"
    metrics_path = tmp_path / "metrics.json"
    rows_path = tmp_path / "rows.jsonl"
    model_path = tmp_path / "model.pt"
    write_jsonl(train_window_path, train_windows)
    write_jsonl(train_record_path, train_records)
    write_jsonl(eval_window_path, eval_windows)
    write_jsonl(eval_record_path, eval_records)

    metrics, rows = train_raw_sensor_field_model(
        train_windows_path=train_window_path,
        train_records_path=train_record_path,
        eval_windows_path=eval_window_path,
        eval_records_path=eval_record_path,
        metrics_path=metrics_path,
        rows_path=rows_path,
        model_path=model_path,
        epochs=2,
        batch_size=4,
        device="cpu",
    )

    assert metrics["input_source"] == "raw_sensor_only"
    assert metrics["n_train_records"] == 8
    assert metrics["n_eval_records"] == 1
    assert rows[0]["caption_prediction"] in {0, 1}
    assert model_path.exists()
