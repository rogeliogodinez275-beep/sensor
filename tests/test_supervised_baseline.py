from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.benchmark import SensorFactBenchmarkBuilder
from sensorfact.hard_benchmark import build_hard_record
from sensorfact.schemas import EvidenceCard
from sensorfact.supervised_baseline import (
    evaluate_supervised_grounding,
    extract_statement_fields,
    statement_feature_vector,
)


def make_card(
    window_id: str,
    intensity: str,
    periodicity: str,
    axis: str,
    frequency: str,
    burstiness: str,
    trend_segments: list[str],
) -> EvidenceCard:
    return EvidenceCard(
        window_id=window_id,
        dataset_id="toy",
        label="toy_label",
        subject_id=window_id,
        intensity=intensity,
        periodicity=periodicity,
        dominant_axis=axis,
        dominant_frequency=frequency,
        cross_channel_relation="no clear relation",
        lead_lag="no clear lag",
        burstiness=burstiness,
        trend_segments=trend_segments,
        confidence=1.0,
        numeric={
            "rms_energy": 0.1 if intensity == "low" else (0.3 if intensity == "medium" else 0.7),
            "autocorr_peak": 0.2 if periodicity == "none" else (0.5 if periodicity == "weak" else 0.9),
            "fft_dominant_ratio": 0.1 if periodicity == "none" else (0.2 if periodicity == "weak" else 0.3),
            "dominant_frequency_hz": 0.4 if frequency == "low" else (0.8 if frequency == "mid" else 1.2),
            "peak_count": 4.0 if burstiness == "smooth" else 12.0,
        },
    )


def make_train_records() -> list[dict]:
    builder = SensorFactBenchmarkBuilder(seed=7)
    return [
        builder.build_record(make_card("w1", "low", "strong", "acc_x", "low", "bursty", ["rise", "fall", "rise"])),
        builder.build_record(make_card("w2", "medium", "weak", "gyro_y", "high", "smooth", ["fall", "fall", "stable"])),
        builder.build_record(make_card("w3", "high", "none", "acc_z", "mid", "smooth", ["stable", "rise", "stable"])),
        builder.build_record(make_card("w4", "medium", "strong", "gyro_x", "high", "bursty", ["fall", "rise", "fall"])),
    ]


def make_eval_records(train_records: list[dict]) -> list[dict]:
    return [build_hard_record(record, seed=19, variant="v3") for record in train_records[:2]]


def test_extract_statement_fields_handles_benchmark_and_hard_v3_phrasings():
    train_records = make_train_records()
    hard_record = build_hard_record(train_records[0], seed=11, variant="v3")

    simple_fields = extract_statement_fields(train_records[0]["positive"]["text"])
    paraphrase_fields = extract_statement_fields(train_records[1]["paraphrases"][0]["text"])
    hard_fields = extract_statement_fields(hard_record["positive"]["text"])

    assert simple_fields["intensity"] == "low"
    assert simple_fields["periodicity"] == "strong"
    assert paraphrase_fields["dominant_axis"] == "gyro_y"
    assert hard_fields["dominant_frequency"] == "low"
    assert hard_fields["burstiness"] == "bursty"


def test_supervised_baseline_scores_toy_hard_records():
    train_records = make_train_records()
    eval_records = make_eval_records(train_records)

    metrics, rows = evaluate_supervised_grounding(
        train_records=train_records,
        eval_records=eval_records,
        model_type="random_forest",
        hard_variant="v3",
    )

    assert metrics["caption_selection_accuracy"] >= 0.99
    assert metrics["cf_reject_accuracy"] >= 0.99
    assert metrics["cf_reject_f1"] >= 0.99
    assert metrics["n_eval_records"] == len(eval_records)
    assert rows[0]["caption_prediction"] == rows[0]["caption_answer_index"]
    assert len(rows[0]["support_probabilities"]) == len(rows[0]["support_items"])


def test_numeric_only_features_do_not_include_discrete_evidence_labels():
    train_records = make_train_records()
    record = train_records[0]
    changed = {
        **record,
        "evidence": {
            **record["evidence"],
            "intensity": "high",
            "periodicity": "none",
            "dominant_axis": "gyro_z",
            "dominant_frequency": "high",
            "burstiness": "smooth",
        },
    }
    text = record["positive"]["text"]

    numeric_original = statement_feature_vector(record, text, feature_mode="numeric_only")
    numeric_changed = statement_feature_vector(changed, text, feature_mode="numeric_only")
    oracle_original = statement_feature_vector(record, text, feature_mode="oracle_fields")
    oracle_changed = statement_feature_vector(changed, text, feature_mode="oracle_fields")

    assert numeric_original.tolist() == numeric_changed.tolist()
    assert oracle_original.tolist() != oracle_changed.tolist()
