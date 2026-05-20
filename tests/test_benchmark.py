from sensorfact.benchmark import SensorFactBenchmarkBuilder
from sensorfact.schemas import EvidenceCard


def make_card() -> EvidenceCard:
    return EvidenceCard(
        window_id="w1",
        dataset_id="toy",
        label="walking",
        subject_id="s1",
        intensity="high",
        periodicity="strong",
        dominant_axis="acc_x",
        dominant_frequency="mid",
        cross_channel_relation="acc_x and acc_y move synchronously",
        lead_lag="no clear lag",
        burstiness="smooth",
        trend_segments=["rise", "stable", "fall"],
        numeric={"rms_energy": 1.2},
    )


def test_benchmark_builder_creates_supported_and_counterfactual_pairs():
    builder = SensorFactBenchmarkBuilder(seed=7)
    record = builder.build_record(make_card())

    assert record["window_id"] == "w1"
    assert record["positive"]["supported"] is True
    assert len(record["counterfactuals"]) >= 4
    assert all(item["supported"] is False for item in record["counterfactuals"])
    assert all(item["changed_fact"] for item in record["counterfactuals"])
    assert len(record["caption_selection"]["candidates"]) == 4
    assert record["caption_selection"]["candidates"][record["caption_selection"]["answer_index"]][
        "supported"
    ]


def test_benchmark_builder_makes_label_obfuscated_questions():
    builder = SensorFactBenchmarkBuilder(seed=3)
    record = builder.build_record(make_card())

    question_text = " ".join(item["question"].lower() for item in record["qa"])

    assert "walking" not in question_text
    assert any(item["answer"] == "acc_x" for item in record["qa"])
