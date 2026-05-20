from sensorfact.hard_benchmark import build_hard_record
from sensorfact.qwen_llm_eval import build_caption_prompt
from tests.test_benchmark import make_card


def test_hard_record_uses_natural_evidence_without_template_field_names():
    base = {
        "window_id": "w1",
        "dataset_id": "toy",
        "label": "walking",
        "subject_id": "s1",
        "evidence": make_card().to_json_dict(),
    }

    record = build_hard_record(base, seed=3)

    assert "evidence_text" in record
    lower = record["evidence_text"].lower()
    assert "motion intensity" not in lower
    assert "dominant movement axis" not in lower
    assert "dominant frequency" not in lower
    assert len(record["caption_selection"]["candidates"]) == 4
    assert record["caption_selection"]["candidates"][record["caption_selection"]["answer_index"]][
        "supported"
    ]
    assert any(item["changed_fact"] == "dominant_axis" for item in record["counterfactuals"])


def test_hard_record_handles_uncertain_axis_and_trends_cleanly():
    card = make_card()
    data = card.to_json_dict()
    data["dominant_axis"] = "uncertain"
    data["trend_segments"] = ["fall", "rise", "rise"]
    base = {
        "window_id": "w2",
        "dataset_id": "toy",
        "label": "standing",
        "subject_id": "s2",
        "evidence": data,
    }

    record = build_hard_record(base, seed=5)

    assert "falling, then rising, then rising" in record["evidence_text"]
    assert "in the no single" not in record["evidence_text"]
    assert "from the no single" not in record["positive"]["text"]


def test_llm_prompt_prefers_hard_evidence_text_when_available():
    base = {
        "window_id": "w1",
        "dataset_id": "toy",
        "label": "walking",
        "subject_id": "s1",
        "evidence": make_card().to_json_dict(),
    }
    record = build_hard_record(base, seed=7)

    prompt = build_caption_prompt(record)

    assert record["evidence_text"] in prompt
    assert "The motion intensity is high" not in prompt


def test_hard_v2_positive_statement_does_not_use_support_labeling_shortcut():
    base = {
        "window_id": "w3",
        "dataset_id": "toy",
        "label": "walking",
        "subject_id": "s3",
        "evidence": make_card().to_json_dict(),
    }

    record = build_hard_record(base, seed=11, variant="v2")

    assert not record["positive"]["text"].lower().startswith("a supported description")
    assert "description claims" not in " ".join(item["text"].lower() for item in record["counterfactuals"])
    assert "the trace shows the trace" not in " ".join(
        item["text"].lower() for item in record["counterfactuals"]
    )
    assert record["difficulty"] == "natural_language_near_miss_v2"


def test_hard_v3_uses_numeric_evidence_and_multi_fact_near_misses():
    base = {
        "window_id": "w4",
        "dataset_id": "toy",
        "label": "walking",
        "subject_id": "s4",
        "evidence": make_card().to_json_dict(),
    }

    record = build_hard_record(base, seed=13, variant="v3")

    assert record["difficulty"] == "numeric_partial_contradiction_v3"
    assert "rms energy" in record["evidence_text"].lower()
    assert "large-amplitude movement" not in record["evidence_text"].lower()
    assert any(len(item.get("changed_facts", [])) >= 2 for item in record["counterfactuals"])
    assert len(record["caption_selection"]["candidates"]) == 4
    assert record["caption_selection"]["candidates"][record["caption_selection"]["answer_index"]][
        "supported"
    ]


def test_hard_v3_preserves_mhealth_axis_space_for_counterfactuals():
    card = make_card()
    data = card.to_json_dict()
    data["dataset_id"] = "mhealth"
    data["dominant_axis"] = "ankle_mag_x"
    base = {
        "window_id": "m1",
        "dataset_id": "mhealth",
        "label": "walking",
        "subject_id": "s4",
        "evidence": data,
    }

    record = build_hard_record(base, seed=13, variant="v3")

    assert "ankle magnetometer x trace" in record["evidence_text"]
    axis_counterfactuals = [
        item for item in record["counterfactuals"] if "dominant_axis" in item.get("changed_facts", [])
    ]
    assert axis_counterfactuals
    assert "chest forward-back acceleration trace" in axis_counterfactuals[0]["text"]
