from scripts.build_evidence_control_benchmark import (
    build_control_records,
    mask_numeric_evidence_text,
)


def make_records():
    return [
        {
            "window_id": "w1",
            "evidence": {"numeric": {"rms_energy": 1.23}},
            "evidence_text": "RMS energy is 1.230 and cadence is 2 Hz.",
            "caption_selection": {"answer_index": 0, "candidates": [{"text": "a"}]},
        },
        {
            "window_id": "w2",
            "evidence": {"numeric": {"rms_energy": 9.87}},
            "evidence_text": "RMS energy is 9.870 and cadence is 4 Hz.",
            "caption_selection": {"answer_index": 0, "candidates": [{"text": "b"}]},
        },
        {
            "window_id": "w3",
            "evidence": {"numeric": {"rms_energy": 5.55}},
            "evidence_text": "RMS energy is 5.550 and cadence is 6 Hz.",
            "caption_selection": {"answer_index": 0, "candidates": [{"text": "c"}]},
        },
    ]


def test_mask_numeric_evidence_text_replaces_numeric_values():
    masked = mask_numeric_evidence_text("RMS energy is 1.230 and cadence is -2 Hz.")

    assert "1.230" not in masked
    assert "-2" not in masked
    assert "<num>" in masked
    assert "RMS energy" in masked


def test_build_control_records_shuffles_evidence_without_changing_candidates():
    rows = build_control_records(make_records(), mode="shuffled", seed=7)

    assert [row["window_id"] for row in rows] == ["w1", "w2", "w3"]
    assert [row["caption_selection"] for row in rows] == [
        record["caption_selection"] for record in make_records()
    ]
    assert all(row["evidence_control"] == "shuffled" for row in rows)
    assert all(row["evidence_source_window_id"] != row["window_id"] for row in rows)
    assert {row["evidence_text"] for row in rows} == {
        record["evidence_text"] for record in make_records()
    }


def test_build_control_records_masks_numeric_evidence():
    rows = build_control_records(make_records()[:1], mode="numeric-mask", seed=7)

    assert rows[0]["window_id"] == "w1"
    assert rows[0]["evidence_control"] == "numeric-mask"
    assert "1.230" not in rows[0]["evidence_text"]
    assert "<num>" in rows[0]["evidence_text"]
    assert rows[0]["caption_selection"] == make_records()[0]["caption_selection"]

