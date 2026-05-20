from scripts.build_evidence_control_benchmark import build_control_records


def make_records():
    return [
        {
            "window_id": "w1",
            "dataset_id": "toy",
            "evidence": {
                "window_id": "w1",
                "dataset_id": "toy",
                "intensity": "low",
                "periodicity": "strong",
                "dominant_axis": "acc_x",
                "dominant_frequency": "low",
                "cross_channel_relation": "no clear relation",
                "lead_lag": "no clear lag",
                "burstiness": "bursty",
                "trend_segments": ["rise", "stable", "fall"],
                "confidence": 1.0,
                "numeric": {
                    "rms_energy": 1.23,
                    "autocorr_peak": 0.81,
                    "fft_dominant_ratio": 0.42,
                    "dominant_frequency_hz": 1.5,
                    "peak_count": 7,
                },
            },
            "caption_selection": {
                "answer_index": 1,
                "candidates": [{"text": "wrong"}, {"text": "right"}],
            },
        },
        {
            "window_id": "w2",
            "dataset_id": "toy",
            "evidence": {
                "window_id": "w2",
                "dataset_id": "toy",
                "intensity": "high",
                "periodicity": "none",
                "dominant_axis": "gyro_y",
                "dominant_frequency": "high",
                "cross_channel_relation": "no clear relation",
                "lead_lag": "no clear lag",
                "burstiness": "smooth",
                "trend_segments": ["fall", "rise", "stable"],
                "confidence": 1.0,
                "numeric": {
                    "rms_energy": 9.87,
                    "autocorr_peak": 0.11,
                    "fft_dominant_ratio": 0.05,
                    "dominant_frequency_hz": 4.0,
                    "peak_count": 2,
                },
            },
            "caption_selection": {
                "answer_index": 0,
                "candidates": [{"text": "right"}, {"text": "wrong"}],
            },
        },
    ]


def test_visible_control_preserves_mapping_and_adds_metadata():
    rows = build_control_records(make_records(), mode="visible", seed=3)

    assert [row["window_id"] for row in rows] == ["w1", "w2"]
    assert [row["caption_selection"] for row in rows] == [
        record["caption_selection"] for record in make_records()
    ]
    assert all(row["evidence_control"] == "visible" for row in rows)
    assert [row["evidence_source_window_id"] for row in rows] == ["w1", "w2"]


def test_numeric_swap_changes_only_numeric_evidence_without_changing_gold_mapping():
    rows = build_control_records(make_records(), mode="numeric_swap", seed=11)

    assert [row["caption_selection"] for row in rows] == [
        record["caption_selection"] for record in make_records()
    ]
    assert rows[0]["evidence"]["intensity"] == "low"
    assert rows[0]["evidence"]["dominant_axis"] == "acc_x"
    assert rows[0]["evidence"]["numeric"]["rms_energy"] == 9.87
    assert rows[0]["evidence_source_window_id"] == "w2"
    assert "9.870" in rows[0]["evidence_text"]
    assert rows[0]["evidence_control"] == "numeric-swap"


def test_axis_permutation_changes_axis_evidence_but_not_candidates():
    rows = build_control_records(make_records(), mode="axis_permutation", seed=5)

    assert rows[0]["caption_selection"] == make_records()[0]["caption_selection"]
    assert rows[0]["evidence"]["dominant_axis"] != "acc_x"
    assert rows[0]["evidence_control"] == "axis-permutation"


def test_trend_flip_changes_rise_and_fall_only():
    rows = build_control_records(make_records(), mode="trend_flip", seed=5)

    assert rows[0]["evidence"]["trend_segments"] == ["fall", "stable", "rise"]
    assert rows[0]["caption_selection"] == make_records()[0]["caption_selection"]
    assert rows[0]["evidence_control"] == "trend-flip"
