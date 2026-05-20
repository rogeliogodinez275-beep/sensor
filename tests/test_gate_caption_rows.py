from scripts.gate_caption_rows import gate_caption_rows


def test_gate_caption_rows_uses_alternate_when_margin_is_high():
    primary = [
        {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 1},
        {"window_id": "w2", "caption_prediction": 0, "caption_answer_index": 0},
    ]
    alternate = [
        {"window_id": "w1", "caption_prediction": 1, "caption_scores": [0.0, 3.0]},
        {"window_id": "w2", "caption_prediction": 1, "caption_scores": [0.0, 1.0]},
    ]

    rows = gate_caption_rows(primary, alternate, min_alternate_margin=2.0)

    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["caption_gate_source"] == "alternate"
    assert rows[1]["caption_prediction"] == 0
    assert rows[1]["caption_gate_source"] == "primary"


def test_gate_caption_rows_remaps_alternate_prediction_to_primary_index_space():
    primary = [{"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 5}]
    alternate = [
        {
            "window_id": "w1",
            "candidate_index_map": [2, 5],
            "caption_prediction": 1,
            "caption_scores": [0.0, 3.0],
        }
    ]

    rows = gate_caption_rows(primary, alternate, min_alternate_margin=2.0)

    assert rows[0]["caption_prediction"] == 5


def test_gate_caption_rows_keeps_primary_when_alternate_map_is_bad():
    primary = [{"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 0}]
    alternate = [
        {
            "window_id": "w1",
            "candidate_index_map": [2],
            "caption_prediction": 3,
            "caption_scores": [0.0, 3.0, 1.0, 6.0],
        }
    ]

    rows = gate_caption_rows(primary, alternate, min_alternate_margin=2.0)

    assert rows[0]["caption_prediction"] == 0
    assert rows[0]["caption_gate_source"] == "primary"
    assert rows[0]["caption_gate_bad_alternate_map"] is True
