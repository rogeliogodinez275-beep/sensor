from scripts.sweep_hybrid_threshold import sweep_thresholds


def test_sweep_thresholds_prefers_threshold_with_best_caption_accuracy():
    structured_rows = [
        {
            "window_id": "w1",
            "caption_answer_index": 0,
            "caption_scores": [0.6, 0.2],
            "support_predictions": [True],
            "support_items": [{"supported": True}],
        },
        {
            "window_id": "w2",
            "caption_answer_index": 0,
            "caption_scores": [0.4, 0.0],
            "support_predictions": [True],
            "support_items": [{"supported": True}],
        },
    ]
    direct_rows = [
        {"window_id": "w1", "caption_prediction": 1},
        {"window_id": "w2", "caption_prediction": 1},
    ]

    rows, best = sweep_thresholds(structured_rows, direct_rows, thresholds=[0.0, 0.25, 0.5, 0.75])

    assert len(rows) == 4
    assert best["threshold"] == 0.25
    assert best["caption_selection_accuracy"] == 1.0
    assert rows[-1]["caption_selection_accuracy"] == 0.0
