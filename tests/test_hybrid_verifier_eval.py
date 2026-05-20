from sensorfact.hybrid_verifier_eval import evaluate_hybrid_grounding


def test_hybrid_uses_structured_caption_when_decisive_and_direct_when_ambiguous():
    structured_rows = [
        {
            "window_id": "w1",
            "caption_answer_index": 1,
            "caption_scores": [0.0, 1.0, 0.0],
            "support_predictions": [True, False],
            "support_items": [
                {"text": "positive", "supported": True},
                {"text": "negative", "supported": False},
            ],
        },
        {
            "window_id": "w2",
            "caption_answer_index": 2,
            "caption_scores": [0.0, 0.0, 0.0],
            "support_predictions": [True, True],
            "support_items": [
                {"text": "positive", "supported": True},
                {"text": "negative", "supported": False},
            ],
        },
    ]
    direct_rows = [{"window_id": "w2", "caption_prediction": 2}]

    metrics, rows = evaluate_hybrid_grounding(structured_rows, direct_rows)

    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["caption_fallback_count"] == 1
    assert metrics["caption_fallback_accuracy"] == 1.0
    assert metrics["support_source"] == "structured_only"
    assert rows[0]["caption_source"] == "structured"
    assert rows[1]["caption_source"] == "direct_fallback"
    assert metrics["cf_reject_accuracy"] == 0.75
    assert metrics["cf_reject_f1"] == 0.8


def test_hybrid_records_missing_direct_fallback_without_using_labels():
    structured_rows = [
        {
            "window_id": "w1",
            "caption_answer_index": 1,
            "caption_scores": [0.0, 0.0, 0.0],
            "support_predictions": [],
            "support_items": [],
        }
    ]

    metrics, rows = evaluate_hybrid_grounding(structured_rows, [])

    assert metrics["missing_direct_count"] == 1
    assert rows[0]["caption_prediction"] == 0


def test_hybrid_cf_metrics_ignore_direct_support_predictions():
    structured_rows = [
        {
            "window_id": "w1",
            "caption_answer_index": 0,
            "caption_scores": [1.0, 0.0, 0.0],
            "support_predictions": [True, False],
            "support_items": [
                {"text": "positive", "supported": True},
                {"text": "negative", "supported": False},
            ],
        }
    ]
    direct_rows = [
        {
            "window_id": "w1",
            "caption_prediction": 2,
            "support_predictions": [False, True],
        }
    ]

    metrics, rows = evaluate_hybrid_grounding(structured_rows, direct_rows)

    assert rows[0]["support_predictions"] == [True, False]
    assert metrics["cf_reject_f1"] == 1.0


def test_hybrid_remaps_direct_prediction_from_constrained_candidate_subset():
    structured_rows = [
        {
            "window_id": "w1",
            "caption_answer_index": 3,
            "caption_scores": [0.0, 0.0, 0.0, 0.0],
            "support_predictions": [],
            "support_items": [],
        }
    ]
    direct_rows = [
        {
            "window_id": "w1",
            "candidate_index_map": [1, 3],
            "caption_prediction": 1,
        }
    ]

    metrics, rows = evaluate_hybrid_grounding(structured_rows, direct_rows)

    assert metrics["caption_fallback_count"] == 1
    assert rows[0]["caption_source"] == "direct_fallback"
    assert rows[0]["caption_prediction"] == 3
