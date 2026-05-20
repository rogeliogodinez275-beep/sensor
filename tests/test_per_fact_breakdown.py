from scripts.analyze_per_fact_breakdown import analyze_per_fact_breakdown


def test_analyze_per_fact_breakdown_groups_by_gold_and_wrong_fact():
    benchmark = [
        {
            "window_id": "w1",
            "caption_selection": {
                "answer_index": 0,
                "candidates": [
                    {"text": "right", "changed_fact": None},
                    {"text": "wrong axis", "changed_fact": "dominant_axis"},
                ],
            },
        },
        {
            "window_id": "w2",
            "caption_selection": {
                "answer_index": 0,
                "candidates": [
                    {"text": "right", "changed_fact": None},
                    {"text": "wrong rhythm", "changed_fact": "periodicity"},
                ],
            },
        },
    ]
    rows = [
        {"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 0},
        {"window_id": "w2", "caption_prediction": 0, "caption_answer_index": 0},
    ]

    report = analyze_per_fact_breakdown(rows, benchmark)

    assert report["overall"]["n"] == 2
    assert report["overall"]["accuracy"] == 0.5
    assert report["by_predicted_wrong_fact"]["dominant_axis"]["n"] == 1
    assert report["by_gold_changed_fact"]["supported"]["accuracy"] == 0.5

