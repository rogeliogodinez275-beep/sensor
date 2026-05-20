from scripts.analyze_failure_taxonomy import analyze_failure_taxonomy


def test_analyze_failure_taxonomy_counts_corrections_and_regressions_by_changed_fact():
    benchmark = [
        {
            "window_id": "fixed",
            "caption_selection": {
                "answer_index": 1,
                "candidates": [
                    {"text": "wrong axis", "changed_fact": "dominant_axis"},
                    {"text": "right", "changed_fact": None},
                ],
            },
        },
        {
            "window_id": "broken",
            "caption_selection": {
                "answer_index": 0,
                "candidates": [
                    {"text": "right", "changed_fact": None},
                    {"text": "wrong periodicity", "changed_fact": "periodicity"},
                ],
            },
        },
    ]
    primary_rows = [
        {"window_id": "fixed", "caption_prediction": 0, "caption_answer_index": 1},
        {"window_id": "broken", "caption_prediction": 0, "caption_answer_index": 0},
    ]
    gated_rows = [
        {
            "window_id": "fixed",
            "caption_prediction": 1,
            "caption_answer_index": 1,
            "caption_gate_source": "alternate",
        },
        {
            "window_id": "broken",
            "caption_prediction": 1,
            "caption_answer_index": 0,
            "caption_gate_source": "alternate",
        },
    ]

    report = analyze_failure_taxonomy(primary_rows, gated_rows, benchmark_records=benchmark)

    assert report["summary"]["corrected_count"] == 1
    assert report["summary"]["regressed_count"] == 1
    assert report["corrected_by_primary_wrong_fact"]["dominant_axis"] == 1
    assert report["regressed_by_gated_wrong_fact"]["periodicity"] == 1
    assert report["examples"]["corrected"][0]["window_id"] == "fixed"

