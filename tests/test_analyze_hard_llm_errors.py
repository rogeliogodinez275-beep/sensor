from scripts.analyze_hard_llm_errors import summarize_errors


def test_summarize_errors_uses_support_items_prompt_order():
    records = [
        {
            "counterfactuals": [
                {"changed_fact": "intensity"},
                {"changed_fact": "periodicity"},
            ]
        }
    ]
    rows = [
        {
            "support_predictions": [False, True, False],
            "support_items": [
                {"kind": "counterfactual", "changed_fact": "intensity"},
                {"kind": "positive"},
                {"kind": "counterfactual", "changed_fact": "periodicity"},
            ],
        }
    ]

    summary = summarize_errors(records, rows)

    assert summary["positive_total"] == 1
    assert summary["positive_supported"] == 1
    assert summary["field_stats"]["intensity"] == [1, 1]
    assert summary["field_stats"]["periodicity"] == [1, 1]
