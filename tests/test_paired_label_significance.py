from scripts.paired_label_significance import compare_paired_rows, mcnemar_midp


def test_mcnemar_midp_uses_only_discordant_pairs():
    result = mcnemar_midp(primary_only_correct=1, challenger_only_correct=4)

    assert result["primary_only_correct"] == 1
    assert result["challenger_only_correct"] == 4
    assert result["discordant_count"] == 5
    assert 0.0 <= result["midp_value"] <= 1.0


def test_compare_paired_rows_counts_corrections_and_regressions():
    primary = [
        {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 1},
        {"window_id": "w2", "caption_prediction": 0, "caption_answer_index": 0},
        {"window_id": "w3", "caption_prediction": 2, "caption_answer_index": 2},
    ]
    challenger = [
        {"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 1},
        {"window_id": "w2", "caption_prediction": 1, "caption_answer_index": 0},
        {"window_id": "w3", "caption_prediction": 2, "caption_answer_index": 2},
    ]

    result = compare_paired_rows(primary, challenger)

    assert result["n_pairs"] == 3
    assert result["challenger_accuracy"] == 2 / 3
    assert result["primary_accuracy"] == 2 / 3
    assert result["challenger_only_correct"] == 1
    assert result["primary_only_correct"] == 1
    assert result["delta"] == 0.0

