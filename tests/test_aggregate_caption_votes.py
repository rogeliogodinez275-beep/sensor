from scripts.aggregate_caption_votes import aggregate_caption_rows


def test_aggregate_caption_rows_uses_majority_vote():
    rows_by_run = [
        [
            {"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 1, "candidate_index_map": [0, 1, 2]},
            {"window_id": "w2", "caption_prediction": 0, "caption_answer_index": 2, "candidate_index_map": [0, 1, 2]},
        ],
        [
            {"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 1, "candidate_index_map": [0, 1, 2]},
            {"window_id": "w2", "caption_prediction": 2, "caption_answer_index": 2, "candidate_index_map": [0, 1, 2]},
        ],
        [
            {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 1, "candidate_index_map": [0, 1, 2]},
            {"window_id": "w2", "caption_prediction": 2, "caption_answer_index": 2, "candidate_index_map": [0, 1, 2]},
        ],
    ]

    metrics, rows = aggregate_caption_rows(rows_by_run, system_name="vote")

    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["n_eval_records"] == 2
    assert metrics["vote_size"] == 3
    assert rows[0]["caption_prediction"] == 1
    assert rows[1]["caption_prediction"] == 2


def test_aggregate_caption_rows_breaks_ties_by_priority_order():
    rows_by_run = [
        [{"window_id": "w1", "caption_prediction": 2, "caption_answer_index": 2, "candidate_index_map": [0, 1, 2]}],
        [{"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 2, "candidate_index_map": [0, 1, 2]}],
    ]

    metrics, rows = aggregate_caption_rows(rows_by_run, system_name="vote")

    assert metrics["caption_selection_accuracy"] == 1.0
    assert rows[0]["caption_prediction"] == 2


def test_aggregate_caption_rows_votes_after_candidate_index_remap():
    rows_by_run = [
        [{"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 1, "candidate_index_map": [0, 2, 1]}],
        [{"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 2, "candidate_index_map": [2, 0, 1]}],
        [{"window_id": "w1", "caption_prediction": 2, "caption_answer_index": 0, "candidate_index_map": [1, 0, 2]}],
    ]

    metrics, rows = aggregate_caption_rows(rows_by_run, system_name="vote")

    assert metrics["caption_selection_accuracy"] == 1.0
    assert rows[0]["caption_prediction"] == 2
    assert rows[0]["caption_answer_index"] == 2
    assert rows[0]["candidate_index_map"] is None
