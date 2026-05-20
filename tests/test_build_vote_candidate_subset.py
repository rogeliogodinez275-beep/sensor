from scripts.build_vote_candidate_subset import build_vote_candidate_subset


def test_build_vote_candidate_subset_preserves_vote_order_and_remaps_answer():
    base = [
        {
            "window_id": "w1",
            "caption_selection": {
                "candidates": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                "answer_index": 2,
            },
            "candidate_index_map": [4, 7, 9],
        }
    ]
    vote_runs = [
        [{"window_id": "w1", "caption_prediction": 1, "candidate_index_map": [4, 7, 9]}],
        [{"window_id": "w1", "caption_prediction": 0, "candidate_index_map": [9, 4, 7]}],
        [{"window_id": "w1", "caption_prediction": 1, "candidate_index_map": [4, 7, 9]}],
    ]

    rows = build_vote_candidate_subset(base, vote_runs)

    assert rows[0]["candidate_index_map"] == [7, 9]
    assert rows[0]["caption_selection"]["answer_index"] == 1
    assert [item["text"] for item in rows[0]["caption_selection"]["candidates"]] == ["b", "c"]
    assert rows[0]["vote_candidate_subset"]["gold_in_vote_candidates"] is True
