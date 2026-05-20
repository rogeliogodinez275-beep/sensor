from sensorfact.audit import (
    answer_position_counts,
    forbidden_shortcut_hits,
    subject_overlap,
)


def test_answer_position_counts_counts_each_supported_candidate_position():
    records = [
        {"caption_selection": {"candidates": [{"supported": True}, {"supported": False}]}},
        {"caption_selection": {"candidates": [{"supported": False}, {"supported": True}]}},
        {"caption_selection": {"candidates": [{"supported": True}, {"supported": False}]}},
    ]

    assert answer_position_counts(records) == {0: 2, 1: 1}


def test_forbidden_shortcut_hits_detects_labeling_words():
    records = [
        {"positive": {"text": "A supported description is calm."}, "counterfactuals": []},
        {"positive": {"text": "The motion is calm."}, "counterfactuals": [{"text": "This claims spikes."}]},
    ]

    hits = forbidden_shortcut_hits(records, ["supported description", "claims"])

    assert hits == {"supported description": 1, "claims": 1}


def test_subject_overlap_reports_train_test_leakage():
    train = [{"subject_id": "1"}, {"subject_id": "2"}]
    test = [{"subject_id": "2"}, {"subject_id": "3"}]

    assert subject_overlap(train, test) == {"2"}
