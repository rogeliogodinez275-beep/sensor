import json

from scripts.build_caption_order_benchmark import build_caption_order_rows


def test_caption_order_shuffle_preserves_answer_target():
    row = {
        "window_id": "w1",
        "caption_selection": {
            "answer_index": 1,
            "candidates": [
                {"text": "wrong idle"},
                {"text": "correct walking"},
                {"text": "wrong running"},
            ],
        },
    }

    [shuffled] = build_caption_order_rows([row], seed=5151)
    candidates = shuffled["caption_selection"]["candidates"]
    answer_index = shuffled["caption_selection"]["answer_index"]

    assert [item["text"] for item in candidates] != [
        "wrong idle",
        "correct walking",
        "wrong running",
    ]
    assert candidates[answer_index]["text"] == "correct walking"
    assert sorted(json.dumps(item, sort_keys=True) for item in candidates) == sorted(
        json.dumps(item, sort_keys=True) for item in row["caption_selection"]["candidates"]
    )
    assert shuffled["caption_order_seed"] == 5151
