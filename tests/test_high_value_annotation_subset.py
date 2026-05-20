from pathlib import Path

from scripts.build_high_value_annotation_subset import build_annotation_subset


def make_record(idx: int) -> dict:
    return {
        "window_id": f"w{idx}",
        "dataset_id": "toy",
        "label": "walk",
        "evidence_text": f"Evidence {idx}",
        "positive": {"text": f"positive {idx}", "supported": True},
        "caption_selection": {
            "answer_index": 0,
            "candidates": [
                {"text": f"positive {idx}", "supported": True},
                {"text": f"negative {idx}", "supported": False},
            ],
        },
    }


def test_annotation_subset_prefers_disagreement_rows(tmp_path: Path):
    records = [make_record(i) for i in range(5)]
    vote_rows = [
        {"window_id": "w0", "caption_prediction": 0, "caption_answer_index": 0},
        {"window_id": "w1", "caption_prediction": 1, "caption_answer_index": 0},
    ]
    gated_rows = [
        {"window_id": "w0", "caption_prediction": 1, "caption_answer_index": 0},
        {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 0},
    ]

    subset = build_annotation_subset(
        records=records,
        dataset_id="toy",
        target_count=3,
        seed=5,
        vote_rows=vote_rows,
        gated_rows=gated_rows,
    )

    assert len(subset) == 3
    assert subset[0]["window_id"] in {"w0", "w1"}
    assert subset[0]["selection_reason"] == "vote_gated_disagreement"
    assert "human_caption" in subset[0]
    assert "counterfactual_validity" in subset[0]
