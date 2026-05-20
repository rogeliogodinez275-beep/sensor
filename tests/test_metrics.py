from sensorfact.metrics import (
    caption_selection_accuracy,
    counterfactual_rejection_metrics,
)


def test_caption_selection_accuracy_uses_highest_score_candidate():
    examples = [
        {"answer_index": 1, "scores": [0.2, 0.8, 0.1]},
        {"answer_index": 0, "scores": [0.4, 0.7, 0.1]},
    ]

    assert caption_selection_accuracy(examples) == 0.5


def test_counterfactual_rejection_metrics_report_binary_scores():
    metrics = counterfactual_rejection_metrics(
        y_true=[1, 1, 0, 0],
        y_score=[0.9, 0.7, 0.8, 0.2],
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert round(metrics["precision"], 3) == 0.667
    assert metrics["recall"] == 1.0
    assert round(metrics["f1"], 3) == 0.8
