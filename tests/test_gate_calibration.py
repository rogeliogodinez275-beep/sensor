from scripts.calibrate_gate_threshold import (
    choose_threshold,
    split_rows_by_window,
    sweep_thresholds,
)


PRIMARY_ROWS = [
    {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 1},
    {"window_id": "w2", "caption_prediction": 0, "caption_answer_index": 0},
    {"window_id": "w3", "caption_prediction": 2, "caption_answer_index": 2},
]

ALTERNATE_ROWS = [
    {"window_id": "w1", "caption_prediction": 1, "caption_scores": [0.0, 3.0]},
    {"window_id": "w2", "caption_prediction": 1, "caption_scores": [0.0, 1.0]},
    {"window_id": "w3", "caption_prediction": 1, "caption_scores": [0.0, 0.0, 0.0]},
]


def test_sweep_thresholds_reports_coverage_accuracy_and_delta():
    rows = sweep_thresholds(PRIMARY_ROWS, ALTERNATE_ROWS, thresholds=[0.0, 2.0, 5.0])

    assert rows[0]["threshold"] == 0.0
    assert rows[0]["alternate_count"] == 2
    assert rows[0]["coverage"] == 2 / 3
    assert rows[0]["caption_selection_accuracy"] == 2 / 3
    assert rows[1]["caption_selection_accuracy"] == 1.0
    assert rows[2]["alternate_count"] == 0
    assert rows[2]["delta_vs_primary"] == 0.0


def test_choose_threshold_maximizes_accuracy_then_prefers_conservative_coverage():
    swept = sweep_thresholds(PRIMARY_ROWS, ALTERNATE_ROWS, thresholds=[0.0, 2.0])

    best = choose_threshold(swept)

    assert best["threshold"] == 2.0
    assert best["caption_selection_accuracy"] == 1.0
    assert best["alternate_count"] == 1


def test_split_rows_by_window_creates_disjoint_dev_and_eval_sets():
    dev_primary, dev_alt, eval_primary, eval_alt = split_rows_by_window(
        PRIMARY_ROWS,
        ALTERNATE_ROWS,
        dev_modulus=2,
        dev_remainders={0},
    )

    dev_ids = {row["window_id"] for row in dev_primary}
    eval_ids = {row["window_id"] for row in eval_primary}
    assert dev_ids
    assert eval_ids
    assert dev_ids.isdisjoint(eval_ids)
    assert [row["window_id"] for row in dev_primary] == [row["window_id"] for row in dev_alt]
    assert [row["window_id"] for row in eval_primary] == [row["window_id"] for row in eval_alt]
