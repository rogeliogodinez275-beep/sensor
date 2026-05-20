import json

from scripts.analyze_balanced_candidate_subset import (
    build_summary,
    candidate_balance_features,
    select_balanced_features,
)


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def benchmark_row(window_id, candidates, answer_index=0):
    return {
        "window_id": window_id,
        "evidence_text": "moderate energy repeatable rhythm slow cadence",
        "caption_selection": {
            "answer_index": answer_index,
            "candidates": [{"text": text} for text in candidates],
        },
    }


def test_candidate_balance_features_detects_length_and_overlap_range():
    row = benchmark_row(
        "w1",
        [
            "moderate energy slow cadence",
            "moderate energy repeatable rhythm slow cadence",
            "a very long candidate with unrelated wording and several extra descriptors",
        ],
    )

    features = candidate_balance_features(row)

    assert features["window_id"] == "w1"
    assert features["length_range"] > 0
    assert features["evidence_overlap_range"] > 0


def test_select_balanced_features_keeps_lowest_balance_score():
    features = [
        {"window_id": "balanced", "balance_score": 0.1, "length_range": 1, "evidence_overlap_range": 0.1},
        {"window_id": "biased", "balance_score": 2.0, "length_range": 10, "evidence_overlap_range": 0.9},
    ]

    selected = select_balanced_features(
        features,
        target_fraction=0.5,
        max_length_range=None,
        max_overlap_range=None,
    )

    assert [row["window_id"] for row in selected] == ["balanced"]


def test_build_summary_evaluates_row_sets_on_balanced_subset(tmp_path):
    benchmark = tmp_path / "bench.jsonl"
    rows = tmp_path / "rows.jsonl"
    write_jsonl(
        benchmark,
        [
            benchmark_row("w1", ["a b c", "a b d", "a c d"]),
            benchmark_row("w2", ["short", "a much much longer option", "another unrelated long option"]),
        ],
    )
    write_jsonl(
        rows,
        [
            {"window_id": "w1", "caption_prediction": 0, "caption_answer_index": 0},
            {"window_id": "w2", "caption_prediction": 1, "caption_answer_index": 0},
        ],
    )

    summary = build_summary(
        benchmark_path=benchmark,
        row_sets=[("system", rows)],
        target_fraction=0.5,
        max_length_range=None,
        max_overlap_range=None,
    )

    assert summary["selection"]["selected_records"] == 1
    assert summary["systems"][0]["balanced_n"] == 1
    assert summary["systems"][0]["balanced_accuracy"] == 1.0
