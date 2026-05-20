import json

from scripts.summarize_position_prior import build_position_prior_markdown, collect_position_prior_rows


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_position_prior_rows_reads_metrics(tmp_path):
    write_json(
        tmp_path / "coder_position_prior_ucihar_hard_v3_constrained_metrics.json",
        {"caption_selection_accuracy": 0.48, "n_eval_records": 773},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"caption_selection_accuracy": 0.76},
    )

    rows = collect_position_prior_rows(tmp_path)

    assert rows[0]["dataset"] == "UCI HAR"
    assert rows[0]["position_prior_acc"] == 0.48
    assert rows[0]["choice_acc"] == 0.76
    assert rows[0]["choice_minus_position"] == 0.28


def test_collect_position_prior_rows_supports_custom_prefixes(tmp_path):
    write_json(
        tmp_path / "qwen3_4b_position_prior_ucihar_hard_v3_constrained_metrics.json",
        {"caption_selection_accuracy": 0.41, "n_eval_records": 773},
    )
    write_json(
        tmp_path / "qwen3_4b_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"caption_selection_accuracy": 0.68},
    )

    rows = collect_position_prior_rows(
        tmp_path,
        position_prefix="qwen3_4b_position_prior",
        choice_prefix="qwen3_4b_choice_logprob",
    )

    assert rows[0]["position_prior_acc"] == 0.41
    assert rows[0]["choice_acc"] == 0.68


def test_build_position_prior_markdown_flags_high_position_prior():
    rows = [
        {
            "dataset": "UCI HAR",
            "position_prior_acc": 0.48,
            "choice_acc": 0.76,
            "choice_minus_position": 0.28,
            "n_eval_records": 773,
        }
    ]

    md = build_position_prior_markdown(rows)

    assert "# Position-Prior Baseline Summary" in md
    assert "UCI HAR" in md
    assert "位置先验" in md
