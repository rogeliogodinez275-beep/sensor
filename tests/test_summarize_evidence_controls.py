import json

from scripts.summarize_evidence_controls import build_summary_markdown, collect_control_rows


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_control_rows_reads_visible_and_control_metrics(tmp_path):
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"caption_selection_accuracy": 0.7607},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_shuffled_evidence_metrics.json",
        {"caption_selection_accuracy": 0.7309},
    )
    write_json(
        tmp_path / "coder_gated_vote5_choice_logprob_ucihar_hard_v3_constrained_shuffled_evidence_margin2p0_metrics.json",
        {"caption_selection_accuracy": 0.7425, "caption_gate_alternate_count": 88, "n_eval_records": 773},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_numeric_mask_metrics.json",
        {"caption_selection_accuracy": 0.7510},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_hidden_evidence_metrics.json",
        {"caption_selection_accuracy": 0.7001},
    )

    rows = collect_control_rows(tmp_path)

    assert rows[0]["dataset"] == "UCI HAR"
    assert rows[0]["condition"] == "visible"
    assert rows[0]["choice_acc"] == 0.7607
    shuffled = [row for row in rows if row["condition"] == "shuffled-evidence"][0]
    assert shuffled["gated_acc"] == 0.7425
    assert shuffled["gated_coverage"] == 88 / 773


def test_build_summary_markdown_includes_guardrail_when_controls_are_close():
    rows = [
        {"dataset": "UCI HAR", "condition": "visible", "choice_acc": 0.7600, "gated_acc": 0.8629, "gated_coverage": 0.1},
        {"dataset": "UCI HAR", "condition": "shuffled-evidence", "choice_acc": 0.7500, "gated_acc": 0.8500, "gated_coverage": 0.1},
    ]

    md = build_summary_markdown(rows)

    assert "# Evidence-Control Summary" in md
    assert "UCI HAR" in md
    assert "shuffled-evidence" in md
    assert "收紧 grounding claim" in md

