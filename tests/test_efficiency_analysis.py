import json

from scripts.analyze_efficiency import build_efficiency_markdown, collect_efficiency_rows


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_efficiency_rows_uses_gate_coverage(tmp_path):
    write_json(
        tmp_path / "coder_gated_vote5_choice_logprob_ucihar_hard_v3_constrained_margin2_metrics.json",
        {"caption_gate_alternate_count": 50, "n_eval_records": 100},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"n_scoring_prompts": 100},
    )

    rows = collect_efficiency_rows(tmp_path)

    assert rows[0]["dataset"] == "UCI HAR"
    assert rows[0]["gate_coverage"] == 0.5
    assert rows[0]["full_choice_prompts"] == 100
    assert rows[0]["effective_override_prompts"] == 50


def test_build_efficiency_markdown_reports_cost_interpretation():
    rows = [
        {
            "dataset": "UCI HAR",
            "n_eval_records": 100,
            "full_choice_prompts": 100,
            "effective_override_prompts": 50,
            "gate_coverage": 0.5,
        }
    ]

    md = build_efficiency_markdown(rows)

    assert "# Efficiency Analysis" in md
    assert "UCI HAR" in md
    assert "override coverage" in md

