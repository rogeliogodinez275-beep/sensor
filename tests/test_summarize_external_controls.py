import json

from scripts.summarize_external_controls import collect_rows, write_markdown


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_rows_compares_controls_to_visible(tmp_path):
    write_json(
        tmp_path / "qwen3_4b_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"caption_selection_accuracy": 0.80, "n_eval_records": 10},
    )
    write_json(
        tmp_path / "qwen3_4b_choice_logprob_ucihar_hard_v3_constrained_hidden_evidence_metrics.json",
        {"caption_selection_accuracy": 0.70, "n_eval_records": 10},
    )
    write_json(
        tmp_path / "qwen3_4b_gated_vote5_choice_logprob_ucihar_hard_v3_constrained_margin2p0_metrics.json",
        {"caption_selection_accuracy": 0.82},
    )

    rows = collect_rows(tmp_path, "qwen3_4b")

    visible = next(row for row in rows if row["dataset"] == "ucihar" and row["condition"] == "visible")
    hidden = next(row for row in rows if row["dataset"] == "ucihar" and row["condition"] == "hidden")
    assert visible["choice_delta_vs_visible"] == 0.0
    assert hidden["choice_delta_vs_visible"] == -0.10000000000000009
    assert visible["gated_accuracy"] == 0.82


def test_external_controls_markdown_flags_bounded_grounding_claim(tmp_path):
    payload = {
        "model_tag": "qwen3_4b",
        "rows": [
            {
                "dataset": "ucihar",
                "condition": "visible",
                "choice_accuracy": 0.8,
                "gated_accuracy": 0.82,
                "nogate_accuracy": 0.78,
                "choice_delta_vs_visible": 0.0,
                "n_eval_records": 10,
            }
        ],
    }

    output = tmp_path / "controls.md"
    write_markdown(payload, output)

    md = output.read_text(encoding="utf-8")
    assert "# External Evidence-Control Summary" in md
    assert "bounded grounding claim" in md
