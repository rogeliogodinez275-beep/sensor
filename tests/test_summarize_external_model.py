import json

from scripts.summarize_external_model import summarize, write_markdown


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_summarize_external_model_reads_metrics_and_significance(tmp_path):
    write_json(
        tmp_path / "qwen3_4b_choice_logprob_ucihar_hard_v3_constrained_full_metrics.json",
        {"caption_selection_accuracy": 0.72, "n_eval_records": 10},
    )
    write_json(
        tmp_path
        / "qwen3_4b_gated_vote5_choice_logprob_ucihar_hard_v3_constrained_margin2p0_metrics.json",
        {
            "caption_selection_accuracy": 0.76,
            "caption_gate_alternate_count": 3,
            "n_eval_records": 10,
        },
    )
    write_json(
        tmp_path / "qwen3_4b_nogate_vote5_choice_logprob_ucihar_hard_v3_constrained_metrics.json",
        {"caption_selection_accuracy": 0.70},
    )
    write_json(
        tmp_path / "paired_label_significance_ucihar_qwen3_4b_vote5_vs_gated_choice_logprob.json",
        {"delta": 0.02, "mcnemar": {"midp_value": 0.01}},
    )

    payload = summarize(tmp_path, "qwen3_4b")

    row = payload["rows"][0]
    assert row["dataset"] == "ucihar"
    assert row["choice_accuracy"] == 0.72
    assert row["gated_accuracy"] == 0.76
    assert row["nogate_accuracy"] == 0.70
    assert row["delta_vs_vote5"] == 0.02
    assert row["mcnemar_midp"] == 0.01
    assert row["gate_alternate_count"] == 3


def test_write_markdown_includes_interpretation(tmp_path):
    payload = {
        "model_tag": "qwen3_4b",
        "rows": [
            {
                "dataset": "ucihar",
                "choice_accuracy": 0.72,
                "gated_accuracy": 0.76,
                "nogate_accuracy": 0.70,
                "delta_vs_vote5": 0.02,
                "mcnemar_midp": 0.01,
                "gate_alternate_count": 3,
                "n_eval_records": 10,
            }
        ],
    }

    output = tmp_path / "external.md"
    write_markdown(payload, output)

    md = output.read_text(encoding="utf-8")
    assert "# External Cross-Model Reranker Summary" in md
    assert "qwen3_4b" in md
    assert "structured verifier" in md
