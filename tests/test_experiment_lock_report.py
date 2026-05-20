import json

from scripts.lock_evidence_first_report import build_lock_report


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_build_lock_report_summarizes_main_results_and_risks(tmp_path):
    write_json(
        tmp_path / "hybrid_regex_coder_gated_vote5_choice_logprob_ucihar_hard_v3_constrained_margin2_metrics.json",
        {"caption_selection_accuracy": 0.8629, "caption_gate_alternate_count": 10},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_hidden_evidence_full_metrics.json",
        {"caption_selection_accuracy": 0.7835},
    )
    write_json(
        tmp_path / "coder_choice_logprob_ucihar_hard_v3_full_baseline_metrics.json",
        {"caption_selection_accuracy": 0.6725},
    )
    bootstrap = {
        "ucihar": {"delta": 0.00645, "ci_low": 0.00373, "ci_high": 0.00984},
        "wisdm": {"delta": 0.00129, "ci_low": -0.00129, "ci_high": 0.00388},
        "mhealth": {"delta": 0.00573, "ci_low": 0.00191, "ci_high": 0.01051},
    }
    write_json(tmp_path / "confidence_gated_forced_choice_paired_bootstrap.json", bootstrap)

    report = build_lock_report(tmp_path)

    assert "两显著一非显著" in report
    assert "UCI HAR" in report
    assert "0.8629" in report
    assert "hidden-evidence" in report
    assert "CF F1 不归功于 reranker" in report
