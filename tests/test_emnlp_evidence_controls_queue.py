from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_evidence_controls_queue.sh")


def test_evidence_controls_queue_runs_required_p1_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "build_evidence_control_benchmark.py" in text
    assert "shuffled" in text
    assert "numeric-mask" in text
    assert "hidden" in text
    assert "run_logprob_reranker.py" in text
    assert "--mode choice" in text
    assert "gate_caption_rows.py" in text
    assert "nogate_vote5_choice_logprob" in text
    assert "--min-alternate-margin -1000000" in text
    assert "--dev-modulus 5" in text
    assert "analyze_margin_curve.py" in text
    assert "analyze_failure_taxonomy.py" in text
    assert "paired_label_significance.py" in text
    assert "dev_threshold_heldout_" in text
    assert "summarize_evidence_controls.py" in text
    assert "lock_evidence_first_report.py" in text
