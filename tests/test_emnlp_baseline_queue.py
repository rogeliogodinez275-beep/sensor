from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_baseline_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")


def test_baseline_queue_runs_embedding_and_hard_v2_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_qwen_embedding_eval.py" in text
    assert "run_qwen_llm_eval.py" in text
    assert "hard_v2_prompt" in text
    assert "hard_v2_stress" in text
    assert "STEP_TIMEOUT_LLM" in text
    assert 'exit "$FAILURES"' in text


def test_report_includes_baseline_and_negative_controls_section():
    text = REPORT.read_text(encoding="utf-8")

    assert "Baseline And Negative Controls" in text
    assert "qwen_embedding_ucihar_hard_v3_*_full_metrics.json" in text
    assert "qwen_llm_ucihar_hard_v2_prompt_*_metrics.json" in text
    assert "qwen_llm_ucihar_hard_v2_*_full_metrics.json" in text
