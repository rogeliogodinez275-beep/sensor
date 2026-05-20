from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_order_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")


def test_order_queue_runs_hard_v2_and_hard_v3_candidate_order_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "build_caption_order_benchmark.py" in text
    assert "ucihar_sensorfact_hard_v3_test.jsonl" in text
    assert "wisdm_sensorfact_hard_v3_test.jsonl" in text
    assert "ucihar_sensorfact_hard_v2_test.jsonl" in text
    assert "wisdm_sensorfact_hard_v2_test.jsonl" in text
    assert "run_qwen_llm_eval.py" in text
    assert "caption_order_seed" in text


def test_report_includes_candidate_order_robustness_section():
    text = REPORT.read_text(encoding="utf-8")

    assert "Candidate Order Robustness" in text
    assert "qwen_llm_ucihar_hard_v3_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json" in text
    assert "qwen_llm_wisdm_hard_v2_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json" in text
