from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_order_prompt_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")


def test_order_prompt_queue_runs_hard_v3_prompt_styles_on_order_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_qwen_llm_eval.py" in text
    assert "ucihar_hard_v3_caption_order_seed" in text
    assert "wisdm_hard_v3_caption_order_seed" in text
    assert "terse" in text
    assert "chain_then_json" in text
    assert "emnlp_order_prompt_status.tsv" in text


def test_report_separates_plain_order_from_prompt_order_metrics():
    text = REPORT.read_text(encoding="utf-8")

    assert "Candidate Order Prompt Robustness" in text
    assert "caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json" in text
    assert "caption_order_seed*_prompt_*_metrics.json" in text
