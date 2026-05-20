from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_mhealth_extra_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")


def test_mhealth_extra_queue_runs_full_prompt_stress_and_order_extensions():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "mhealth_sensorfact_hard_v2_test.jsonl" in text
    assert "mhealth_sensorfact_hard_v3_test.jsonl" in text
    assert "qwen_llm_mhealth_hard_v3_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_prompt_terse_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_numeric_mask_full_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_numeric_mask_full_prompt_terse_metrics.json" in text
    assert "mhealth_hard_v3_caption_order_seed" in text
    assert "qwen_llm_mhealth_hard_v3_caption_order_seed5151_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_caption_order_seed5151_prompt_terse_metrics.json" in text
    assert "emnlp_mhealth_extra_status.tsv" in text
    assert 'exit "$FAILURES"' in text


def test_report_includes_mhealth_extensions_in_robustness_tables():
    text = REPORT.read_text(encoding="utf-8")

    assert "MHEALTH hard v3 prompt robustness" in text
    assert "MHEALTH hard v3 full prompt robustness" in text
    assert "MHEALTH hard v3 stress controls" in text
    assert "MHEALTH hard v3 full stress controls" in text
    assert "MHEALTH hard v3 stress prompt robustness" in text
    assert "MHEALTH hard v3 candidate order robustness" in text
    assert "MHEALTH hard v3 candidate order prompt robustness" in text
    assert "qwen_llm_mhealth_hard_v3_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json" in text
