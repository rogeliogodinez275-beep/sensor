from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_cross_support_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")


def test_cross_support_queue_runs_three_dataset_support_ablations_and_mhealth_embedding_stress():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_cross_support_status.tsv" in text
    assert "qwen_embedding_mhealth_hard_v3_numeric_mask_full_metrics.json" in text
    assert "qwen_llm_ucihar_hard_v3_support_order_seed7001_metrics.json" in text
    assert "qwen_llm_wisdm_hard_v3_support_order_seed7001_metrics.json" in text
    assert "qwen_llm_ucihar_hard_v3_support_balanced_neg2_metrics.json" in text
    assert "qwen_llm_wisdm_hard_v3_support_balanced_neg3_metrics.json" in text
    assert "--support-seed" in text
    assert "--support-negative-count" in text
    assert "wait_for_llm_slot" in text
    assert 'exit "$FAILURES"' in text


def test_report_includes_cross_dataset_support_ablation_rows():
    text = REPORT.read_text(encoding="utf-8")

    assert "UCI hard v3 support order robustness" in text
    assert "WISDM hard v3 support order robustness" in text
    assert "UCI hard v3 support balance controls" in text
    assert "WISDM hard v3 support balance controls" in text
    assert "qwen_llm_ucihar_hard_v3_support_order_seed*_metrics.json" in text
    assert "qwen_llm_wisdm_hard_v3_support_balanced_neg*_metrics.json" in text
