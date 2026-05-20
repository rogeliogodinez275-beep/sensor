from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_support_ablation_queue.sh")
REPORT = Path("scripts/make_emnlp_experiment_report.py")
LLM_CLI = Path("scripts/run_qwen_llm_eval.py")


def test_qwen_llm_cli_exposes_support_ablation_controls():
    text = LLM_CLI.read_text(encoding="utf-8")

    assert "--support-seed" in text
    assert "--support-negative-count" in text


def test_support_ablation_queue_runs_order_and_balance_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_support_ablation_status.tsv" in text
    assert "wait_for_llm_slot" in text
    assert "qwen_llm_mhealth_hard_v3_support_order_seed7001_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_support_order_seed7002_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_support_balanced_neg2_metrics.json" in text
    assert "--support-seed" in text
    assert "--support-negative-count" in text
    assert 'exit "$FAILURES"' in text


def test_report_includes_support_ablation_robustness():
    text = REPORT.read_text(encoding="utf-8")

    assert "Support-Statement Ablations" in text
    assert "MHEALTH hard v3 support order robustness" in text
    assert "MHEALTH hard v3 support balance controls" in text
    assert "qwen_llm_mhealth_hard_v3_support_order_seed*_metrics.json" in text
    assert "qwen_llm_mhealth_hard_v3_support_balanced_neg*_metrics.json" in text
