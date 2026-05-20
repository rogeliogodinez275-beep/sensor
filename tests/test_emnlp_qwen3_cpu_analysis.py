from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_qwen3_cpu_analysis.sh")


def test_qwen3_cpu_analysis_calibrates_and_updates_balanced_subset():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "dev_threshold_heldout_${dataset}_qwen3_choice_logprob.json" in text
    assert "qwen3_choice_logprob_margin_curve_${dataset}.json" in text
    assert "qwen3_choice=outputs/qwen3_4b_choice_logprob_${dataset}_hard_v3_constrained_full_rows.jsonl" in text
    assert "qwen3_gate_calibration_summary_2026-05-20.md" in text
