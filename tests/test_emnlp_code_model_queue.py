from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_code_model_queue.sh")


def test_code_model_queue_runs_structured_verifier_across_three_datasets():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_code_model_status.tsv" in text
    assert "run_structured_verifier_eval.py" in text
    assert "structured_regex_ucihar_hard_v3" in text
    assert "structured_model_ucihar_hard_v3" in text
    assert "structured_regex_wisdm_hard_v3" in text
    assert "structured_model_wisdm_hard_v3" in text
    assert "structured_regex_mhealth_hard_v3" in text
    assert "structured_model_mhealth_hard_v3" in text
    assert "qwen_structured_regex_ucihar_hard_v3_metrics.json" in text
    assert "qwen_structured_model_mhealth_hard_v3_metrics.json" in text
