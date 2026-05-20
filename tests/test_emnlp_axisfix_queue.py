from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_axisfix_queue.sh")


def test_axisfix_queue_reruns_regex_qwen_and_coder_structured_models():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_axisfix_status.tsv" in text
    assert "test_structured_verifier_eval.py" in text
    assert "axisfix_structured_regex_ucihar_hard_v3" in text
    assert "axisfix_qwen_structured_model_wisdm_hard_v3" in text
    assert "axisfix_coder_structured_model_mhealth_hard_v3" in text
    assert "QWEN_MODEL_DIR" in text
    assert "CODER_MODEL_DIR" in text
    assert "run_structured_verifier_eval.py" in text
    assert '--parser-mode "$parser_mode"' in text
    assert "model_evidence" in text
    assert "--prompt-style strict_json" in text
