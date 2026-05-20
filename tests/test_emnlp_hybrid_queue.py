from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_hybrid_queue.sh")


def test_hybrid_queue_runs_qwen_and_coder_fallbacks_for_three_datasets():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_hybrid_status.tsv" in text
    assert "run_hybrid_verifier_eval.py" in text
    assert "hybrid_regex_qwen_ucihar_hard_v3" in text
    assert "hybrid_regex_coder_wisdm_hard_v3" in text
    assert "hybrid_regex_coder_mhealth_hard_v3" in text
    assert "axisfix_structured_regex_ucihar_hard_v3_rows.jsonl" in text
    assert "axisfix_structured_regex_wisdm_hard_v3_rows.jsonl" in text
    assert "axisfix_structured_regex_mhealth_hard_v3_rows.jsonl" in text
    assert "test_hybrid_verifier_eval.py" in text
