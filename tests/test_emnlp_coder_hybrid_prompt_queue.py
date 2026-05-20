from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_coder_hybrid_prompt_queue.sh")


def test_coder_hybrid_prompt_queue_runs_chain_prompt_and_hybrid_prompt_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_coder_hybrid_prompt_status.tsv" in text
    assert "Qwen_Qwen2.5-Coder-7B-Instruct" in text
    assert "hybrid_regex_coder_ucihar_hard_v3_prompt_terse" in text
    assert "coder_llm_ucihar_hard_v3_prompt_chain_then_json" in text
    assert "hybrid_regex_coder_wisdm_hard_v3_prompt_chain_then_json" in text
    assert "hybrid_regex_coder_mhealth_hard_v3_prompt_chain_then_json" in text
    assert "run_qwen_llm_eval.py" in text
    assert "run_hybrid_verifier_eval.py" in text

