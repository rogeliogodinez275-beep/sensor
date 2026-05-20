from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_qwen_fewshot_queue.sh")


def test_qwen_fewshot_queue_runs_full_hard_v3_fewshot_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_qwen_fewshot_status.tsv" in text
    assert "qwen_llm_ucihar_hard_v3_prompt_fewshot_json" in text
    assert "qwen_llm_wisdm_hard_v3_prompt_fewshot_json" in text
    assert "qwen_llm_mhealth_hard_v3_prompt_fewshot_json" in text
    assert "--prompt-style fewshot_json" in text
    assert "run_qwen_llm_eval.py" in text
    assert 'exit "$FAILURES"' in text
