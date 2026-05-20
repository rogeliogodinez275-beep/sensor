from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_coder_prompt_queue.sh")


def test_coder_prompt_queue_runs_prompt_order_and_support_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_coder_prompt_status.tsv" in text
    assert "Qwen_Qwen2.5-Coder-7B-Instruct" in text
    assert "coder_llm_ucihar_hard_v3_prompt_terse" in text
    assert 'coder_llm_${dataset_name}_hard_v3_caption_order_seed5153' in text
    assert "support_order_seed7101" in text
    assert "run_qwen_llm_eval.py" in text
    assert "build_caption_order_benchmark.py" in text
