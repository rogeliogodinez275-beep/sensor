from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_coder_fewshot_stress_queue.sh")


def test_coder_fewshot_stress_queue_runs_order_and_support_controls():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_coder_fewshot_stress_status.tsv" in text
    assert "build_caption_order_benchmark.py" in text
    assert "coder_llm_ucihar_hard_v3_caption_order_seed5153_prompt_fewshot_json" in text
    assert "coder_llm_wisdm_hard_v3_caption_order_seed5153_prompt_fewshot_json" in text
    assert "coder_llm_mhealth_hard_v3_caption_order_seed5153_prompt_fewshot_json" in text
    assert "coder_llm_ucihar_hard_v3_support_order_seed7101_prompt_fewshot_json" in text
    assert "coder_llm_wisdm_hard_v3_support_order_seed7101_prompt_fewshot_json" in text
    assert "coder_llm_mhealth_hard_v3_support_order_seed7101_prompt_fewshot_json" in text
    assert "coder_llm_ucihar_hard_v3_support_balanced_neg3_prompt_fewshot_json" in text
    assert "coder_llm_wisdm_hard_v3_support_balanced_neg3_prompt_fewshot_json" in text
    assert "coder_llm_mhealth_hard_v3_support_balanced_neg3_prompt_fewshot_json" in text
    assert "run_qwen_llm_eval.py" in text

