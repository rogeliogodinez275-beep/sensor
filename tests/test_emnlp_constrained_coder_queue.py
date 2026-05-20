from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_coder_queue.sh")


def test_constrained_coder_queue_builds_subset_runs_direct_and_hybrid():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "build_constrained_caption_subset.py" in text
    assert "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" in text
    assert 'coder_llm_${dataset_name}_hard_v3_constrained_prompt_fewshot_json' in text
    assert 'hybrid_regex_coder_${dataset_name}_hard_v3_constrained_prompt_fewshot_json' in text
    assert "run_hybrid_verifier_eval.py" in text
