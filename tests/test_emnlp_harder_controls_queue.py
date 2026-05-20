from pathlib import Path


def test_harder_controls_queue_runs_new_control_modes_on_constrained_subsets():
    text = Path("scripts/run_emnlp_harder_controls_queue.sh").read_text(encoding="utf-8")

    assert "numeric-swap" in text
    assert "axis-permutation" in text
    assert "trend-flip" in text
    assert "outputs/constrained_${dataset_name}_hard_v3_subset.jsonl" in text
    assert "coder_gated_vote5_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" in text
    assert "run_logprob_reranker.py" in text
