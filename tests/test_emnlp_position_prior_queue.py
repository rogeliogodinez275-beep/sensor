from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_position_prior_queue.sh")


def test_position_prior_queue_runs_answer_index_baseline_for_all_datasets():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "--mode position-prior" in text
    assert "for dataset in ucihar wisdm mhealth" in text
    assert 'coder_position_prior_${dataset_name}_hard_v3_constrained' in text
    assert "tests/test_logprob_reranker.py" in text
