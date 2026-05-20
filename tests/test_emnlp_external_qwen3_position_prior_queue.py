from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_external_qwen3_position_prior_queue.sh")


def test_external_qwen3_position_prior_queue_runs_answer_index_baseline():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Qwen_Qwen3-4B-Instruct-2507" in text
    assert "--mode position-prior" in text
    assert "for dataset in ucihar wisdm mhealth" in text
    assert "${MODEL_TAG}_position_prior_${dataset_name}_hard_v3_constrained" in text
    assert "--position-prefix" in text
