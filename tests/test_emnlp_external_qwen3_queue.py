from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_external_qwen3_queue.sh")


def test_external_qwen3_queue_runs_cross_model_reranker_for_all_datasets():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Qwen_Qwen3-4B-Instruct-2507" in text
    assert "--mode choice" in text
    assert "for dataset in ucihar wisdm mhealth" in text
    assert "${MODEL_TAG}_choice_logprob_${dataset_name}_hard_v3_constrained_full" in text
    assert "paired_label_significance_${dataset_name}_${MODEL_TAG}_gated" in text
    assert "scripts/summarize_external_model.py" in text
