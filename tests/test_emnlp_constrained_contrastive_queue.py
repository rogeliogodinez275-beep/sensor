from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_contrastive_queue.sh")


def test_constrained_contrastive_queue_runs_new_prompt_style():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "contrastive_elim_json" in text
    assert "run_qwen_llm_eval.py" in text
    assert "--caption-only" in text
    assert "constrained_prompt_contrastive_elim_json_caption_only" in text


def test_constrained_contrastive_queue_evaluates_all_datasets():
    text = SCRIPT.read_text(encoding="utf-8")
    for dataset in ("ucihar", "wisdm", "mhealth"):
        assert dataset in text
    assert "run_hybrid_verifier_eval.py" in text
