from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_axis_ledger_queue.sh")


def test_axis_ledger_queue_runs_new_prompt_style_only():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "axis_ledger_json" in text
    assert "run_qwen_llm_eval.py" in text
    assert "--caption-only" in text
    assert "constrained_prompt_axis_ledger_json_caption_only" in text
    assert 'run_caption_only "$dataset" fewshot_json' not in text
    assert 'run_caption_only "$dataset" contrastive_elim_json' not in text


def test_axis_ledger_queue_evaluates_all_datasets():
    text = SCRIPT.read_text(encoding="utf-8")
    for dataset in ("ucihar", "wisdm", "mhealth"):
        assert dataset in text
    assert "run_hybrid_verifier_eval.py" in text
