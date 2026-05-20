from pathlib import Path


SCRIPT = Path("scripts/relay_after_coder_fewshot_stress.sh")


def test_relay_waits_for_coder_fewshot_stress_then_runs_qwen_fewshot_queue():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_emnlp_coder_fewshot_stress_queue.sh" in text
    assert "run_qwen_llm_eval.py" in text
    assert "run_emnlp_qwen_fewshot_queue.sh" in text
    assert "relay_after_coder_fewshot_stress.log" in text
    assert "LOCK_PATH" in text
    assert "relay_start_epoch" in text
    assert "stat -c %Y" in text
    assert "fresh coder fewshot stress status file" in text
