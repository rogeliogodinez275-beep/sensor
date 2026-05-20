from pathlib import Path


SCRIPT = Path("scripts/relay_after_coder_prompt.sh")


def test_relay_waits_for_coder_prompt_then_runs_coder_hybrid_prompt_queue():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_emnlp_coder_prompt_queue.sh" in text
    assert "run_qwen_llm_eval.py" in text
    assert "run_emnlp_coder_hybrid_prompt_queue.sh" in text
    assert "relay_after_coder_prompt.log" in text
    assert "LOCK_PATH" in text
    assert "relay_start_epoch" in text
    assert "stat -c %Y" in text
    assert "fresh coder prompt status file" in text
