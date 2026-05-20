from pathlib import Path


SCRIPT = Path("scripts/relay_after_coder_fewshot.sh")


def test_relay_waits_for_coder_fewshot_then_runs_fewshot_stress_queue():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_emnlp_coder_fewshot_queue.sh" in text
    assert "run_qwen_llm_eval.py" in text
    assert "run_emnlp_coder_fewshot_stress_queue.sh" in text
    assert "relay_after_coder_fewshot.log" in text
