from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_relay_queue.sh")


def test_relay_queue_waits_for_active_llm_before_starting_next_job():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "wait_for_llm_slot" in text
    assert "pgrep -af \"scripts/run_qwen_llm_eval.py\"" in text
    assert "sleep \"$WAIT_SECONDS\"" in text


def test_relay_queue_runs_stress_controls_and_full_prompt_extensions():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ucihar_sensorfact_hard_v3_test.jsonl" in text
    assert "wisdm_sensorfact_hard_v3_test.jsonl" in text
    assert "numeric_mask" in text
    assert "shuffled_evidence" in text
    assert "hidden_evidence" in text
    assert "STEP_TIMEOUT_LLM" in text
    assert 'exit "$FAILURES"' in text
