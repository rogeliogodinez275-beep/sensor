from pathlib import Path


SCRIPT = Path("scripts/relay_after_axisfix.sh")


def test_relay_waits_for_axisfix_then_runs_hybrid_and_coder_prompt():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "run_emnlp_axisfix_queue.sh" in text
    assert "run_structured_verifier_eval.py" in text
    assert "run_emnlp_hybrid_queue.sh" in text
    assert "run_emnlp_coder_prompt_queue.sh" in text
    assert "relay_after_axisfix.log" in text
    assert "LOCK_PATH" in text
    assert "relay_start_epoch" in text
    assert "stat -c %Y" in text
    assert "fresh axisfix status file" in text
    assert 'WAIT_SECONDS="${WAIT_SECONDS:-60}"' in text
    assert 'sleep "$WAIT_SECONDS"' in text
