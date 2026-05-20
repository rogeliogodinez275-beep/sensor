from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_external_qwen3_controls_queue.sh")


def test_external_qwen3_controls_queue_runs_all_control_modes():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Qwen_Qwen3-4B-Instruct-2507" in text
    assert "for mode in shuffled numeric-mask hidden" in text
    assert "build_evidence_control_benchmark.py" in text
    assert "${MODEL_TAG}_choice_logprob_${dataset_name}_hard_v3_constrained_${tag}" in text
    assert "scripts/summarize_external_controls.py" in text
