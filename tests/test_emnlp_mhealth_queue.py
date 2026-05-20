from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_mhealth_queue.sh")


def test_mhealth_queue_builds_dataset_and_runs_qwen_sample():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "build_mhealth_benchmark.py" in text
    assert "mhealth_sensorfact_hard_v3_test.jsonl" in text
    assert "mhealth_hard_v3_sample_1024_seed2026.jsonl" in text
    assert "run_qwen_llm_eval.py" in text
    assert "qwen_llm_mhealth_hard_v3_sample1024_metrics.json" in text
    assert "supervised_mhealth_hard_v3_metrics.json" in text
    assert "supervised_numeric_mhealth_hard_v3_metrics.json" in text
    assert "emnlp_mhealth_status.tsv" in text
    assert 'exit "$FAILURES"' in text
