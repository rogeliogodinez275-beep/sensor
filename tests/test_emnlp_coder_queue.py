from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_coder_queue.sh")


def test_coder_queue_downloads_model_and_runs_direct_and_structured_variants():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_coder_status.tsv" in text
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in text
    assert "download_hf_snapshot.py" in text
    assert "model.safetensors.index.json" in text
    assert "model-00004-of-00004.safetensors" in text
    assert "--max-attempts" in text
    assert 'run_download || exit "$FAILURES"' in text
    assert "run_qwen_llm_eval.py" in text
    assert "run_structured_verifier_eval.py" in text
    assert "run_direct ucihar data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" in text
    assert "run_direct wisdm data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" in text
    assert "run_direct mhealth data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" in text
    assert "run_structured ucihar data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" in text
    assert "run_structured wisdm data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" in text
    assert "run_structured mhealth data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" in text
