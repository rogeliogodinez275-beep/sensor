from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_structured_prompt_queue.sh")


def test_structured_prompt_queue_runs_three_datasets_and_four_prompt_styles():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "emnlp_structured_prompt_status.tsv" in text
    assert "run_structured_verifier_eval.py" in text
    assert "--parser-mode model_evidence" in text
    assert "--prompt-style" in text
    assert "strict_json" in text
    assert "terse" in text
    assert "chain_then_json" in text
    assert "fewshot_json" in text
    assert "run_structured_prompt ucihar data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl" in text
    assert "run_structured_prompt wisdm data/benchmark/wisdm_sensorfact_hard_v3_test.jsonl" in text
    assert "run_structured_prompt mhealth data/benchmark/mhealth_sensorfact_hard_v3_test.jsonl" in text
