from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_order_vote5_queue.sh")


def test_constrained_order_vote5_queue_extends_existing_order_vote_runs():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "build_caption_order_benchmark.py" in text
    assert "seed5161" in text
    assert "seed5167" in text
    assert "aggregate_caption_votes.py" in text
    assert "constrained_caption_order_vote5_prompt_fewshot_json_caption_only" in text
    assert "run_hybrid_verifier_eval.py" in text
