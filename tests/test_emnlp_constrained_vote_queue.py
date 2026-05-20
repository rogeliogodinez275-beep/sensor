from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_vote_queue.sh")


def test_constrained_vote_queue_runs_caption_only_and_vote_aggregation():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--caption-only" in text
    assert "aggregate_caption_votes.py" in text
    assert "run_hybrid_verifier_eval.py" in text
    assert "build_constrained_caption_subset.py" in text
