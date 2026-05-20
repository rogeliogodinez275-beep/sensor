from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_constrained_order_vote7_queue.sh")


def test_constrained_order_vote7_queue_adds_two_new_order_seeds():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "seed5171" in text
    assert "seed5179" in text
    assert "constrained_caption_order_vote7_prompt_fewshot_json_caption_only" in text


def test_constrained_order_vote7_queue_aggregates_seven_vote_files():
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("prompt_fewshot_json_caption_only_rows.jsonl") >= 7
    assert "caption_order_seed5171_prompt_fewshot_json_caption_only_rows.jsonl" in text
    assert "caption_order_seed5179_prompt_fewshot_json_caption_only_rows.jsonl" in text
