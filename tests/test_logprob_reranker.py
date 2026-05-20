from sensorfact.logprob_reranker import (
    build_candidate_support_prompt,
    build_choice_prompt,
    build_position_prior_prompt,
    evaluate_choice_logprob_reranker,
    evaluate_logprob_reranker,
    evaluate_position_prior_logprob_reranker,
)


class FakeSupportScorer:
    def score_support(self, prompts):
        rows = []
        for prompt in prompts:
            score = 3.0 if "correct caption" in prompt else -2.0
            rows.append(
                {
                    "positive_logprob": score,
                    "negative_logprob": 0.0,
                    "margin": score,
                }
            )
        return rows


class FakeChoiceScorer:
    def score_choices(self, prompts, candidate_counts):
        assert candidate_counts == [2]
        return [
            [
                {"choice": "A", "logprob": -4.0},
                {"choice": "B", "logprob": -0.5},
            ]
        ]


def make_record():
    return {
        "window_id": "w1",
        "evidence_text": "RMS energy is high and the cadence is slow.",
        "candidate_index_map": [2, 5],
        "caption_selection": {
            "answer_index": 1,
            "candidates": [
                {"text": "wrong caption"},
                {"text": "correct caption"},
            ],
        },
    }


def test_build_candidate_support_prompt_asks_binary_entailment_question():
    prompt = build_candidate_support_prompt(make_record(), "correct caption")

    assert "RMS energy is high" in prompt
    assert "correct caption" in prompt
    assert "yes or no" in prompt


def test_build_choice_prompt_preserves_all_caption_candidates():
    prompt = build_choice_prompt(make_record())

    assert "RMS energy is high" in prompt
    assert "A. wrong caption" in prompt
    assert "B. correct caption" in prompt
    assert prompt.rstrip().endswith("Answer:")


def test_build_position_prior_prompt_omits_evidence_and_candidate_text():
    prompt = build_position_prior_prompt(3)

    assert "A" in prompt
    assert "B" in prompt
    assert "C" in prompt
    assert "RMS energy" not in prompt
    assert "correct caption" not in prompt


def test_evaluate_logprob_reranker_selects_highest_margin_and_preserves_index_map():
    metrics, rows = evaluate_logprob_reranker([make_record()], FakeSupportScorer())

    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["n_eval_records"] == 1
    assert metrics["n_scoring_prompts"] == 2
    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["caption_answer_index"] == 1
    assert rows[0]["candidate_index_map"] == [2, 5]
    assert rows[0]["caption_scores"][1] > rows[0]["caption_scores"][0]


def test_evaluate_choice_logprob_reranker_selects_highest_choice_score():
    metrics, rows = evaluate_choice_logprob_reranker([make_record()], FakeChoiceScorer())

    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["n_scoring_prompts"] == 1
    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["caption_scores"] == [-4.0, -0.5]


def test_evaluate_position_prior_logprob_reranker_scores_labels_without_evidence():
    metrics, rows = evaluate_position_prior_logprob_reranker([make_record()], FakeChoiceScorer())

    assert metrics["system"] == "qwen_position_prior_caption_reranker"
    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["n_scoring_prompts"] == 1
    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["caption_scores"] == [-4.0, -0.5]


def test_evaluate_choice_logprob_reranker_rejects_short_scorer_output():
    class ShortScorer:
        def score_choices(self, prompts, candidate_counts):
            return []

    try:
        evaluate_choice_logprob_reranker([make_record()], ShortScorer())
    except ValueError as exc:
        assert "choice score count" in str(exc)
    else:
        raise AssertionError("expected short scorer output to fail")
