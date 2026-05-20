from sensorfact.qwen_llm_eval import (
    build_caption_prompt,
    build_support_prompt,
    ordered_support_items,
    evaluate_llm_grounding,
    parse_caption_index,
    parse_support_labels,
)


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.prompts = []

    def generate(self, prompts):
        self.prompts.extend(prompts)
        return self.responses[: len(prompts)]


def make_record():
    return {
        "window_id": "w1",
        "evidence": {
            "window_id": "w1",
            "dataset_id": "toy",
            "intensity": "high",
            "periodicity": "strong",
            "dominant_axis": "acc_x",
            "dominant_frequency": "mid",
            "cross_channel_relation": "no clear relation",
            "lead_lag": "no clear lag",
            "burstiness": "smooth",
            "trend_segments": ["rise"],
        },
        "caption_selection": {
            "answer_index": 1,
            "candidates": [{"text": "wrong"}, {"text": "right"}],
        },
        "positive": {"text": "right"},
        "counterfactuals": [{"text": "wrong"}],
    }


def test_parse_caption_index_accepts_json_and_letters():
    assert parse_caption_index('{"answer_index": 2}', 4) == 2
    assert parse_caption_index("The best answer is B.", 4) == 1


def test_parse_support_labels_accepts_json_booleans():
    assert parse_support_labels('{"supported": [true, false, false]}', 3) == [True, False, False]
    assert parse_support_labels("S0: supported\nS1: unsupported", 2) == [True, False]


def test_evaluate_llm_grounding_uses_two_prompts_per_record():
    expected_labels = [item["supported"] for item in ordered_support_items(make_record())]
    llm = FakeLLM(
        [
            '{"answer_index": 1}',
            '{"supported": ' + str(expected_labels).lower() + "}",
        ]
    )

    metrics, rows = evaluate_llm_grounding([make_record()], llm)

    assert len(llm.prompts) == 2
    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["cf_reject_accuracy"] == 1.0
    assert metrics["cf_reject_f1"] == 1.0
    assert metrics["n_eval_records"] == 1
    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["support_predictions"] == expected_labels
    assert rows[0]["support_items"] == ordered_support_items(make_record())


def test_evaluate_llm_grounding_caption_only_uses_one_prompt_per_record():
    llm = FakeLLM(['{"answer_index": 1}'])

    metrics, rows = evaluate_llm_grounding([make_record()], llm, caption_only=True)

    assert len(llm.prompts) == 1
    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["support_evaluated"] is False
    assert metrics["cf_reject_f1"] is None
    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["support_predictions"] == []
    assert rows[0]["support_items"] == []


def test_ordered_support_items_can_use_alternate_seed_without_changing_truth_set():
    record = make_record()
    record["counterfactuals"] = [{"text": f"wrong {idx}"} for idx in range(5)]

    first = ordered_support_items(record, seed=7001)
    second = ordered_support_items(record, seed=7002)

    assert [item["text"] for item in first] != [item["text"] for item in second]
    assert sorted((item["text"], item["supported"]) for item in first) == sorted(
        (item["text"], item["supported"]) for item in second
    )


def test_ordered_support_items_can_balance_negative_count():
    record = make_record()
    record["counterfactuals"] = [{"text": f"wrong {idx}"} for idx in range(5)]

    items = ordered_support_items(record, seed=7001, negative_count=2)

    assert len(items) == 3
    assert sum(1 for item in items if item["supported"]) == 1
    assert sum(1 for item in items if not item["supported"]) == 2


def test_evaluate_llm_grounding_records_support_ablation_controls():
    record = make_record()
    record["counterfactuals"] = [{"text": f"wrong {idx}"} for idx in range(5)]
    expected_labels = [
        item["supported"] for item in ordered_support_items(record, seed=7001, negative_count=2)
    ]
    llm = FakeLLM(
        [
            '{"answer_index": 1}',
            '{"supported": ' + str(expected_labels).lower() + "}",
        ]
    )

    metrics, rows = evaluate_llm_grounding(
        [record],
        llm,
        support_seed=7001,
        support_negative_count=2,
    )

    assert metrics["support_seed"] == 7001
    assert metrics["support_negative_count"] == 2
    assert metrics["cf_reject_accuracy"] == 1.0
    assert rows[0]["support_items"] == ordered_support_items(record, seed=7001, negative_count=2)


def test_parse_success_rate_requires_support_parse_success():
    llm = FakeLLM(
        [
            '{"answer_index": 1}',
            "not valid support json",
        ]
    )

    metrics, rows = evaluate_llm_grounding([make_record()], llm)

    assert rows[0]["caption_prediction"] == 1
    assert rows[0]["support_predictions"] == [False, False]
    assert metrics["parse_success_rate"] == 0.0


def test_support_prompt_order_does_not_put_positive_at_fixed_s0_position():
    positions = []
    for idx in range(24):
        record = make_record()
        record["window_id"] = f"w{idx}"
        items = ordered_support_items(record)
        positions.append(next(i for i, item in enumerate(items) if item["kind"] == "positive"))

    assert len(set(positions)) > 1
    assert positions.count(0) < len(positions)


def test_prompt_styles_change_instruction_surface():
    record = make_record()

    strict_caption = build_caption_prompt(record, prompt_style="strict_json")
    terse_caption = build_caption_prompt(record, prompt_style="terse")
    chain_caption = build_caption_prompt(record, prompt_style="chain_then_json")
    fewshot_caption = build_caption_prompt(record, prompt_style="fewshot_json")
    contrastive_caption = build_caption_prompt(record, prompt_style="contrastive_elim_json")
    axis_ledger_caption = build_caption_prompt(record, prompt_style="axis_ledger_json")
    strict_support = build_support_prompt(record, prompt_style="strict_json")
    terse_support = build_support_prompt(record, prompt_style="terse")
    fewshot_support = build_support_prompt(record, prompt_style="fewshot_json")

    assert strict_caption != terse_caption
    assert "Return JSON only" in strict_caption
    assert "Only output JSON" in terse_caption
    assert "briefly identify" in chain_caption
    assert '"answer_index": 1' in fewshot_caption
    assert "eliminate" in contrastive_caption.lower()
    assert "intensity" in contrastive_caption
    assert "rhythm" in contrastive_caption
    assert "channel" in contrastive_caption
    assert "ledger" in axis_ledger_caption.lower()
    assert "dominant channel/axis" in axis_ledger_caption
    assert "axis-level contradiction" in axis_ledger_caption
    assert '"supported": [true, false' in fewshot_support
    assert strict_support != terse_support
    assert "JSON only" in strict_support
