from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from sensorfact.benchmark import evidence_to_text
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy, counterfactual_rejection_metrics
from sensorfact.schemas import EvidenceCard


class TextGenerator(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...


def record_evidence_text(record: dict) -> str:
    if record.get("evidence_text"):
        return str(record["evidence_text"])
    return evidence_to_text(EvidenceCard.from_json_dict(record["evidence"]))


def _extract_json(text: str) -> object | None:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def parse_caption_index(response: str, candidate_count: int) -> int | None:
    parsed = _extract_json(response)
    if isinstance(parsed, dict):
        for key in ["answer_index", "caption_index", "choice_index", "index"]:
            if key in parsed:
                try:
                    value = int(parsed[key])
                except (TypeError, ValueError):
                    continue
                if 0 <= value < candidate_count:
                    return value

    letter_match = re.search(r"\b([A-Z])\b", response.upper())
    if letter_match:
        value = ord(letter_match.group(1)) - ord("A")
        if 0 <= value < candidate_count:
            return value

    number_match = re.search(r"\b(\d+)\b", response)
    if number_match:
        value = int(number_match.group(1))
        if 0 <= value < candidate_count:
            return value
    return None


def _to_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "supported", "support", "1"}:
            return True
        if lowered in {"false", "no", "unsupported", "reject", "rejected", "0"}:
            return False
    return None


def parse_support_labels(response: str, statement_count: int) -> list[bool] | None:
    parsed = _extract_json(response)
    values: list[object] | None = None
    if isinstance(parsed, dict):
        for key in ["supported", "labels", "answers", "is_supported"]:
            if isinstance(parsed.get(key), list):
                values = list(parsed[key])
                break
    elif isinstance(parsed, list):
        values = list(parsed)
    if values is not None:
        labels = [_to_bool(item) for item in values[:statement_count]]
        if len(labels) == statement_count and all(item is not None for item in labels):
            return [bool(item) for item in labels]

    labels_by_index: dict[int, bool] = {}
    for match in re.finditer(
        r"\bS?(\d+)\s*[:=-]\s*(supported|unsupported|true|false|yes|no|reject(?:ed)?)\b",
        response,
        flags=re.IGNORECASE,
    ):
        idx = int(match.group(1))
        value = _to_bool(match.group(2))
        if 0 <= idx < statement_count and value is not None:
            labels_by_index[idx] = value
    if len(labels_by_index) == statement_count:
        return [labels_by_index[idx] for idx in range(statement_count)]
    return None


def _caption_instruction(prompt_style: str) -> str:
    if prompt_style == "terse":
        return 'Only output JSON: {"answer_index": <0-based index>}.'
    if prompt_style == "chain_then_json":
        return (
            "First briefly identify the evidence fields that rule out wrong options. "
            'Then return a final JSON object on the last line: {"answer_index": <zero-based index>}.'
        )
    if prompt_style == "contrastive_elim_json":
        return (
            "Eliminate captions that conflict with any evidence dimension: intensity, rhythm, channel, cadence, "
            "burstiness, trend, or cross-channel relation. Then choose the remaining caption best entailed by the evidence. "
            'Return only JSON: {"answer_index": <zero-based index>}.'
        )
    if prompt_style == "axis_ledger_json":
        return (
            "Privately build an evidence ledger over these axes: intensity, rhythm/periodicity, dominant channel/axis, "
            "cadence/frequency, burstiness, trend, and cross-channel relation. For each caption, reject any option with "
            "an axis-level contradiction, then choose the caption with the strongest remaining entailment. "
            'Return only JSON: {"answer_index": <zero-based index>}.'
        )
    if prompt_style == "fewshot_json":
        return (
            "Choose the best supported caption. Follow this style example exactly:\n"
            '{"answer_index": 1}'
        )
    return 'Return JSON only: {"answer_index": <zero-based index of the best supported caption>}.'


def _support_instruction(prompt_style: str) -> str:
    if prompt_style == "terse":
        return 'Only output JSON: {"supported": [true_or_false_for_S0, true_or_false_for_S1, ...]}.'
    if prompt_style == "chain_then_json":
        return (
            "First briefly identify which evidence fields each statement matches or contradicts. "
            'Then return a final JSON object on the last line: {"supported": [true_or_false_for_S0, true_or_false_for_S1, ...]}.'
        )
    if prompt_style == "fewshot_json":
        return (
            "Decide support for every statement. Follow this style example exactly:\n"
            '{"supported": [true, false, false]}'
        )
    return 'Return JSON only: {"supported": [true_or_false_for_S0, true_or_false_for_S1, ...]}.'


def build_caption_prompt(record: dict, prompt_style: str = "strict_json") -> str:
    evidence = record_evidence_text(record)
    candidates = record["caption_selection"]["candidates"]
    options = "\n".join(
        f"{chr(ord('A') + idx)}. {item['text']}" for idx, item in enumerate(candidates)
    )
    return f"""You are evaluating whether captions are supported by wearable sensor evidence.

Evidence:
{evidence}

Caption candidates:
{options}

{_caption_instruction(prompt_style)}
"""


def ordered_support_items(
    record: dict,
    seed: int = 42,
    negative_count: int | None = None,
) -> list[dict]:
    items = [
        {
            "kind": "positive",
            "original_index": 0,
            "text": record["positive"]["text"],
            "supported": True,
            "changed_fact": None,
            "changed_facts": [],
        }
    ]
    counterfactuals = list(record["counterfactuals"])
    if negative_count is not None and negative_count >= 0:
        counterfactuals = counterfactuals[:negative_count]
    for idx, item in enumerate(counterfactuals):
        changed_facts = item.get("changed_facts")
        if changed_facts is None:
            changed_facts = [item["changed_fact"]] if item.get("changed_fact") else []
        items.append(
            {
                "kind": "counterfactual",
                "original_index": idx,
                "text": item["text"],
                "supported": False,
                "changed_fact": item.get("changed_fact"),
                "changed_facts": list(changed_facts),
            }
        )
    rng = random.Random(f"{seed}:{record.get('window_id', '')}:support")
    rng.shuffle(items)
    return items


def build_support_prompt(
    record: dict,
    prompt_style: str = "strict_json",
    support_seed: int = 42,
    support_negative_count: int | None = None,
) -> str:
    evidence = record_evidence_text(record)
    support_items = ordered_support_items(
        record,
        seed=support_seed,
        negative_count=support_negative_count,
    )
    statements = [item["text"] for item in support_items]
    rows = "\n".join(f"S{idx}: {text}" for idx, text in enumerate(statements))
    return f"""You are checking caption support against wearable sensor evidence.

Evidence:
{evidence}

Statements:
{rows}

For each statement, decide whether it is supported by the evidence. Mark a statement true when its main claim is entailed by the evidence, even if wording is not identical. Mark it false when it contradicts intensity, rhythm, channel, cadence, burstiness, or trend evidence.
{_support_instruction(prompt_style)}
"""


def evaluate_llm_grounding(
    records: list[dict],
    llm: TextGenerator,
    prompt_style: str = "strict_json",
    support_seed: int = 42,
    support_negative_count: int | None = None,
    caption_only: bool = False,
) -> tuple[dict, list[dict]]:
    caption_prompts = [build_caption_prompt(record, prompt_style=prompt_style) for record in records]
    support_prompts = []
    if not caption_only:
        support_prompts = [
            build_support_prompt(
                record,
                prompt_style=prompt_style,
                support_seed=support_seed,
                support_negative_count=support_negative_count,
            )
            for record in records
        ]
    responses = llm.generate(caption_prompts + support_prompts)
    caption_responses = responses[: len(records)]
    support_responses = responses[len(records) :]
    if caption_only:
        support_responses = [None for _ in records]

    selection_examples = []
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []

    for record, caption_response, support_response in zip(records, caption_responses, support_responses):
        candidate_count = len(record["caption_selection"]["candidates"])
        caption_prediction = parse_caption_index(caption_response, candidate_count)
        caption_scores = [0.0 for _ in range(candidate_count)]
        if caption_prediction is not None:
            caption_scores[caption_prediction] = 1.0
        selection_examples.append(
            {
                "answer_index": record["caption_selection"]["answer_index"],
                "scores": caption_scores,
            }
        )

        support_items: list[dict] = []
        support_predictions: list[bool] = []
        support_parse_succeeded = True
        if not caption_only:
            support_items = ordered_support_items(
                record,
                seed=support_seed,
                negative_count=support_negative_count,
            )
            statement_count = len(support_items)
            support_predictions = parse_support_labels(str(support_response), statement_count)
            support_parse_succeeded = support_predictions is not None
            if support_predictions is None:
                support_predictions = [False for _ in range(statement_count)]
            truths = [1 if item["supported"] else 0 for item in support_items]
            scores = [1.0 if item else 0.0 for item in support_predictions]
            y_true.extend(truths)
            y_score.extend(scores)

        rows.append(
            {
                "window_id": record["window_id"],
                "candidate_index_map": record.get("candidate_index_map"),
                "caption_prediction": caption_prediction,
                "caption_answer_index": record["caption_selection"]["answer_index"],
                "caption_response": caption_response,
                "support_predictions": support_predictions,
                "support_parse_succeeded": support_parse_succeeded,
                "support_response": support_response,
                "support_items": support_items,
            }
        )

    cf = None if caption_only else counterfactual_rejection_metrics(y_true, y_score, threshold=0.5)
    parse_success_rate = float(
        np.mean(
            [
                row["caption_prediction"] is not None
                and (caption_only or (row["support_parse_succeeded"] and len(row["support_predictions"]) == len(row["support_items"])))
                for row in rows
            ]
        )
    ) if records else 0.0
    metrics = {
        "system": "qwen_frozen_llm",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": None if cf is None else cf["accuracy"],
        "cf_reject_precision": None if cf is None else cf["precision"],
        "cf_reject_recall": None if cf is None else cf["recall"],
        "cf_reject_f1": None if cf is None else cf["f1"],
        "n_eval_records": len(records),
        "n_generation_prompts": len(caption_prompts) + len(support_prompts),
        "parse_success_rate": parse_success_rate,
        "prompt_style": prompt_style,
        "support_seed": support_seed,
        "support_negative_count": support_negative_count,
        "caption_only": caption_only,
        "support_evaluated": not caption_only,
    }
    return metrics, rows


class TransformersCausalLM:
    def __init__(
        self,
        model_dir: str | Path,
        batch_size: int = 2,
        max_new_tokens: int = 96,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            trust_remote_code=True,
        ).to(target_device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = target_device

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a precise evaluator. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def generate(self, prompts: list[str]) -> list[str]:
        import torch

        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = [self._format_prompt(prompt) for prompt in prompts[start : start + self.batch_size]]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            input_length = encoded["input_ids"].shape[1]
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[:, input_length:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


def run_qwen_llm_eval(
    benchmark_path: str | Path,
    model_dir: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path,
    max_records: int = 32,
    batch_size: int = 2,
    max_new_tokens: int = 96,
    device: str | None = None,
    prompt_style: str = "strict_json",
    support_seed: int = 42,
    support_negative_count: int | None = None,
    caption_only: bool = False,
) -> dict:
    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    llm = TransformersCausalLM(
        model_dir=model_dir,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    metrics, rows = evaluate_llm_grounding(
        records,
        llm,
        prompt_style=prompt_style,
        support_seed=support_seed,
        support_negative_count=support_negative_count,
        caption_only=caption_only,
    )
    metrics["model_dir"] = str(model_dir)
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    metrics["batch_size"] = batch_size
    metrics["max_new_tokens"] = max_new_tokens
    metrics["prompt_style"] = prompt_style
    metrics["support_seed"] = support_seed
    metrics["support_negative_count"] = support_negative_count
    metrics["caption_only"] = caption_only
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics
