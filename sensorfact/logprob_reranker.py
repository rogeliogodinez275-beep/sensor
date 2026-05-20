from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from sensorfact.qwen_llm_eval import record_evidence_text
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy


class SupportScorer(Protocol):
    def score_support(self, prompts: list[str]) -> list[dict]:
        ...


class ChoiceScorer(Protocol):
    def score_choices(self, prompts: list[str], candidate_counts: list[int]) -> list[list[dict]]:
        ...


def build_candidate_support_prompt(record: dict, candidate_text: str) -> str:
    evidence = record_evidence_text(record)
    return f"""You are checking whether one caption is entailed by wearable sensor evidence.

Evidence:
{evidence}

Caption:
{candidate_text}

Answer yes or no. The answer is"""


def build_choice_prompt(record: dict) -> str:
    evidence = record_evidence_text(record)
    candidates = record["caption_selection"]["candidates"]
    options = "\n".join(
        f"{chr(ord('A') + idx)}. {item['text']}" for idx, item in enumerate(candidates)
    )
    return f"""You are selecting the caption best supported by wearable sensor evidence.

Evidence:
{evidence}

Caption candidates:
{options}

Return only the letter of the best supported caption.
Answer:"""


def build_position_prior_prompt(candidate_count: int) -> str:
    labels = ", ".join(chr(ord("A") + idx) for idx in range(candidate_count))
    return f"""Choose one option label from this closed set without seeing evidence or caption text.

Available labels: {labels}

Return only one label.
Answer:"""


def evaluate_logprob_reranker(
    records: list[dict],
    scorer: SupportScorer,
) -> tuple[dict, list[dict]]:
    prompts: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(prompts)
        prompts.extend(
            build_candidate_support_prompt(record, item["text"])
            for item in record["caption_selection"]["candidates"]
        )
        spans.append((start, len(prompts)))

    scored = scorer.score_support(prompts) if prompts else []
    selection_examples = []
    rows: list[dict] = []
    for record, (start, end) in zip(records, spans):
        candidate_scores = [
            float(item.get("margin", item.get("positive_logprob", 0.0) - item.get("negative_logprob", 0.0)))
            for item in scored[start:end]
        ]
        prediction = int(np.argmax(candidate_scores)) if candidate_scores else None
        answer_index = int(record["caption_selection"]["answer_index"])
        selection_examples.append({"answer_index": answer_index, "scores": candidate_scores})
        rows.append(
            {
                "window_id": record["window_id"],
                "candidate_index_map": record.get("candidate_index_map"),
                "caption_prediction": prediction,
                "caption_answer_index": answer_index,
                "caption_scores": candidate_scores,
                "candidate_support_scores": scored[start:end],
            }
        )

    metrics = {
        "system": "qwen_logprob_candidate_support_reranker",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "n_eval_records": len(records),
        "n_scoring_prompts": len(prompts),
        "support_evaluated": False,
    }
    return metrics, rows


def evaluate_choice_logprob_reranker(
    records: list[dict],
    scorer: ChoiceScorer,
) -> tuple[dict, list[dict]]:
    prompts = [build_choice_prompt(record) for record in records]
    candidate_counts = [len(record["caption_selection"]["candidates"]) for record in records]
    scored = scorer.score_choices(prompts, candidate_counts) if prompts else []
    if len(scored) != len(records):
        raise ValueError(f"choice score count mismatch: expected {len(records)}, got {len(scored)}")
    selection_examples = []
    rows: list[dict] = []
    for record, choice_scores in zip(records, scored):
        caption_scores = [float(item.get("logprob", float("-inf"))) for item in choice_scores]
        if len(caption_scores) != len(record["caption_selection"]["candidates"]):
            raise ValueError(
                "choice score candidate count mismatch for "
                f"window_id={record.get('window_id')}: expected "
                f"{len(record['caption_selection']['candidates'])}, got {len(caption_scores)}"
            )
        if any(not np.isfinite(score) for score in caption_scores):
            raise ValueError(f"non-finite choice score for window_id={record.get('window_id')}")
        prediction = int(np.argmax(caption_scores)) if caption_scores else None
        answer_index = int(record["caption_selection"]["answer_index"])
        selection_examples.append({"answer_index": answer_index, "scores": caption_scores})
        rows.append(
            {
                "window_id": record["window_id"],
                "candidate_index_map": record.get("candidate_index_map"),
                "caption_prediction": prediction,
                "caption_answer_index": answer_index,
                "caption_scores": caption_scores,
                "choice_logprobs": choice_scores,
            }
        )

    metrics = {
        "system": "qwen_choice_logprob_caption_reranker",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "n_eval_records": len(records),
        "n_scoring_prompts": len(prompts),
        "support_evaluated": False,
    }
    return metrics, rows


def evaluate_position_prior_logprob_reranker(
    records: list[dict],
    scorer: ChoiceScorer,
) -> tuple[dict, list[dict]]:
    prompts = [
        build_position_prior_prompt(len(record["caption_selection"]["candidates"]))
        for record in records
    ]
    candidate_counts = [len(record["caption_selection"]["candidates"]) for record in records]
    scored = scorer.score_choices(prompts, candidate_counts) if prompts else []
    if len(scored) != len(records):
        raise ValueError(f"choice score count mismatch: expected {len(records)}, got {len(scored)}")
    selection_examples = []
    rows: list[dict] = []
    for record, choice_scores in zip(records, scored):
        caption_scores = [float(item.get("logprob", float("-inf"))) for item in choice_scores]
        if len(caption_scores) != len(record["caption_selection"]["candidates"]):
            raise ValueError(
                "position-prior candidate count mismatch for "
                f"window_id={record.get('window_id')}: expected "
                f"{len(record['caption_selection']['candidates'])}, got {len(caption_scores)}"
            )
        if any(not np.isfinite(score) for score in caption_scores):
            raise ValueError(f"non-finite position-prior score for window_id={record.get('window_id')}")
        prediction = int(np.argmax(caption_scores)) if caption_scores else None
        answer_index = int(record["caption_selection"]["answer_index"])
        selection_examples.append({"answer_index": answer_index, "scores": caption_scores})
        rows.append(
            {
                "window_id": record["window_id"],
                "candidate_index_map": record.get("candidate_index_map"),
                "caption_prediction": prediction,
                "caption_answer_index": answer_index,
                "caption_scores": caption_scores,
                "choice_logprobs": choice_scores,
            }
        )

    metrics = {
        "system": "qwen_position_prior_caption_reranker",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "n_eval_records": len(records),
        "n_scoring_prompts": len(prompts),
        "support_evaluated": False,
    }
    return metrics, rows


class TransformersYesNoScorer:
    def __init__(
        self,
        model_dir: str | Path,
        batch_size: int = 4,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.positive_token_id = self._single_token_id(" yes")
        self.negative_token_id = self._single_token_id(" no")

    def _single_token_id(self, text: str) -> int:
        token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"text did not tokenize: {text!r}")
        return int(token_ids[-1])

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a precise binary entailment evaluator.",
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

    def score_support(self, prompts: list[str]) -> list[dict]:
        import torch

        rows: list[dict] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = [self._format_prompt(prompt) for prompt in prompts[start : start + self.batch_size]]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            positive = log_probs[:, self.positive_token_id]
            negative = log_probs[:, self.negative_token_id]
            margins = positive - negative
            for pos, neg, margin in zip(positive.tolist(), negative.tolist(), margins.tolist()):
                rows.append(
                    {
                        "positive_token": "yes",
                        "negative_token": "no",
                        "positive_logprob": float(pos),
                        "negative_logprob": float(neg),
                        "margin": float(margin),
                    }
                )
        return rows


class TransformersChoiceScorer:
    def __init__(
        self,
        model_dir: str | Path,
        batch_size: int = 4,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    def _choice_token_id(self, label: str) -> int:
        token_ids = self.tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"choice did not tokenize: {label!r}")
        return int(token_ids[-1])

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a precise forced-choice evaluator.",
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

    def score_choices(self, prompts: list[str], candidate_counts: list[int]) -> list[list[dict]]:
        import torch

        rows: list[list[dict]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            batch_counts = candidate_counts[start : start + self.batch_size]
            batch = [self._format_prompt(prompt) for prompt in batch_prompts]
            encoded = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits[:, -1, :]
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row_idx, candidate_count in enumerate(batch_counts):
                choice_rows = []
                for choice_idx in range(candidate_count):
                    label = chr(ord("A") + choice_idx)
                    token_id = self._choice_token_id(label)
                    choice_rows.append(
                        {
                            "choice": label,
                            "token_id": token_id,
                            "logprob": float(log_probs[row_idx, token_id].item()),
                        }
                    )
                rows.append(choice_rows)
        return rows


def run_logprob_reranker(
    benchmark_path: str | Path,
    model_dir: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path,
    max_records: int | None = None,
    batch_size: int = 4,
    device: str | None = None,
) -> dict:
    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    scorer = TransformersYesNoScorer(model_dir=model_dir, batch_size=batch_size, device=device)
    metrics, rows = evaluate_logprob_reranker(records, scorer)
    metrics["model_dir"] = str(model_dir)
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    metrics["batch_size"] = batch_size
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics


def run_choice_logprob_reranker(
    benchmark_path: str | Path,
    model_dir: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path,
    max_records: int | None = None,
    batch_size: int = 4,
    device: str | None = None,
) -> dict:
    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    scorer = TransformersChoiceScorer(model_dir=model_dir, batch_size=batch_size, device=device)
    metrics, rows = evaluate_choice_logprob_reranker(records, scorer)
    metrics["model_dir"] = str(model_dir)
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    metrics["batch_size"] = batch_size
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics


def run_position_prior_logprob_reranker(
    benchmark_path: str | Path,
    model_dir: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path,
    max_records: int | None = None,
    batch_size: int = 4,
    device: str | None = None,
) -> dict:
    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    scorer = TransformersChoiceScorer(model_dir=model_dir, batch_size=batch_size, device=device)
    metrics, rows = evaluate_position_prior_logprob_reranker(records, scorer)
    metrics["model_dir"] = str(model_dir)
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    metrics["batch_size"] = batch_size
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics
