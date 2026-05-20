from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from sensorfact.benchmark import evidence_to_text
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy, counterfactual_rejection_metrics
from sensorfact.schemas import EvidenceCard


class TextEmbedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        ...


def record_evidence_text(record: dict) -> str:
    if record.get("evidence_text"):
        return str(record["evidence_text"])
    return evidence_to_text(EvidenceCard.from_json_dict(record["evidence"]))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def score_candidates(embedder: TextEmbedder, evidence_text: str, candidate_texts: list[str]) -> list[float]:
    if not candidate_texts:
        return []
    embeddings = _normalize_rows(embedder.encode([evidence_text, *candidate_texts]))
    evidence = embeddings[0]
    candidates = embeddings[1:]
    return _cosine_scores(evidence, candidates)


def _cosine_scores(evidence_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> list[float]:
    evidence = _normalize_rows(np.asarray(evidence_embedding, dtype=np.float32))[0]
    candidates = _normalize_rows(np.asarray(candidate_embeddings, dtype=np.float32))
    return [float(score) for score in candidates @ evidence]


def evaluate_embedding_grounding(
    records: list[dict],
    embedder: TextEmbedder,
    threshold: float = 0.5,
    pairwise_margin: float = 0.0,
) -> tuple[dict, list[dict]]:
    selection_examples = []
    y_true: list[int] = []
    y_score: list[float] = []
    score_rows: list[dict] = []
    prepared = []
    texts: list[str] = []
    pairwise_truth: list[int] = []
    pairwise_pred: list[int] = []
    pairwise_margins: list[float] = []

    for record in records:
        evidence_text = record_evidence_text(record)
        candidates = [item["text"] for item in record["caption_selection"]["candidates"]]
        counterfactuals = [item["text"] for item in record["counterfactuals"]]
        text_start = len(texts)
        texts.extend([evidence_text, *candidates, record["positive"]["text"], *counterfactuals])
        prepared.append(
            {
                "record": record,
                "candidate_count": len(candidates),
                "counterfactual_count": len(counterfactuals),
                "text_start": text_start,
            }
        )

    embeddings = _normalize_rows(embedder.encode(texts)) if texts else np.empty((0, 0), dtype=np.float32)

    for item in prepared:
        record = item["record"]
        start = item["text_start"]
        candidate_count = item["candidate_count"]
        counterfactual_count = item["counterfactual_count"]
        evidence_embedding = embeddings[start]
        candidate_start = start + 1
        candidate_end = candidate_start + candidate_count
        positive_index = candidate_end
        counterfactual_start = positive_index + 1
        counterfactual_end = counterfactual_start + counterfactual_count

        caption_scores = _cosine_scores(evidence_embedding, embeddings[candidate_start:candidate_end])
        selection_examples.append(
            {
                "answer_index": record["caption_selection"]["answer_index"],
                "scores": caption_scores,
            }
        )

        positive_score = _cosine_scores(evidence_embedding, embeddings[positive_index : positive_index + 1])[0]
        y_true.append(1)
        y_score.append(positive_score)

        counterfactual_scores = _cosine_scores(
            evidence_embedding,
            embeddings[counterfactual_start:counterfactual_end],
        )
        for score in counterfactual_scores:
            y_true.append(0)
            y_score.append(score)
            pairwise_truth.append(0)
            pairwise_pred.append(0 if score < positive_score - pairwise_margin else 1)
            pairwise_margins.append(float(positive_score - score))
        if counterfactual_scores:
            pairwise_truth.append(1)
            pairwise_pred.append(1)

        score_rows.append(
            {
                "window_id": record["window_id"],
                "caption_scores": caption_scores,
                "answer_index": record["caption_selection"]["answer_index"],
                "positive_score": positive_score,
                "counterfactual_scores": counterfactual_scores,
            }
        )

    cf = counterfactual_rejection_metrics(y_true, y_score, threshold=threshold)
    metrics = {
        "system": "qwen_embedding_similarity",
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": cf["accuracy"],
        "cf_reject_precision": cf["precision"],
        "cf_reject_recall": cf["recall"],
        "cf_reject_f1": cf["f1"],
        "cf_pairwise_reject_accuracy": float(
            sum(1 for truth, pred in zip(pairwise_truth, pairwise_pred) if truth == pred)
            / max(1, len(pairwise_truth))
        ),
        "cf_pairwise_margin_mean": float(np.mean(pairwise_margins)) if pairwise_margins else 0.0,
        "n_eval_records": len(records),
        "support_score_threshold": threshold,
        "pairwise_margin": pairwise_margin,
    }
    return metrics, score_rows


class SentenceTransformerEmbedder:
    def __init__(self, model_dir: str | Path, device: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_dir), device=device)
        self.batch_size = batch_size

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def run_qwen_embedding_eval(
    benchmark_path: str | Path,
    model_dir: str | Path,
    output_metrics_path: str | Path,
    output_scores_path: str | Path | None = None,
    max_records: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
    threshold: float = 0.5,
    pairwise_margin: float = 0.0,
) -> dict:
    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    embedder = SentenceTransformerEmbedder(model_dir=model_dir, device=device, batch_size=batch_size)
    metrics, score_rows = evaluate_embedding_grounding(
        records,
        embedder,
        threshold=threshold,
        pairwise_margin=pairwise_margin,
    )
    metrics["model_dir"] = str(model_dir)
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    write_json(output_metrics_path, metrics)
    if output_scores_path is not None:
        write_jsonl(output_scores_path, score_rows)
    return metrics
