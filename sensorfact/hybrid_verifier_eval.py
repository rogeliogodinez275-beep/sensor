from __future__ import annotations

from pathlib import Path

from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy, counterfactual_rejection_metrics


def _argmax_index(scores: list[float]) -> int | None:
    if not scores:
        return None
    return max(range(len(scores)), key=lambda idx: float(scores[idx]))


def _structured_decision(scores: list[float], threshold: float) -> int | None:
    positive = [idx for idx, score in enumerate(scores) if float(score) >= threshold]
    if len(positive) == 1:
        return positive[0]
    return None


def evaluate_hybrid_grounding(
    structured_rows: list[dict],
    direct_rows: list[dict],
    *,
    structured_threshold: float = 0.5,
    system_name: str = "hybrid_structured_direct",
) -> tuple[dict, list[dict]]:
    direct_by_window = {str(row.get("window_id")): row for row in direct_rows}
    selection_examples = []
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    fallback_count = 0
    missing_direct_count = 0
    fallback_correct = 0

    for structured in structured_rows:
        window_id = str(structured.get("window_id"))
        direct = direct_by_window.get(window_id)
        structured_scores = [float(score) for score in structured.get("caption_scores", [])]
        prediction = _structured_decision(structured_scores, structured_threshold)
        caption_source = "structured"

        if prediction is None:
            caption_source = "direct_fallback"
            fallback_count += 1
            direct_prediction = None if direct is None else direct.get("caption_prediction")
            direct_index_map = None if direct is None else direct.get("candidate_index_map")
            if direct_prediction is not None:
                mapped_prediction = int(direct_prediction)
                if direct_index_map is not None:
                    try:
                        mapped_prediction = int(direct_index_map[int(direct_prediction)])
                    except (TypeError, ValueError, IndexError):
                        mapped_prediction = int(direct_prediction)
                prediction = mapped_prediction
            else:
                missing_direct_count += 1
                prediction = _argmax_index(structured_scores)

        answer_index = int(structured["caption_answer_index"])
        caption_scores = [0.0 for _ in structured_scores]
        if prediction is not None and 0 <= prediction < len(caption_scores):
            caption_scores[prediction] = 1.0
        if caption_source == "direct_fallback":
            fallback_correct += int(prediction == answer_index)
        selection_examples.append({"answer_index": answer_index, "scores": caption_scores})

        # By design, hybrid fallback only changes caption selection.
        # Support-side counterfactual rejection stays on the structured verifier path
        # unless we explicitly add a support-routing mechanism.
        support_items = list(structured.get("support_items", []))
        support_predictions = [bool(item) for item in structured.get("support_predictions", [])]
        for item, pred in zip(support_items, support_predictions):
            y_true.append(1 if item.get("supported") else 0)
            y_score.append(1.0 if pred else 0.0)

        rows.append(
            {
                "window_id": window_id,
                "caption_prediction": prediction,
                "caption_answer_index": answer_index,
                "caption_scores": caption_scores,
                "caption_source": caption_source,
                "structured_caption_scores": structured_scores,
                "direct_caption_prediction": None if direct is None else direct.get("caption_prediction"),
                "direct_candidate_index_map": None if direct is None else direct.get("candidate_index_map"),
                "support_predictions": support_predictions,
                "support_items": support_items,
            }
        )

    cf = counterfactual_rejection_metrics(y_true, y_score, threshold=0.5)
    n_rows = len(structured_rows)
    metrics = {
        "system": system_name,
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": cf["accuracy"],
        "cf_reject_precision": cf["precision"],
        "cf_reject_recall": cf["recall"],
        "cf_reject_f1": cf["f1"],
        "support_source": "structured_only",
        "n_eval_records": n_rows,
        "caption_fallback_rate": fallback_count / max(1, n_rows),
        "caption_fallback_count": fallback_count,
        "missing_direct_count": missing_direct_count,
        "caption_fallback_accuracy": fallback_correct / max(1, fallback_count),
        "structured_decisive_rate": 1.0 - fallback_count / max(1, n_rows),
    }
    return metrics, rows


def run_hybrid_verifier_eval(
    structured_rows_path: str | Path,
    direct_rows_path: str | Path,
    output_metrics_path: str | Path,
    output_rows_path: str | Path,
    *,
    structured_threshold: float = 0.5,
    system_name: str = "hybrid_structured_direct",
) -> dict:
    structured_rows = list(read_jsonl(structured_rows_path))
    direct_rows = list(read_jsonl(direct_rows_path))
    metrics, rows = evaluate_hybrid_grounding(
        structured_rows,
        direct_rows,
        structured_threshold=structured_threshold,
        system_name=system_name,
    )
    metrics["structured_rows_path"] = str(structured_rows_path)
    metrics["direct_rows_path"] = str(direct_rows_path)
    metrics["structured_threshold"] = structured_threshold
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics
