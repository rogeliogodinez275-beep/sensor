from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.statistics import bootstrap_ci, paired_bootstrap_delta_ci


METRIC_KEYS = ["caption_selection_accuracy", "cf_reject_accuracy", "cf_reject_f1"]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return float(math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1)))


def aggregate_metric_family(output_dir: str | Path, pattern: str) -> dict:
    root = Path(output_dir)
    rows = [read_json(path) for path in sorted(root.glob(pattern))]
    rows = [row for row in rows if row]
    summary = {"count": len(rows)}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in rows if key in row]
        summary[f"{key}_mean"] = _mean(values)
        summary[f"{key}_std"] = _std(values)
    return summary


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def fmt_ci(point: float, lower: float, upper: float) -> str:
    return f"{point:.4f} [{lower:.4f}, {upper:.4f}]"


def metric_row(title: str, metrics: dict) -> list[str]:
    if not metrics:
        return [title, "missing", "", "", "", ""]
    diagnostic = metrics.get(
        "parse_success_rate",
        metrics.get(
            "evidence_parse_complete_rate",
            metrics.get("caption_fallback_rate", metrics.get("cf_pairwise_reject_accuracy", "")),
        ),
    )
    return [
        title,
        fmt(metrics.get("n_eval_records", metrics.get("count", "N/A"))),
        fmt(metrics.get("caption_selection_accuracy", metrics.get("caption_selection_accuracy_mean", ""))),
        fmt(metrics.get("cf_reject_accuracy", metrics.get("cf_reject_accuracy_mean", ""))),
        fmt(metrics.get("cf_reject_f1", metrics.get("cf_reject_f1_mean", ""))),
        fmt(diagnostic),
    ]


def threshold_sweep_row(title: str, sweep: dict) -> list[str]:
    if not sweep:
        return [title, "missing", "", "", "", ""]
    best = dict(sweep.get("best", {}))
    best_caption = float(best.get("caption_selection_accuracy", 0.0))
    plateau_values = []
    for item in sweep.get("thresholds", []):
        if float(item.get("caption_selection_accuracy", -1.0)) == best_caption:
            plateau_values.append(float(item.get("threshold", 0.0)))
    if plateau_values:
        plateau_text = f"{min(plateau_values):.2f}-{max(plateau_values):.2f}"
    else:
        plateau_text = fmt(best.get("threshold", ""))
    return [
        title,
        fmt(best.get("threshold", "")),
        fmt(best.get("caption_selection_accuracy", "")),
        fmt(best.get("cf_reject_f1", "")),
        fmt(best.get("caption_fallback_rate", "")),
        plateau_text,
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _accuracy(samples: list[dict]) -> float:
    if not samples:
        return 0.0
    return float(sum(int(sample["correct"]) for sample in samples) / len(samples))


def caption_samples(rows: list[dict]) -> list[dict]:
    samples = []
    for row in rows:
        if row.get("caption_prediction") is None:
            continue
        samples.append(
            {
                "key": str(row.get("window_id")),
                "correct": int(int(row["caption_prediction"]) == int(row["caption_answer_index"])),
            }
        )
    return samples


def support_samples(rows: list[dict], threshold: float = 0.5) -> list[dict]:
    samples = []
    for row in rows:
        items = list(row.get("support_items", []))
        if "support_predictions" in row:
            predictions = [bool(item) for item in row.get("support_predictions", [])]
        elif "support_probabilities" in row:
            predictions = [float(item) >= threshold for item in row.get("support_probabilities", [])]
        else:
            continue
        for idx, (item, pred) in enumerate(zip(items, predictions)):
            truth = bool(item.get("supported"))
            key = f"{row.get('window_id')}::{item.get('text', idx)}"
            samples.append({"key": key, "correct": int(bool(pred) == truth), "truth": int(truth), "pred": int(bool(pred))})
    return samples


def support_classification_summary(rows: list[dict]) -> dict[str, float]:
    samples = support_samples(rows)
    if not samples:
        return {}
    tp = sum(1 for sample in samples if sample["truth"] == 1 and sample["pred"] == 1)
    tn = sum(1 for sample in samples if sample["truth"] == 0 and sample["pred"] == 0)
    fp = sum(1 for sample in samples if sample["truth"] == 0 and sample["pred"] == 1)
    fn = sum(1 for sample in samples if sample["truth"] == 1 and sample["pred"] == 0)
    total = max(1, len(samples))
    pos_recall = tp / max(1, tp + fn)
    neg_recall = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 0.0 if precision + pos_recall == 0 else 2 * precision * pos_recall / (precision + pos_recall)
    all_negative_acc = sum(1 for sample in samples if sample["truth"] == 0) / total
    return {
        "support_accuracy": float((tp + tn) / total),
        "positive_recall": float(pos_recall),
        "negative_recall": float(neg_recall),
        "balanced_accuracy": float((pos_recall + neg_recall) / 2.0),
        "support_f1": float(f1),
        "all_negative_accuracy": float(all_negative_acc),
        "n_support_items": float(total),
    }


def metric_sanity_row(title: str, rows: list[dict]) -> list[str]:
    summary = support_classification_summary(rows)
    if not summary:
        return [title, "missing", "", "", "", "", ""]
    return [
        title,
        str(int(summary["n_support_items"])),
        fmt(summary["support_accuracy"]),
        fmt(summary["positive_recall"]),
        fmt(summary["negative_recall"]),
        fmt(summary["balanced_accuracy"]),
        fmt(summary["support_f1"]),
        fmt(summary["all_negative_accuracy"]),
    ]


def ci_row(title: str, rows: list[dict]) -> list[str]:
    caps = caption_samples(rows)
    supports = support_samples(rows)
    if not caps and not supports:
        return [title, "missing", "", ""]
    cap_ci = bootstrap_ci(caps, _accuracy, n_bootstrap=500, seed=42) if caps else (0.0, 0.0, 0.0)
    support_ci = bootstrap_ci(supports, _accuracy, n_bootstrap=500, seed=43) if supports else (0.0, 0.0, 0.0)
    return [
        title,
        str(len(caps)),
        fmt_ci(*cap_ci),
        fmt_ci(*support_ci),
    ]


def _paired_by_key(left: list[dict], right: list[dict]) -> tuple[list[dict], list[dict]]:
    right_by_key = {sample["key"]: sample for sample in right}
    left_out = []
    right_out = []
    for sample in left:
        key = sample["key"]
        if key in right_by_key:
            left_out.append(sample)
            right_out.append(right_by_key[key])
    return left_out, right_out


def paired_rows(title: str, left_rows: list[dict], right_rows: list[dict]) -> list[str]:
    left_caption, right_caption = _paired_by_key(caption_samples(left_rows), caption_samples(right_rows))
    left_support, right_support = _paired_by_key(support_samples(left_rows), support_samples(right_rows))
    if left_caption:
        cap = paired_bootstrap_delta_ci(left_caption, right_caption, _accuracy, n_bootstrap=500, seed=44)
        cap_text = f"{fmt_ci(cap[0], cap[1], cap[2])}; p~{cap[3]:.3f}"
    else:
        cap_text = "missing"
    if left_support:
        support = paired_bootstrap_delta_ci(left_support, right_support, _accuracy, n_bootstrap=500, seed=45)
        support_text = f"{fmt_ci(support[0], support[1], support[2])}; p~{support[3]:.3f}"
    else:
        support_text = "missing"
    return [title, str(len(left_caption)), str(len(left_support)), cap_text, support_text]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an EMNLP-style experiment sufficiency report.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--output", default="outputs/emnlp_experiment_report.md")
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve()
    output_dir = workspace / "outputs"

    full_metrics = [
        ("UCI hard v2 embedding", "qwen_embedding_ucihar_hard_v2_metrics.json"),
        ("UCI hard v2 Qwen 4B", "qwen_llm_ucihar_hard_v2_metrics.json"),
        ("UCI hard v3 embedding", "qwen_embedding_ucihar_hard_v3_metrics.json"),
        ("UCI hard v3 Qwen 4B", "qwen_llm_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v2 embedding", "qwen_embedding_wisdm_hard_v2_metrics.json"),
        ("WISDM hard v2 Qwen 4B", "qwen_llm_wisdm_hard_v2_metrics.json"),
        ("WISDM hard v3 embedding", "qwen_embedding_wisdm_hard_v3_metrics.json"),
        ("WISDM hard v3 Qwen 4B", "qwen_llm_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v2 embedding", "qwen_embedding_mhealth_hard_v2_metrics.json"),
        ("MHEALTH hard v2 Qwen 4B", "qwen_llm_mhealth_hard_v2_metrics.json"),
        ("MHEALTH hard v3 embedding", "qwen_embedding_mhealth_hard_v3_metrics.json"),
        ("MHEALTH hard v3 Qwen 4B", "qwen_llm_mhealth_hard_v3_metrics.json"),
    ]
    rows = [metric_row(title, read_json(output_dir / filename)) for title, filename in full_metrics]

    robustness = [
        (
            "UCI hard v3 prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "UCI hard v3 seed robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_seed*_metrics.json"),
        ),
        (
            "WISDM hard v3 seed robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_seed*_metrics.json"),
        ),
    ]
    full_prompt_robustness = [
        (
            "UCI hard v3 full prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_full_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 full prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_full_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 full prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_full_prompt_*_metrics.json"),
        ),
    ]
    stress_controls = [
        (
            "UCI hard v3 stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_*_sample_metrics.json"),
        ),
        (
            "WISDM hard v3 stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_*_sample_metrics.json"),
        ),
        (
            "MHEALTH hard v3 stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_*_sample_metrics.json"),
        ),
    ]
    full_stress_controls = [
        (
            "UCI hard v3 full stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_*_full_metrics.json"),
        ),
        (
            "WISDM hard v3 full stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_*_full_metrics.json"),
        ),
        (
            "MHEALTH hard v3 full stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_*_full_metrics.json"),
        ),
    ]
    stress_prompt_robustness = [
        (
            "UCI hard v3 stress prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_*_full_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 stress prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_*_full_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 stress prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_*_full_prompt_*_metrics.json"),
        ),
    ]
    embedding_stress_baselines = [
        (
            "UCI hard v3 embedding full stress baselines",
            aggregate_metric_family(output_dir, "qwen_embedding_ucihar_hard_v3_*_full_metrics.json"),
        ),
        (
            "WISDM hard v3 embedding full stress baselines",
            aggregate_metric_family(output_dir, "qwen_embedding_wisdm_hard_v3_*_full_metrics.json"),
        ),
        (
            "MHEALTH hard v3 embedding full stress baselines",
            aggregate_metric_family(output_dir, "qwen_embedding_mhealth_hard_v3_*_full_metrics.json"),
        ),
    ]
    hard_v2_prompt_controls = [
        (
            "UCI hard v2 prompt controls",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v2_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v2 prompt controls",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v2_prompt_*_metrics.json"),
        ),
    ]
    hard_v2_stress_controls = [
        (
            "UCI hard v2 stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v2_*_full_metrics.json"),
        ),
        (
            "WISDM hard v2 stress controls",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v2_*_full_metrics.json"),
        ),
    ]
    candidate_order_robustness = [
        (
            "UCI hard v3 candidate order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json"),
        ),
        (
            "WISDM hard v3 candidate order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json"),
        ),
        (
            "MHEALTH hard v3 candidate order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json"),
        ),
        (
            "UCI hard v2 candidate order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v2_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json"),
        ),
        (
            "WISDM hard v2 candidate order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v2_caption_order_seed[0-9][0-9][0-9][0-9]_metrics.json"),
        ),
    ]
    candidate_order_prompt_robustness = [
        (
            "UCI hard v3 candidate order prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_caption_order_seed*_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 candidate order prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_caption_order_seed*_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 candidate order prompt robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_caption_order_seed*_prompt_*_metrics.json"),
        ),
    ]
    support_ablation_robustness = [
        (
            "UCI hard v3 support order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "WISDM hard v3 support order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 support order robustness",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "UCI hard v3 support balance controls",
            aggregate_metric_family(output_dir, "qwen_llm_ucihar_hard_v3_support_balanced_neg*_metrics.json"),
        ),
        (
            "WISDM hard v3 support balance controls",
            aggregate_metric_family(output_dir, "qwen_llm_wisdm_hard_v3_support_balanced_neg*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 support balance controls",
            aggregate_metric_family(output_dir, "qwen_llm_mhealth_hard_v3_support_balanced_neg*_metrics.json"),
        ),
    ]
    supervised_metrics = [
        ("UCI hard v3 oracle-field supervised verifier", "supervised_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 oracle-field supervised verifier", "supervised_wisdm_hard_v3_metrics.json"),
        ("UCI hard v3 numeric-only supervised verifier", "supervised_numeric_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 numeric-only supervised verifier", "supervised_numeric_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 oracle-field supervised verifier", "supervised_mhealth_hard_v3_metrics.json"),
        ("MHEALTH hard v3 numeric-only supervised verifier", "supervised_numeric_mhealth_hard_v3_metrics.json"),
    ]
    structured_metrics = [
        ("UCI hard v3 structured regex verifier", "qwen_structured_regex_ucihar_hard_v3_metrics.json"),
        ("UCI hard v3 structured model verifier", "qwen_structured_model_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 structured regex verifier", "qwen_structured_regex_wisdm_hard_v3_metrics.json"),
        ("WISDM hard v3 structured model verifier", "qwen_structured_model_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 structured regex verifier", "qwen_structured_regex_mhealth_hard_v3_metrics.json"),
        ("MHEALTH hard v3 structured model verifier", "qwen_structured_model_mhealth_hard_v3_metrics.json"),
    ]
    structured_prompt_ablations = [
        (
            "UCI hard v3 structured prompt ablations",
            aggregate_metric_family(output_dir, "structured_prompt_ucihar_hard_v3_*_metrics.json"),
        ),
        (
            "WISDM hard v3 structured prompt ablations",
            aggregate_metric_family(output_dir, "structured_prompt_wisdm_hard_v3_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 structured prompt ablations",
            aggregate_metric_family(output_dir, "structured_prompt_mhealth_hard_v3_*_metrics.json"),
        ),
    ]
    code_model_metrics = [
        ("UCI hard v3 Coder-7B direct LLM", "coder_llm_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 Coder-7B direct LLM", "coder_llm_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 Coder-7B direct LLM", "coder_llm_mhealth_hard_v3_metrics.json"),
        ("UCI hard v3 Coder-7B structured model", "coder_structured_model_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 Coder-7B structured model", "coder_structured_model_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 Coder-7B structured model", "coder_structured_model_mhealth_hard_v3_metrics.json"),
    ]
    coder_prompt_robustness = [
        (
            "UCI hard v3 Coder prompt robustness",
            aggregate_metric_family(output_dir, "coder_llm_ucihar_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 Coder prompt robustness",
            aggregate_metric_family(output_dir, "coder_llm_wisdm_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 Coder prompt robustness",
            aggregate_metric_family(output_dir, "coder_llm_mhealth_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "UCI hard v3 Coder candidate order robustness",
            aggregate_metric_family(output_dir, "coder_llm_ucihar_hard_v3_caption_order_seed*_metrics.json"),
        ),
        (
            "WISDM hard v3 Coder candidate order robustness",
            aggregate_metric_family(output_dir, "coder_llm_wisdm_hard_v3_caption_order_seed*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 Coder candidate order robustness",
            aggregate_metric_family(output_dir, "coder_llm_mhealth_hard_v3_caption_order_seed*_metrics.json"),
        ),
        (
            "UCI hard v3 Coder support order robustness",
            aggregate_metric_family(output_dir, "coder_llm_ucihar_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "WISDM hard v3 Coder support order robustness",
            aggregate_metric_family(output_dir, "coder_llm_wisdm_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 Coder support order robustness",
            aggregate_metric_family(output_dir, "coder_llm_mhealth_hard_v3_support_order_seed*_metrics.json"),
        ),
        (
            "UCI hard v3 Coder support balance controls",
            aggregate_metric_family(output_dir, "coder_llm_ucihar_hard_v3_support_balanced_neg*_metrics.json"),
        ),
        (
            "WISDM hard v3 Coder support balance controls",
            aggregate_metric_family(output_dir, "coder_llm_wisdm_hard_v3_support_balanced_neg*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 Coder support balance controls",
            aggregate_metric_family(output_dir, "coder_llm_mhealth_hard_v3_support_balanced_neg*_metrics.json"),
        ),
    ]
    qwen_fewshot_metrics = [
        ("UCI hard v3 Qwen few-shot direct LLM", "qwen_llm_ucihar_hard_v3_prompt_fewshot_json_metrics.json"),
        ("WISDM hard v3 Qwen few-shot direct LLM", "qwen_llm_wisdm_hard_v3_prompt_fewshot_json_metrics.json"),
        ("MHEALTH hard v3 Qwen few-shot direct LLM", "qwen_llm_mhealth_hard_v3_prompt_fewshot_json_metrics.json"),
    ]
    axisfix_metrics = [
        ("UCI hard v3 axisfix regex verifier", "axisfix_structured_regex_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 axisfix regex verifier", "axisfix_structured_regex_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 axisfix regex verifier", "axisfix_structured_regex_mhealth_hard_v3_metrics.json"),
        ("UCI hard v3 axisfix Qwen structured model", "axisfix_qwen_structured_model_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 axisfix Qwen structured model", "axisfix_qwen_structured_model_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 axisfix Qwen structured model", "axisfix_qwen_structured_model_mhealth_hard_v3_metrics.json"),
        ("UCI hard v3 axisfix Coder structured model", "axisfix_coder_structured_model_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 axisfix Coder structured model", "axisfix_coder_structured_model_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 axisfix Coder structured model", "axisfix_coder_structured_model_mhealth_hard_v3_metrics.json"),
    ]
    hybrid_metrics = [
        ("UCI hard v3 regex + Qwen direct fallback", "hybrid_regex_qwen_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 regex + Qwen direct fallback", "hybrid_regex_qwen_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 regex + Qwen direct fallback", "hybrid_regex_qwen_mhealth_hard_v3_metrics.json"),
        ("UCI hard v3 regex + Coder direct fallback", "hybrid_regex_coder_ucihar_hard_v3_metrics.json"),
        ("WISDM hard v3 regex + Coder direct fallback", "hybrid_regex_coder_wisdm_hard_v3_metrics.json"),
        ("MHEALTH hard v3 regex + Coder direct fallback", "hybrid_regex_coder_mhealth_hard_v3_metrics.json"),
    ]
    hybrid_prompt_robustness = [
        (
            "UCI hard v3 hybrid prompt robustness",
            aggregate_metric_family(output_dir, "hybrid_regex_coder_ucihar_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "WISDM hard v3 hybrid prompt robustness",
            aggregate_metric_family(output_dir, "hybrid_regex_coder_wisdm_hard_v3_prompt_*_metrics.json"),
        ),
        (
            "MHEALTH hard v3 hybrid prompt robustness",
            aggregate_metric_family(output_dir, "hybrid_regex_coder_mhealth_hard_v3_prompt_*_metrics.json"),
        ),
    ]
    hybrid_threshold_sweeps = [
        ("UCI hard v3 hybrid threshold sweep", "hybrid_threshold_sweep_ucihar.json"),
        ("WISDM hard v3 hybrid threshold sweep", "hybrid_threshold_sweep_wisdm.json"),
        ("MHEALTH hard v3 hybrid threshold sweep", "hybrid_threshold_sweep_mhealth.json"),
    ]
    qwen_ucihar_rows = read_jsonl(output_dir / "qwen_llm_ucihar_hard_v3_rows.jsonl")
    qwen_wisdm_rows = read_jsonl(output_dir / "qwen_llm_wisdm_hard_v3_rows.jsonl")
    qwen_mhealth_rows = read_jsonl(output_dir / "qwen_llm_mhealth_hard_v3_sample1024_rows.jsonl")
    structured_regex_ucihar_rows = read_jsonl(output_dir / "qwen_structured_regex_ucihar_hard_v3_rows.jsonl")
    structured_model_ucihar_rows = read_jsonl(output_dir / "qwen_structured_model_ucihar_hard_v3_rows.jsonl")
    structured_regex_wisdm_rows = read_jsonl(output_dir / "qwen_structured_regex_wisdm_hard_v3_rows.jsonl")
    structured_model_wisdm_rows = read_jsonl(output_dir / "qwen_structured_model_wisdm_hard_v3_rows.jsonl")
    structured_regex_mhealth_rows = read_jsonl(output_dir / "qwen_structured_regex_mhealth_hard_v3_rows.jsonl")
    structured_model_mhealth_rows = read_jsonl(output_dir / "qwen_structured_model_mhealth_hard_v3_rows.jsonl")
    hybrid_qwen_ucihar_rows = read_jsonl(output_dir / "hybrid_regex_qwen_ucihar_hard_v3_rows.jsonl")
    hybrid_qwen_wisdm_rows = read_jsonl(output_dir / "hybrid_regex_qwen_wisdm_hard_v3_rows.jsonl")
    hybrid_qwen_mhealth_rows = read_jsonl(output_dir / "hybrid_regex_qwen_mhealth_hard_v3_rows.jsonl")
    hybrid_coder_ucihar_rows = read_jsonl(output_dir / "hybrid_regex_coder_ucihar_hard_v3_rows.jsonl")
    hybrid_coder_wisdm_rows = read_jsonl(output_dir / "hybrid_regex_coder_wisdm_hard_v3_rows.jsonl")
    hybrid_coder_mhealth_rows = read_jsonl(output_dir / "hybrid_regex_coder_mhealth_hard_v3_rows.jsonl")
    supervised_ucihar_rows = read_jsonl(output_dir / "supervised_ucihar_hard_v3_rows.jsonl")
    supervised_wisdm_rows = read_jsonl(output_dir / "supervised_wisdm_hard_v3_rows.jsonl")
    supervised_numeric_ucihar_rows = read_jsonl(output_dir / "supervised_numeric_ucihar_hard_v3_rows.jsonl")
    supervised_numeric_wisdm_rows = read_jsonl(output_dir / "supervised_numeric_wisdm_hard_v3_rows.jsonl")
    supervised_mhealth_rows = read_jsonl(output_dir / "supervised_mhealth_hard_v3_rows.jsonl")
    supervised_numeric_mhealth_rows = read_jsonl(output_dir / "supervised_numeric_mhealth_hard_v3_rows.jsonl")
    confidence_rows = [
        ci_row("UCI hard v3 Qwen 4B", qwen_ucihar_rows),
        ci_row("WISDM hard v3 Qwen 4B", qwen_wisdm_rows),
        ci_row("MHEALTH hard v3 Qwen 4B sample1024", qwen_mhealth_rows),
        ci_row("UCI hard v3 oracle-field supervised verifier", supervised_ucihar_rows),
        ci_row("WISDM hard v3 oracle-field supervised verifier", supervised_wisdm_rows),
        ci_row("MHEALTH hard v3 oracle-field supervised verifier", supervised_mhealth_rows),
        ci_row("UCI hard v3 structured regex verifier", structured_regex_ucihar_rows),
        ci_row("UCI hard v3 structured model verifier", structured_model_ucihar_rows),
        ci_row("WISDM hard v3 structured regex verifier", structured_regex_wisdm_rows),
        ci_row("WISDM hard v3 structured model verifier", structured_model_wisdm_rows),
        ci_row("MHEALTH hard v3 structured regex verifier", structured_regex_mhealth_rows),
        ci_row("MHEALTH hard v3 structured model verifier", structured_model_mhealth_rows),
        ci_row("UCI hard v3 regex + Coder direct fallback", hybrid_coder_ucihar_rows),
        ci_row("WISDM hard v3 regex + Coder direct fallback", hybrid_coder_wisdm_rows),
        ci_row("MHEALTH hard v3 regex + Coder direct fallback", hybrid_coder_mhealth_rows),
        ci_row("UCI hard v3 numeric-only supervised verifier", supervised_numeric_ucihar_rows),
        ci_row("WISDM hard v3 numeric-only supervised verifier", supervised_numeric_wisdm_rows),
        ci_row("MHEALTH hard v3 numeric-only supervised verifier", supervised_numeric_mhealth_rows),
    ]
    paired_comparisons = [
        paired_rows("UCI hard v3 Qwen 4B - oracle-field supervised verifier", qwen_ucihar_rows, supervised_ucihar_rows),
        paired_rows("WISDM hard v3 Qwen 4B - oracle-field supervised verifier", qwen_wisdm_rows, supervised_wisdm_rows),
        paired_rows("MHEALTH hard v3 Qwen 4B - oracle-field supervised verifier", qwen_mhealth_rows, supervised_mhealth_rows),
        paired_rows("UCI hard v3 Qwen 4B - numeric-only supervised verifier", qwen_ucihar_rows, supervised_numeric_ucihar_rows),
        paired_rows("WISDM hard v3 Qwen 4B - numeric-only supervised verifier", qwen_wisdm_rows, supervised_numeric_wisdm_rows),
        paired_rows("MHEALTH hard v3 Qwen 4B - numeric-only supervised verifier", qwen_mhealth_rows, supervised_numeric_mhealth_rows),
        paired_rows("UCI hard v3 structured regex verifier - Qwen 4B", structured_regex_ucihar_rows, qwen_ucihar_rows),
        paired_rows("UCI hard v3 structured model verifier - Qwen 4B", structured_model_ucihar_rows, qwen_ucihar_rows),
        paired_rows("WISDM hard v3 structured regex verifier - Qwen 4B", structured_regex_wisdm_rows, qwen_wisdm_rows),
        paired_rows("WISDM hard v3 structured model verifier - Qwen 4B", structured_model_wisdm_rows, qwen_wisdm_rows),
        paired_rows("MHEALTH hard v3 structured regex verifier - Qwen 4B", structured_regex_mhealth_rows, qwen_mhealth_rows),
        paired_rows("MHEALTH hard v3 structured model verifier - Qwen 4B", structured_model_mhealth_rows, qwen_mhealth_rows),
        paired_rows("UCI hard v3 regex + Qwen direct fallback - structured regex verifier", hybrid_qwen_ucihar_rows, structured_regex_ucihar_rows),
        paired_rows("WISDM hard v3 regex + Qwen direct fallback - structured regex verifier", hybrid_qwen_wisdm_rows, structured_regex_wisdm_rows),
        paired_rows("MHEALTH hard v3 regex + Qwen direct fallback - structured regex verifier", hybrid_qwen_mhealth_rows, structured_regex_mhealth_rows),
        paired_rows("UCI hard v3 regex + Coder direct fallback - structured regex verifier", hybrid_coder_ucihar_rows, structured_regex_ucihar_rows),
        paired_rows("WISDM hard v3 regex + Coder direct fallback - structured regex verifier", hybrid_coder_wisdm_rows, structured_regex_wisdm_rows),
        paired_rows("MHEALTH hard v3 regex + Coder direct fallback - structured regex verifier", hybrid_coder_mhealth_rows, structured_regex_mhealth_rows),
        paired_rows("UCI hard v3 regex + Coder direct fallback - Coder direct LLM", hybrid_coder_ucihar_rows, read_jsonl(output_dir / "coder_llm_ucihar_hard_v3_rows.jsonl")),
        paired_rows("WISDM hard v3 regex + Coder direct fallback - Coder direct LLM", hybrid_coder_wisdm_rows, read_jsonl(output_dir / "coder_llm_wisdm_hard_v3_rows.jsonl")),
        paired_rows("MHEALTH hard v3 regex + Coder direct fallback - Coder direct LLM", hybrid_coder_mhealth_rows, read_jsonl(output_dir / "coder_llm_mhealth_hard_v3_rows.jsonl")),
        paired_rows("UCI hard v3 oracle-field - numeric-only verifier", supervised_ucihar_rows, supervised_numeric_ucihar_rows),
        paired_rows("WISDM hard v3 oracle-field - numeric-only verifier", supervised_wisdm_rows, supervised_numeric_wisdm_rows),
        paired_rows("MHEALTH hard v3 oracle-field - numeric-only verifier", supervised_mhealth_rows, supervised_numeric_mhealth_rows),
    ]
    metric_sanity = [
        metric_sanity_row("UCI hard v3 Qwen 4B", qwen_ucihar_rows),
        metric_sanity_row("WISDM hard v3 Qwen 4B", qwen_wisdm_rows),
        metric_sanity_row("MHEALTH hard v3 Qwen 4B sample1024", qwen_mhealth_rows),
        metric_sanity_row("UCI hard v3 oracle-field supervised verifier", supervised_ucihar_rows),
        metric_sanity_row("WISDM hard v3 oracle-field supervised verifier", supervised_wisdm_rows),
        metric_sanity_row("MHEALTH hard v3 oracle-field supervised verifier", supervised_mhealth_rows),
        metric_sanity_row("UCI hard v3 structured regex verifier", structured_regex_ucihar_rows),
        metric_sanity_row("UCI hard v3 structured model verifier", structured_model_ucihar_rows),
        metric_sanity_row("WISDM hard v3 structured regex verifier", structured_regex_wisdm_rows),
        metric_sanity_row("WISDM hard v3 structured model verifier", structured_model_wisdm_rows),
        metric_sanity_row("MHEALTH hard v3 structured regex verifier", structured_regex_mhealth_rows),
        metric_sanity_row("MHEALTH hard v3 structured model verifier", structured_model_mhealth_rows),
        metric_sanity_row("UCI hard v3 regex + Coder direct fallback", hybrid_coder_ucihar_rows),
        metric_sanity_row("WISDM hard v3 regex + Coder direct fallback", hybrid_coder_wisdm_rows),
        metric_sanity_row("MHEALTH hard v3 regex + Coder direct fallback", hybrid_coder_mhealth_rows),
        metric_sanity_row("UCI hard v3 numeric-only supervised verifier", supervised_numeric_ucihar_rows),
        metric_sanity_row("WISDM hard v3 numeric-only supervised verifier", supervised_numeric_wisdm_rows),
        metric_sanity_row("MHEALTH hard v3 numeric-only supervised verifier", supervised_numeric_mhealth_rows),
    ]

    text = [
        "# EMNLP 2026 Experiment Sufficiency Report",
        "",
        "## Current Evidence",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Pairwise"],
            rows,
        ),
        "",
        "## Structured Verifier Results",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Pairwise"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in structured_metrics],
        ),
        "",
        "## Structured Prompt Ablations",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in structured_prompt_ablations
            ],
        ),
        "",
        "## Code Model And Hybrid Results",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Fallback"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in code_model_metrics],
        ),
        "",
        "## Coder Prompt And Robustness Controls",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in coder_prompt_robustness
            ],
        ),
        "",
        "## Qwen Few-Shot Control",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Pairwise"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in qwen_fewshot_metrics],
        ),
        "",
        "## Axisfix Structured Retest",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Fallback"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in axisfix_metrics],
        ),
        "",
        "## Hybrid Verifier Fallbacks",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Fallback"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in hybrid_metrics],
        ),
        "",
        "## Hybrid Prompt Robustness",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in hybrid_prompt_robustness
            ],
        ),
        "",
        "## Hybrid Threshold Sweep",
        "",
        markdown_table(
            ["Run", "Best Threshold", "Best Caption Acc", "Best CF F1", "Fallback Rate", "threshold plateau"],
            [threshold_sweep_row(title, read_json(output_dir / filename)) for title, filename in hybrid_threshold_sweeps],
        ),
        "",
        "## Robustness Aggregates",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in robustness
            ],
        ),
        "",
        "## Full Prompt Extensions",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in full_prompt_robustness
            ],
        ),
        "",
        "## Stress Controls",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in stress_controls
            ],
        ),
        "",
        "## Full Stress Controls",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in full_stress_controls
            ],
        ),
        "",
        "## Stress Prompt Robustness",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in stress_prompt_robustness
            ],
        ),
        "",
        "## Baseline And Negative Controls",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in embedding_stress_baselines
                + hard_v2_prompt_controls
                + hard_v2_stress_controls
            ],
        ),
        "",
        "## Candidate Order Robustness",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in candidate_order_robustness
            ],
        ),
        "",
        "## Candidate Order Prompt Robustness",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in candidate_order_prompt_robustness
            ],
        ),
        "",
        "## Support-Statement Ablations",
        "",
        markdown_table(
            ["Family", "Runs", "Caption Acc Mean", "CF Acc Mean", "CF F1 Mean", "Std Notes"],
            [
                [
                    title,
                    str(summary.get("count", 0)),
                    fmt(summary.get("caption_selection_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_accuracy_mean", 0.0)),
                    fmt(summary.get("cf_reject_f1_mean", 0.0)),
                    "caption std="
                    + fmt(summary.get("caption_selection_accuracy_std", 0.0))
                    + "; cf_f1 std="
                    + fmt(summary.get("cf_reject_f1_std", 0.0)),
                ]
                for title, summary in support_ablation_robustness
            ],
        ),
        "",
        "## Trainable Supervised Baselines",
        "",
        markdown_table(
            ["Run", "N", "Caption Acc", "CF Acc", "CF F1", "Parse/Pairwise"],
            [metric_row(title, read_json(output_dir / filename)) for title, filename in supervised_metrics],
        ),
        "",
        "## Statistical Confidence",
        "",
        markdown_table(
            ["Run", "Caption N", "Caption Acc 95% CI", "Support Acc 95% CI"],
            confidence_rows,
        ),
        "",
        "## Paired Comparisons",
        "",
        markdown_table(
            ["Comparison", "Caption Pairs", "Support Pairs", "Caption Delta 95% CI", "Support Delta 95% CI"],
            paired_comparisons,
        ),
        "",
        "## Metric Sanity",
        "",
        "Support accuracy is class-imbalanced because each record has one supported statement and multiple counterfactual statements; all-negative rejection can therefore look strong on accuracy while failing support recall.",
        "",
        markdown_table(
            ["Run", "Support N", "Support Acc", "Pos Recall", "Neg Recall", "Balanced Acc", "Support F1", "All-Neg Acc"],
            metric_sanity,
        ),
        "",
        "## EMNLP Main Readiness Assessment",
        "",
        "- Strengths: two real sensor datasets, subject-disjoint splits, hard v2/v3 difficulty, leakage audit, field-level error analysis, prompt robustness, seed robustness, stress controls, baseline/negative-control tracking, trainable supervised verifier, and bootstrap confidence reporting.",
        "- Remaining pressure points: cite and position against prior sensor-to-text/LLM grounding work; use the supervised verifier as a sanity baseline and avoid claiming method superiority unless the paired comparisons support it.",
        "- Submission stance: hard v3 is the most credible headline setting because it removes near-perfect behavior and exposes model limitations.",
        "",
    ]
    out = workspace / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
