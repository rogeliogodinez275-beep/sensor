import json
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_emnlp_experiment_report import aggregate_metric_family
from scripts.make_emnlp_experiment_report import main as report_main
from scripts.make_emnlp_experiment_report import support_classification_summary


def write_metric(path: Path, caption: float, cf_f1: float) -> None:
    path.write_text(
        json.dumps(
            {
                "caption_selection_accuracy": caption,
                "cf_reject_f1": cf_f1,
                "n_eval_records": 10,
            }
        ),
        encoding="utf-8",
    )


def test_aggregate_metric_family_reports_mean_std_and_count(tmp_path: Path):
    write_metric(tmp_path / "run_seed1_metrics.json", 0.5, 0.4)
    write_metric(tmp_path / "run_seed2_metrics.json", 0.7, 0.8)

    summary = aggregate_metric_family(tmp_path, "run_seed*_metrics.json")

    assert summary["count"] == 2
    assert summary["caption_selection_accuracy_mean"] == 0.6
    assert summary["cf_reject_f1_mean"] == pytest.approx(0.6)
    assert summary["caption_selection_accuracy_std"] > 0.0
    assert summary["cf_reject_f1_std"] > 0.0


def test_emnlp_report_includes_relay_stress_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    write_metric(outputs / "qwen_llm_ucihar_hard_v3_numeric_mask_sample_metrics.json", 0.4, 0.3)
    write_metric(outputs / "qwen_llm_ucihar_hard_v3_shuffled_evidence_sample_metrics.json", 0.3, 0.2)
    write_metric(outputs / "qwen_llm_ucihar_hard_v3_numeric_mask_full_metrics.json", 0.6, 0.5)
    write_metric(outputs / "qwen_llm_ucihar_hard_v3_numeric_mask_full_prompt_terse_metrics.json", 0.5, 0.4)
    write_metric(outputs / "qwen_llm_wisdm_hard_v3_hidden_evidence_sample_metrics.json", 0.2, 0.1)
    write_metric(outputs / "qwen_llm_wisdm_hard_v3_hidden_evidence_full_metrics.json", 0.3, 0.2)
    write_metric(outputs / "qwen_llm_wisdm_hard_v3_hidden_evidence_full_prompt_chain_then_json_metrics.json", 0.25, 0.15)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_emnlp_experiment_report.py",
            "--workspace",
            str(tmp_path),
            "--output",
            "outputs/report.md",
        ],
    )

    report_main()

    text = (outputs / "report.md").read_text(encoding="utf-8")
    assert "Stress Controls" in text
    assert "UCI hard v3 stress controls" in text
    assert "WISDM hard v3 stress controls" in text
    assert "Full Stress Controls" in text
    assert "UCI hard v3 full stress controls" in text
    assert "Stress Prompt Robustness" in text
    assert "WISDM hard v3 stress prompt robustness" in text


def test_emnlp_report_mentions_supervised_baselines_and_statistics():
    text = Path("scripts/make_emnlp_experiment_report.py").read_text(encoding="utf-8")

    assert "Trainable Supervised Baselines" in text
    assert "Code Model And Hybrid Results" in text
    assert "Coder Prompt And Robustness Controls" in text
    assert "Structured Prompt Ablations" in text
    assert "Axisfix Structured Retest" in text
    assert "Hybrid Verifier Fallbacks" in text
    assert "Hybrid Prompt Robustness" in text
    assert "Qwen Few-Shot Control" in text
    assert "Hybrid Threshold Sweep" in text
    assert "regex + Coder direct fallback - structured regex verifier" in text
    assert "Statistical Confidence" in text
    assert "Paired Comparisons" in text
    assert "Metric Sanity" in text
    assert "supervised_ucihar_hard_v3_metrics.json" in text
    assert "supervised_wisdm_hard_v3_metrics.json" in text
    assert "supervised_numeric_ucihar_hard_v3_metrics.json" in text
    assert "supervised_numeric_wisdm_hard_v3_metrics.json" in text
    assert "coder_llm_ucihar_hard_v3_metrics.json" in text
    assert "coder_llm_ucihar_hard_v3_prompt_*_metrics.json" in text
    assert "qwen_llm_ucihar_hard_v3_prompt_fewshot_json_metrics.json" in text
    assert "hybrid_regex_coder_ucihar_hard_v3_prompt_*_metrics.json" in text
    assert "hybrid_threshold_sweep_ucihar.json" in text
    assert "hybrid_threshold_sweep_wisdm.json" in text
    assert "hybrid_threshold_sweep_mhealth.json" in text
    assert "coder_llm_wisdm_hard_v3_caption_order_seed*_metrics.json" in text
    assert "coder_llm_mhealth_hard_v3_support_balanced_neg*_metrics.json" in text
    assert "axisfix_coder_structured_model_mhealth_hard_v3_metrics.json" in text
    assert "hybrid_regex_coder_wisdm_hard_v3_metrics.json" in text
    assert "hybrid_regex_coder_mhealth_hard_v3_rows.jsonl" in text
    assert "mhealth" in text


def test_emnlp_report_includes_qwen_fewshot_control_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    write_metric(outputs / "qwen_llm_ucihar_hard_v3_prompt_fewshot_json_metrics.json", 0.61, 0.52)
    write_metric(outputs / "qwen_llm_wisdm_hard_v3_prompt_fewshot_json_metrics.json", 0.63, 0.55)
    write_metric(outputs / "qwen_llm_mhealth_hard_v3_prompt_fewshot_json_metrics.json", 0.66, 0.58)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_emnlp_experiment_report.py",
            "--workspace",
            str(tmp_path),
            "--output",
            "outputs/report.md",
        ],
    )

    report_main()

    text = (outputs / "report.md").read_text(encoding="utf-8")
    assert "Qwen Few-Shot Control" in text
    assert "UCI hard v3 Qwen few-shot direct LLM" in text
    assert "WISDM hard v3 Qwen few-shot direct LLM" in text
    assert "MHEALTH hard v3 Qwen few-shot direct LLM" in text


def test_emnlp_report_includes_hybrid_threshold_sweep_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "hybrid_threshold_sweep_ucihar.json").write_text(
        json.dumps(
            {
                "best": {
                    "threshold": 0.5,
                    "caption_selection_accuracy": 0.8,
                    "cf_reject_f1": 0.76,
                    "caption_fallback_rate": 0.26,
                },
                "thresholds": [
                    {"threshold": 0.05, "caption_selection_accuracy": 0.8},
                    {"threshold": 0.5, "caption_selection_accuracy": 0.8},
                    {"threshold": 1.0, "caption_selection_accuracy": 0.8},
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "hybrid_threshold_sweep_wisdm.json").write_text(
        json.dumps(
            {
                "best": {
                    "threshold": 0.5,
                    "caption_selection_accuracy": 0.79,
                    "cf_reject_f1": 0.69,
                    "caption_fallback_rate": 0.41,
                },
                "thresholds": [
                    {"threshold": 0.05, "caption_selection_accuracy": 0.79},
                    {"threshold": 0.5, "caption_selection_accuracy": 0.79},
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs / "hybrid_threshold_sweep_mhealth.json").write_text(
        json.dumps(
            {
                "best": {
                    "threshold": 0.5,
                    "caption_selection_accuracy": 0.81,
                    "cf_reject_f1": 0.76,
                    "caption_fallback_rate": 0.31,
                },
                "thresholds": [
                    {"threshold": 0.05, "caption_selection_accuracy": 0.81},
                    {"threshold": 0.5, "caption_selection_accuracy": 0.81},
                    {"threshold": 1.0, "caption_selection_accuracy": 0.81},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_emnlp_experiment_report.py",
            "--workspace",
            str(tmp_path),
            "--output",
            "outputs/report.md",
        ],
    )

    report_main()

    text = (outputs / "report.md").read_text(encoding="utf-8")
    assert "Hybrid Threshold Sweep" in text
    assert "UCI hard v3 hybrid threshold sweep" in text
    assert "threshold plateau" in text


def test_support_classification_summary_flags_all_negative_accuracy_trap():
    rows = [
        {
            "window_id": "w1",
            "support_predictions": [False, False, False],
            "support_items": [
                {"text": "positive", "supported": True},
                {"text": "negative1", "supported": False},
                {"text": "negative2", "supported": False},
            ],
        }
    ]

    summary = support_classification_summary(rows)

    assert summary["support_accuracy"] == pytest.approx(2 / 3)
    assert summary["positive_recall"] == 0.0
    assert summary["negative_recall"] == 1.0
    assert summary["balanced_accuracy"] == 0.5
    assert summary["support_f1"] == 0.0
    assert summary["all_negative_accuracy"] == pytest.approx(2 / 3)
