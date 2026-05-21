import importlib.util
from pathlib import Path


SCRIPT = Path("scripts/run_emnlp_raw_sensor_gpu_queue.sh")
SUMMARY = Path("scripts/summarize_raw_sensor_results.py")


def test_raw_sensor_gpu_queue_covers_three_datasets_and_gpu_path():
    text = SCRIPT.read_text(encoding="utf-8")

    for dataset in ("ucihar", "wisdm", "mhealth"):
        assert f"data/processed/{dataset}_train.jsonl" in text
        assert f"data/processed/{dataset}_test.jsonl" in text
        assert f"data/benchmark/{dataset}_sensorfact_train.jsonl" in text
        assert f"${{1}}_sensorfact_hard_v3_test.jsonl" in text
        assert f"raw_sensor_${{dataset}}_metrics.json" in text

    assert "nvidia-smi" in text
    assert "--device cuda" in text
    assert "run_raw_sensor_baseline.py" in text
    assert "summarize_raw_sensor_results.py" in text
    assert "emnlp_raw_sensor_gpu_status.tsv" in text
    assert 'exit "$FAILURES"' in text


def test_raw_sensor_summary_marks_oracle_modes_as_upper_bound():
    text = SUMMARY.read_text(encoding="utf-8")

    assert "raw_sensor_result_lock.json" in text
    assert "docs/raw_sensor_leakage_audit.md" in text
    assert "oracle_fields" in text
    assert "upper-bound" in text or "upper bound" in text
    assert "raw_sensor_field_aligner" in text


def test_raw_sensor_summary_renders_payload():
    spec = importlib.util.spec_from_file_location("summarize_raw_sensor", SUMMARY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "raw_sensor": [
            {
                "dataset": "UCI HAR",
                "caption_accuracy": 0.42,
                "caption_macro_f1": 0.41,
                "cf_reject_f1": 0.52,
                "support_ece": 0.2,
                "support_brier": 0.3,
                "evidence_field_accuracy": 0.6,
                "device": "cuda",
                "n_train_records": 10,
                "n_eval_records": 5,
                "threshold": 0.7,
                "status": "done",
                "metrics_path": "metrics.json",
                "rows_path": "rows.jsonl",
            }
        ]
    }

    rendered = module._render(payload)

    assert "UCI HAR" in rendered
    assert "0.4200" in rendered
    assert "cuda" in rendered
    assert "Numeric-only modes" in rendered
