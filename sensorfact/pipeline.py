from __future__ import annotations

from pathlib import Path

from sensorfact.baselines import evaluate_grounding, evaluate_har, write_checkpoint
from sensorfact.benchmark import SensorFactBenchmarkBuilder
from sensorfact.data import (
    download_modelscope_dataset,
    load_ucihar,
    make_synthetic_windows,
    write_window_files,
)
from sensorfact.evidence import EvidenceCalibrator, EvidenceExtractor
from sensorfact.io import read_jsonl, write_json, write_jsonl
from sensorfact.schemas import EvidenceCard, SensorWindow


CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]


def prepare_windows(
    workspace: Path,
    dataset_id: str,
    max_train_windows: int,
    max_test_windows: int,
    synthetic: bool,
    sample_rate: float,
) -> tuple[list[SensorWindow], list[SensorWindow], dict]:
    raw_dir = workspace / "data" / "raw" / dataset_id.replace("/", "_")
    processed_dir = workspace / "data" / "processed"
    if synthetic:
        train, test = make_synthetic_windows(
            train_count=120 if max_train_windows < 0 else max_train_windows,
            test_count=60 if max_test_windows < 0 else max_test_windows,
            sample_rate=sample_rate,
        )
        source = "synthetic"
    else:
        if not raw_dir.exists():
            download_modelscope_dataset(dataset_id, raw_dir)
        train, test = load_ucihar(
            raw_dir,
            max_train_windows=None if max_train_windows < 0 else max_train_windows,
            max_test_windows=None if max_test_windows < 0 else max_test_windows,
        )
        source = dataset_id
    files = write_window_files(train, test, processed_dir)
    return train, test, {"source": source, "files": files}


def build_evidence_and_benchmark(
    workspace: Path,
    train: list[SensorWindow],
    test: list[SensorWindow],
    sample_rate: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    benchmark_dir = workspace / "data" / "benchmark"
    calibrator = EvidenceCalibrator.fit([row.sensor for row in train], sample_rate=sample_rate)
    extractor = EvidenceExtractor(calibrator, CHANNEL_NAMES, sample_rate=sample_rate)
    builder = SensorFactBenchmarkBuilder(seed=seed)

    train_cards = [extractor.extract(row).to_json_dict() for row in train]
    test_cards = [extractor.extract(row).to_json_dict() for row in test]
    train_records = [builder.build_record(EvidenceCard.from_json_dict(card)) for card in train_cards]
    test_records = [builder.build_record(EvidenceCard.from_json_dict(card)) for card in test_cards]

    write_jsonl(benchmark_dir / "ucihar_evidence_train.jsonl", train_cards)
    write_jsonl(benchmark_dir / "ucihar_evidence_test.jsonl", test_cards)
    write_jsonl(benchmark_dir / "ucihar_sensorfact_train.jsonl", train_records)
    write_jsonl(benchmark_dir / "ucihar_sensorfact_test.jsonl", test_records)
    write_jsonl(
        benchmark_dir / "ucihar_caption_selection_test.jsonl",
        [record["caption_selection"] for record in test_records],
    )
    write_jsonl(
        benchmark_dir / "ucihar_counterfactual_test.jsonl",
        [
            {
                "window_id": record["window_id"],
                "positive": record["positive"],
                "counterfactuals": record["counterfactuals"],
            }
            for record in test_records
        ],
    )
    write_jsonl(
        benchmark_dir / "ucihar_qa_test.jsonl",
        [{"window_id": record["window_id"], "qa": record["qa"]} for record in test_records],
    )
    write_json(benchmark_dir / "ucihar_evidence_calibrator.json", calibrator.to_json_dict())
    return train_records, test_records, {"calibrator": calibrator.to_json_dict()}


def run_experiments(
    workspace: Path,
    train: list[SensorWindow],
    test: list[SensorWindow],
    test_records: list[dict],
) -> dict[str, dict]:
    outputs = workspace / "outputs"
    checkpoints = workspace / "checkpoints"
    outputs.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    har_metrics, classifier = evaluate_har(train, test)
    write_json(outputs / "b0_har_metrics.json", har_metrics)
    write_checkpoint(checkpoints / "b0_har.json", classifier.to_json_dict())

    systems = {
        "B1": "statistics_prompt_only",
        "B2": "trend_only_alignment",
        "B3": "rich_evidence_alignment",
        "M1": "sensorfact_counterfactual_alignment",
    }
    results = {"B0": {"system": "HAR classifier", **har_metrics}}
    for system_id, name in systems.items():
        metrics = evaluate_grounding(test_records, system_id)
        metrics["system"] = name
        metrics["har_accuracy"] = har_metrics["har_accuracy"] if system_id in {"B2", "B3", "M1"} else None
        results[system_id] = metrics
        out_name = {
            "B1": "b1_prompt_only_metrics.json",
            "B2": "b2_trend_only_metrics.json",
            "B3": "b3_rich_facts_metrics.json",
            "M1": "m1_sensorfact_metrics.json",
        }[system_id]
        write_json(outputs / out_name, metrics)
        if system_id in {"B2", "B3", "M1"}:
            write_checkpoint(checkpoints / f"{system_id.lower()}_{name}.json", metrics)

    write_results_tables(outputs, results)
    write_pilot_report(outputs, results)
    return results


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_results_tables(outputs: Path, results: dict[str, dict]) -> None:
    headers = [
        "System",
        "HAR Acc",
        "Caption Sel Acc",
        "CF Reject Acc",
        "CF Reject F1",
        "QA Acc",
        "Notes",
    ]
    rows = []
    notes = {
        "B0": "HAR anchor only",
        "B1": "No training; evidence-statistics scorer",
        "B2": "Trend-only grounding scorer",
        "B3": "Rich evidence without counterfactual margin",
        "M1": "Counterfactual-aware SensorFact scorer",
    }
    for key in ["B0", "B1", "B2", "B3", "M1"]:
        item = results[key]
        rows.append(
            [
                key,
                _fmt(item.get("har_accuracy")),
                _fmt(item.get("caption_selection_accuracy")),
                _fmt(item.get("cf_reject_accuracy")),
                _fmt(item.get("cf_reject_f1")),
                _fmt(item.get("qa_accuracy")),
                notes[key],
            ]
        )
    markdown = _markdown_table(headers, rows)
    (outputs / "main_results_table.md").write_text(markdown, encoding="utf-8")
    (outputs / "baseline_results_table.md").write_text(
        _markdown_table(headers, rows[:3]), encoding="utf-8"
    )
    csv = ",".join(headers) + "\n" + "\n".join(",".join(row) for row in rows) + "\n"
    (outputs / "main_results_table.csv").write_text(csv, encoding="utf-8")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_pilot_report(outputs: Path, results: dict[str, dict]) -> None:
    b2 = results["B2"]
    b3 = results["B3"]
    m1 = results["M1"]
    m1_gain_b2 = (m1.get("cf_reject_f1") or 0.0) - (b2.get("cf_reject_f1") or 0.0)
    m1_gain_b3 = (m1.get("cf_reject_f1") or 0.0) - (b3.get("cf_reject_f1") or 0.0)
    qa_gain_b2 = (m1.get("qa_accuracy") or 0.0) - (b2.get("qa_accuracy") or 0.0)
    if m1_gain_b2 >= 0.05 and (m1_gain_b3 >= 0.0 or qa_gain_b2 >= 0.05):
        decision = "GO"
    elif m1_gain_b2 > 0.0 or qa_gain_b2 > 0.0:
        decision = "WEAK GO"
    else:
        decision = "NO-GO"

    text = f"""# SensorFact Pilot Report

## Decision

{decision}

## Main Signal

- M1 vs B2 counterfactual rejection F1 gain: {m1_gain_b2:.4f}
- M1 vs B3 counterfactual rejection F1 gain: {m1_gain_b3:.4f}
- M1 vs B2 QA accuracy gain: {qa_gain_b2:.4f}

## Interpretation

This report is generated by the pilot runner. Treat it as a first checkpoint, not a final EMNLP claim. If this run used synthetic data or a small subset, repeat on the full ModelScope UCI-HAR mirror and then add a second dataset before converting result slots into factual abstract claims.

## Required Next Step

Use `outputs/main_results_table.md` to update `docs/paper/SensorFact_detailed_abstract.md` only after confirming the run used the intended server data.
"""
    (outputs / "pilot_report.md").write_text(text, encoding="utf-8")


def run_pilot(
    workspace: Path,
    dataset_id: str = "OmniData/HAR",
    max_train_windows: int = 2000,
    max_test_windows: int = 1000,
    synthetic: bool = False,
    sample_rate: float = 50.0,
    seed: int = 42,
) -> dict[str, dict]:
    train, test, source_info = prepare_windows(
        workspace=workspace,
        dataset_id=dataset_id,
        max_train_windows=max_train_windows,
        max_test_windows=max_test_windows,
        synthetic=synthetic,
        sample_rate=sample_rate,
    )
    train_records, test_records, benchmark_info = build_evidence_and_benchmark(
        workspace=workspace,
        train=train,
        test=test,
        sample_rate=sample_rate,
        seed=seed,
    )
    results = run_experiments(workspace, train, test, test_records)
    write_json(
        workspace / "outputs" / "run_manifest.json",
        {
            "dataset_id": dataset_id,
            "source": source_info,
            "benchmark": benchmark_info,
            "train_records": len(train_records),
            "test_records": len(test_records),
            "synthetic": synthetic,
            "sample_rate": sample_rate,
            "seed": seed,
        },
    )
    return results


def load_benchmark_records(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))
