from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.benchmark import SensorFactBenchmarkBuilder
from sensorfact.data import (
    MHEALTH_CHANNEL_NAMES,
    MHEALTH_URL,
    download_mhealth_dataset,
    find_mhealth_log_files,
    load_mhealth,
)
from sensorfact.evidence import EvidenceCalibrator, EvidenceExtractor
from sensorfact.hard_benchmark import build_hard_benchmark
from sensorfact.io import write_json, write_jsonl
from sensorfact.schemas import EvidenceCard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MHEALTH SensorFact benchmarks.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--raw-dir", default="data/raw/mhealth")
    parser.add_argument("--url", default=MHEALTH_URL)
    parser.add_argument("--download", action="store_true", help="Download MHEALTH if subject logs are absent.")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--test-subject-ratio", type=float, default=0.2)
    parser.add_argument("--max-train-windows", type=int, default=-1)
    parser.add_argument("--max-test-windows", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hard-variants", default="v2,v3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    raw_dir = workspace / args.raw_dir
    if not find_mhealth_log_files(raw_dir):
        if not args.download:
            raise FileNotFoundError(f"No MHEALTH logs found under {raw_dir}")
        download_mhealth_dataset(raw_dir, url=args.url)

    train, test = load_mhealth(
        raw_dir,
        window_size=args.window_size,
        stride=args.stride,
        test_subject_ratio=args.test_subject_ratio,
        seed=args.seed,
        max_train_windows=None if args.max_train_windows < 0 else args.max_train_windows,
        max_test_windows=None if args.max_test_windows < 0 else args.max_test_windows,
    )
    if not train or not test:
        raise RuntimeError(f"MHEALTH produced train={len(train)} test={len(test)} windows")

    processed_dir = workspace / "data" / "processed"
    benchmark_dir = workspace / "data" / "benchmark"
    write_jsonl(processed_dir / "mhealth_train.jsonl", [row.to_json_dict() for row in train])
    write_jsonl(processed_dir / "mhealth_test.jsonl", [row.to_json_dict() for row in test])
    write_json(
        processed_dir / "mhealth_metadata.json",
        {
            "dataset_id": "mhealth",
            "train_windows": len(train),
            "test_windows": len(test),
            "channels": MHEALTH_CHANNEL_NAMES,
            "window_size": args.window_size,
            "stride": args.stride,
            "sample_rate": args.sample_rate,
            "test_subject_ratio": args.test_subject_ratio,
        },
    )

    calibrator = EvidenceCalibrator.fit([row.sensor for row in train], sample_rate=args.sample_rate)
    extractor = EvidenceExtractor(
        calibrator=calibrator,
        channel_names=MHEALTH_CHANNEL_NAMES,
        sample_rate=args.sample_rate,
    )
    builder = SensorFactBenchmarkBuilder(seed=args.seed)
    train_cards = [extractor.extract(row).to_json_dict() for row in train]
    test_cards = [extractor.extract(row).to_json_dict() for row in test]
    train_records = [builder.build_record(EvidenceCard.from_json_dict(card)) for card in train_cards]
    test_records = [builder.build_record(EvidenceCard.from_json_dict(card)) for card in test_cards]

    base_test_path = benchmark_dir / "mhealth_sensorfact_test.jsonl"
    write_jsonl(benchmark_dir / "mhealth_evidence_train.jsonl", train_cards)
    write_jsonl(benchmark_dir / "mhealth_evidence_test.jsonl", test_cards)
    write_jsonl(benchmark_dir / "mhealth_sensorfact_train.jsonl", train_records)
    write_jsonl(base_test_path, test_records)
    write_json(benchmark_dir / "mhealth_evidence_calibrator.json", calibrator.to_json_dict())

    variants = [item.strip() for item in args.hard_variants.split(",") if item.strip()]
    for variant in variants:
        output_path = benchmark_dir / f"mhealth_sensorfact_hard_{variant}_test.jsonl"
        count = build_hard_benchmark(
            input_path=base_test_path,
            output_path=output_path,
            seed=args.seed,
            max_records=-1,
            variant=variant,
        )
        print(f"Wrote {count} MHEALTH hard-{variant} records to {output_path}")

    print(
        "MHEALTH benchmark finished: "
        f"train={len(train)} test={len(test)} channels={','.join(MHEALTH_CHANNEL_NAMES)}"
    )


if __name__ == "__main__":
    main()
