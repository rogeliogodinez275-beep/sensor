from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.hard_benchmark import build_hard_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build natural-language hard SensorFact benchmark.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--input", default="data/benchmark/ucihar_sensorfact_test.jsonl")
    parser.add_argument("--output", default="data/benchmark/ucihar_sensorfact_hard_test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], default="v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    count = build_hard_benchmark(
        input_path=workspace / args.input,
        output_path=workspace / args.output,
        seed=args.seed,
        max_records=args.max_records,
        variant=args.variant,
    )
    print(f"Wrote {count} hard benchmark records to {workspace / args.output}")


if __name__ == "__main__":
    main()
