from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.sampling import sample_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic random JSONL sample.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = sample_jsonl(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(f"Wrote {count} sampled rows to {args.output}")


if __name__ == "__main__":
    main()
