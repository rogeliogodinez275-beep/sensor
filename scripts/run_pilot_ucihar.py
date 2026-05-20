from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.pipeline import run_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SensorFact UCI-HAR pilot.")
    parser.add_argument("--workspace", default=".", help="Project workspace root.")
    parser.add_argument("--dataset-id", default="OmniData/HAR", help="ModelScope dataset id.")
    parser.add_argument("--max-train-windows", type=int, default=2000)
    parser.add_argument("--max-test-windows", type=int, default=1000)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic UCI-HAR-like windows instead of downloading ModelScope data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pilot(
        workspace=Path(args.workspace).resolve(),
        dataset_id=args.dataset_id,
        max_train_windows=args.max_train_windows,
        max_test_windows=args.max_test_windows,
        synthetic=args.synthetic,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    print("SensorFact pilot finished.")
    for system_id, metrics in results.items():
        print(system_id, metrics)


if __name__ == "__main__":
    main()
