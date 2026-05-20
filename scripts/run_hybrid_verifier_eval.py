from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.hybrid_verifier_eval import run_hybrid_verifier_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid structured verifier + direct LLM fallback evaluation.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--structured-rows", required=True)
    parser.add_argument("--direct-rows", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--structured-threshold", type=float, default=0.5)
    parser.add_argument("--system-name", default="hybrid_structured_direct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_hybrid_verifier_eval(
        structured_rows_path=workspace / args.structured_rows,
        direct_rows_path=workspace / args.direct_rows,
        output_metrics_path=workspace / args.output_metrics,
        output_rows_path=workspace / args.output_rows,
        structured_threshold=args.structured_threshold,
        system_name=args.system_name,
    )
    print("Hybrid verifier evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
