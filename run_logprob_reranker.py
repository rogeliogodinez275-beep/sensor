from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.logprob_reranker import run_logprob_reranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run yes/no log-prob caption reranking.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--model-dir", default="models/Qwen_Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--benchmark-path", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--output-rows", required=True)
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_logprob_reranker(
        benchmark_path=workspace / args.benchmark_path,
        model_dir=workspace / args.model_dir,
        output_metrics_path=workspace / args.output_metrics,
        output_rows_path=workspace / args.output_rows,
        max_records=args.max_records,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Log-prob reranker evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
