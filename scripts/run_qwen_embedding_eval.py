from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.qwen_eval import run_qwen_embedding_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen embedding grounding evaluation.")
    parser.add_argument("--workspace", default=".", help="Project workspace root.")
    parser.add_argument("--model-dir", default="models/Qwen_Qwen3-Embedding-0.6B")
    parser.add_argument("--benchmark-path", default="data/benchmark/ucihar_sensorfact_test.jsonl")
    parser.add_argument("--output-metrics", default="outputs/qwen_embedding_metrics.json")
    parser.add_argument("--output-scores", default="outputs/qwen_embedding_scores.jsonl")
    parser.add_argument("--max-records", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pairwise-margin", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_qwen_embedding_eval(
        benchmark_path=workspace / args.benchmark_path,
        model_dir=workspace / args.model_dir,
        output_metrics_path=workspace / args.output_metrics,
        output_scores_path=workspace / args.output_scores,
        max_records=args.max_records,
        batch_size=args.batch_size,
        device=args.device,
        threshold=args.threshold,
        pairwise_margin=args.pairwise_margin,
    )
    print("Qwen embedding evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
