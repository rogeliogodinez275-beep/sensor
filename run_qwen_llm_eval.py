from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.qwen_llm_eval import run_qwen_llm_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Qwen LLM grounding evaluation.")
    parser.add_argument("--workspace", default=".", help="Project workspace root.")
    parser.add_argument("--model-dir", default="models/Qwen_Qwen3-4B-Instruct-2507")
    parser.add_argument("--benchmark-path", default="data/benchmark/ucihar_sensorfact_test.jsonl")
    parser.add_argument("--output-metrics", default="outputs/qwen_llm_metrics.json")
    parser.add_argument("--output-rows", default="outputs/qwen_llm_rows.jsonl")
    parser.add_argument("--max-records", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt-style",
        choices=["strict_json", "terse", "chain_then_json"],
        default="strict_json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_qwen_llm_eval(
        benchmark_path=workspace / args.benchmark_path,
        model_dir=workspace / args.model_dir,
        output_metrics_path=workspace / args.output_metrics,
        output_rows_path=workspace / args.output_rows,
        max_records=args.max_records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        prompt_style=args.prompt_style,
    )
    print("Qwen LLM evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
