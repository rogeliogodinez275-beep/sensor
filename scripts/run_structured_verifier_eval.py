from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sensorfact.structured_verifier_eval import run_structured_verifier_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run structured evidence verifier evaluation.")
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--benchmark-path", default="data/benchmark/ucihar_sensorfact_hard_v3_test.jsonl")
    parser.add_argument("--output-metrics", default="outputs/structured_verifier_metrics.json")
    parser.add_argument("--output-rows", default="outputs/structured_verifier_rows.jsonl")
    parser.add_argument(
        "--parser-mode",
        choices=["regex_evidence", "model_evidence"],
        default="regex_evidence",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["strict_json", "terse", "chain_then_json", "fewshot_json"],
        default="strict_json",
    )
    parser.add_argument("--model-dir", default="models/Qwen_Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-records", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = Path(args.workspace).resolve()
    metrics = run_structured_verifier_eval(
        benchmark_path=workspace / args.benchmark_path,
        output_metrics_path=workspace / args.output_metrics,
        output_rows_path=workspace / args.output_rows,
        parser_mode=args.parser_mode,
        model_dir=workspace / args.model_dir if args.parser_mode == "model_evidence" else None,
        max_records=args.max_records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        prompt_style=args.prompt_style,
    )
    print("Structured verifier evaluation finished.")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
