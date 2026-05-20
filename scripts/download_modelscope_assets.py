from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modelscope.hub.snapshot_download import dataset_snapshot_download, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SensorFact ModelScope assets.")
    parser.add_argument("--dataset-id", default="OmniData/HAR")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    model_root = Path(args.model_dir)
    if not args.skip_dataset:
        dataset_dir = data_root / args.dataset_id.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_snapshot_download(args.dataset_id, local_dir=str(dataset_dir))
        print(f"Dataset downloaded to: {dataset_path}")
    if not args.skip_models:
        for model_id in [args.embedding_model, args.llm_model]:
            target = model_root / model_id.replace("/", "_")
            target.mkdir(parents=True, exist_ok=True)
            model_path = snapshot_download(model_id=model_id, local_dir=str(target))
            print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()
