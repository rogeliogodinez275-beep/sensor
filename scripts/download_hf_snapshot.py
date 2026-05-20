from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
except ImportError:  # pragma: no cover - optional dependency in local tests
    hf_snapshot_download = None

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
except ImportError:  # pragma: no cover - optional dependency in local tests
    ms_snapshot_download = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot to a local directory.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--resume-download", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--retry-delay-s", type=float, default=5.0)
    return parser.parse_args()


def missing_weight_files(local_dir: Path) -> list[str]:
    index_path = local_dir / "model.safetensors.index.json"
    single_path = local_dir / "model.safetensors"

    if index_path.exists():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        required = sorted(set(weight_map.values()))
        return [name for name in required if not (local_dir / name).exists()]

    if single_path.exists():
        return []

    return ["model.safetensors.index.json or model.safetensors"]


def download_snapshot(
    repo_id: str,
    local_dir: Path,
    resume_download: bool,
    max_attempts: int = 120,
    retry_delay_s: float = 5.0,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)

    last_path = local_dir
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        local_missing = missing_weight_files(local_dir)
        if local_missing and ms_snapshot_download is not None and (
            local_dir / "model.safetensors.index.json"
        ).exists():
            print(
                f"Local snapshot for {repo_id} is incomplete before remote check: "
                f"{local_missing}. Using ModelScope resume."
            )
            last_path = Path(ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir)))
        elif hf_snapshot_download is not None:
            try:
                last_path = Path(
                    hf_snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(local_dir),
                        resume_download=resume_download,
                    )
                )
                missing_after_hf = missing_weight_files(local_dir)
                if missing_after_hf and ms_snapshot_download is not None:
                    print(
                        f"Hugging Face returned incomplete local snapshot for {repo_id}: "
                        f"{missing_after_hf}. Falling back to ModelScope."
                    )
                    last_path = Path(ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir)))
            except Exception as exc:
                last_exc = exc
                if ms_snapshot_download is None:
                    raise RuntimeError("huggingface download failed and modelscope fallback is unavailable") from exc
                print(f"Hugging Face download failed for {repo_id}: {exc}. Falling back to ModelScope.")
                last_path = Path(ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir)))
        else:
            if ms_snapshot_download is None:
                raise RuntimeError("no model download backend available")
            last_path = Path(ms_snapshot_download(model_id=repo_id, local_dir=str(local_dir)))

        missing = missing_weight_files(local_dir)
        if not missing:
            return last_path

        print(f"Attempt {attempt}/{max_attempts}: still missing weight files for {repo_id}: {missing}")
        if attempt < max_attempts:
            time.sleep(retry_delay_s)

    raise RuntimeError(
        f"download incomplete for {repo_id}; missing files: {missing_weight_files(local_dir)}"
    ) from last_exc


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    path = download_snapshot(
        repo_id=args.repo_id,
        local_dir=local_dir,
        resume_download=args.resume_download,
        max_attempts=args.max_attempts,
        retry_delay_s=args.retry_delay_s,
    )
    print(path)


if __name__ == "__main__":
    main()
