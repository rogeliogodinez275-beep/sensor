from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    files: list[str | Path],
    experiment_name: str,
    model_id: str | None = None,
    seed: int | None = None,
    workspace: str | Path | None = None,
) -> dict:
    root = Path(workspace).resolve() if workspace is not None else Path.cwd().resolve()
    rows = []
    for item in files:
        path = Path(item)
        resolved = path.resolve()
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "experiment_name": experiment_name,
        "model_id": model_id,
        "seed": seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(root),
        "git_commit": _git_commit(root),
        "files": rows,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility manifest with file hashes.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workspace", default=".")
    parser.add_argument("files", nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        files=[Path(item) for item in args.files],
        experiment_name=args.experiment_name,
        model_id=args.model_id,
        seed=args.seed,
        workspace=args.workspace,
    )
    write_manifest(args.output, manifest)
    print(f"Wrote manifest: {args.output}")
    print(f"files: {len(manifest['files'])}")


if __name__ == "__main__":
    main()
