from __future__ import annotations

import random
from pathlib import Path

from sensorfact.io import ensure_parent


def sample_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    sample_size: int,
    seed: int = 42,
) -> int:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    source = Path(input_path)
    rows = source.read_text(encoding="utf-8").splitlines()
    non_empty = [row for row in rows if row.strip()]
    if sample_size >= len(non_empty):
        selected = non_empty
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(non_empty)), sample_size))
        selected = [non_empty[idx] for idx in indices]
    out = ensure_parent(output_path)
    out.write_text("".join(f"{row}\n" for row in selected), encoding="utf-8")
    return len(selected)
