from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np

from sensorfact.schemas import EvidenceCard, SensorWindow


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def sensor_features(window: SensorWindow | np.ndarray) -> np.ndarray:
    sensor = window.sensor if isinstance(window, SensorWindow) else np.asarray(window, dtype=np.float32)
    sensor = np.asarray(sensor, dtype=np.float32)
    axis_energy = np.sqrt(np.mean(np.square(sensor), axis=0))
    magnitude = np.linalg.norm(sensor, axis=1)
    stats = [
        np.mean(sensor, axis=0),
        np.std(sensor, axis=0),
        np.min(sensor, axis=0),
        np.max(sensor, axis=0),
        axis_energy,
        np.asarray(
            [
                float(np.mean(magnitude)),
                float(np.std(magnitude)),
                float(np.max(magnitude) - np.min(magnitude)),
            ],
            dtype=np.float32,
        ),
    ]
    return np.concatenate([np.ravel(item) for item in stats]).astype(np.float32)


def normalize_matrix(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (features - mean) / std, mean, std


def apply_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.array(std, copy=True)
    safe_std[safe_std < 1e-6] = 1.0
    return (features - mean) / safe_std


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def hashed_text_features(text: str, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    counts = Counter(tokenize(text))
    for token, count in counts.items():
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        vec[idx] += float(count)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def evidence_feature_tokens(card: EvidenceCard, mode: str) -> set[str]:
    tokens: set[str] = set()
    if mode in {"rich", "sensorfact"}:
        tokens.update(
            [
                f"intensity:{card.intensity}",
                f"periodicity:{card.periodicity}",
                f"axis:{card.dominant_axis}",
                f"frequency:{card.dominant_frequency}",
                f"burstiness:{card.burstiness}",
            ]
        )
    if mode in {"trend", "rich", "sensorfact"}:
        tokens.update(f"trend:{item}" for item in card.trend_segments)
    return tokens
