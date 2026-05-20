from __future__ import annotations

from collections.abc import Callable, Sequence
import random
from typing import TypeVar


T = TypeVar("T")


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap_ci(
    samples: Sequence[T],
    metric_fn: Callable[[list[T]], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rows = list(samples)
    if not rows:
        return 0.0, 0.0, 0.0
    point = float(metric_fn(rows))
    rng = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        resample = [rows[rng.randrange(len(rows))] for _ in rows]
        boot.append(float(metric_fn(resample)))
    lower = _percentile(boot, alpha / 2.0)
    upper = _percentile(boot, 1.0 - alpha / 2.0)
    return point, lower, upper


def paired_bootstrap_delta_ci(
    stronger: Sequence[T],
    weaker: Sequence[T],
    metric_fn: Callable[[list[T]], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    left = list(stronger)
    right = list(weaker)
    if not left or not right:
        return 0.0, 0.0, 0.0, 1.0
    if len(left) != len(right):
        raise ValueError("paired bootstrap requires equal-length sample lists")

    point = float(metric_fn(left) - metric_fn(right))
    rng = random.Random(seed)
    boot = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(len(left)) for _ in left]
        left_resample = [left[idx] for idx in indices]
        right_resample = [right[idx] for idx in indices]
        boot.append(float(metric_fn(left_resample) - metric_fn(right_resample)))
    lower = _percentile(boot, alpha / 2.0)
    upper = _percentile(boot, 1.0 - alpha / 2.0)
    if point >= 0.0:
        p_value = sum(1 for value in boot if value <= 0.0) / len(boot)
    else:
        p_value = sum(1 for value in boot if value >= 0.0) / len(boot)
    return point, lower, upper, float(p_value)
