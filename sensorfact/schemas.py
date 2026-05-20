from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorWindow:
    window_id: str
    dataset_id: str
    split: str
    sensor: np.ndarray
    label: str | None = None
    subject_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "dataset_id": self.dataset_id,
            "split": self.split,
            "label": self.label,
            "subject_id": self.subject_id,
            "sensor": self.sensor.astype(float).tolist(),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "SensorWindow":
        return cls(
            window_id=str(data["window_id"]),
            dataset_id=str(data.get("dataset_id", "unknown")),
            split=str(data.get("split", "unknown")),
            label=data.get("label"),
            subject_id=None if data.get("subject_id") is None else str(data.get("subject_id")),
            sensor=np.asarray(data["sensor"], dtype=np.float32),
        )


@dataclass
class EvidenceCard:
    window_id: str
    dataset_id: str
    intensity: str
    periodicity: str
    dominant_axis: str
    dominant_frequency: str
    cross_channel_relation: str
    lead_lag: str
    burstiness: str
    trend_segments: list[str]
    confidence: float = 1.0
    label: str | None = None
    subject_id: str | None = None
    numeric: dict[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = float(self.confidence)
        data["numeric"] = {k: float(v) for k, v in self.numeric.items()}
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "EvidenceCard":
        return cls(
            window_id=str(data["window_id"]),
            dataset_id=str(data.get("dataset_id", "unknown")),
            label=data.get("label"),
            subject_id=None if data.get("subject_id") is None else str(data.get("subject_id")),
            intensity=str(data["intensity"]),
            periodicity=str(data["periodicity"]),
            dominant_axis=str(data["dominant_axis"]),
            dominant_frequency=str(data["dominant_frequency"]),
            cross_channel_relation=str(data.get("cross_channel_relation", "no clear relation")),
            lead_lag=str(data.get("lead_lag", "no clear lag")),
            burstiness=str(data.get("burstiness", "smooth")),
            trend_segments=[str(item) for item in data.get("trend_segments", [])],
            confidence=float(data.get("confidence", 1.0)),
            numeric={str(k): float(v) for k, v in data.get("numeric", {}).items()},
        )
