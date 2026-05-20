from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sensorfact.schemas import EvidenceCard, SensorWindow


def _as_2d(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D sensor window, got shape {arr.shape}")
    if arr.shape[0] < 4:
        raise ValueError("Sensor window is too short for evidence extraction")
    return arr


def _rms_energy(window: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(window))))


def _axis_energy(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(window), axis=0))


def _dominant_frequency(window: np.ndarray, sample_rate: float) -> tuple[float, float]:
    centered = window - np.mean(window, axis=0, keepdims=True)
    signal = centered[:, int(np.argmax(_axis_energy(centered)))]
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / sample_rate)
    if spectrum.shape[0] <= 1 or float(np.sum(spectrum[1:])) == 0.0:
        return 0.0, 0.0
    idx = int(np.argmax(spectrum[1:]) + 1)
    dominant = float(freqs[idx])
    ratio = float(spectrum[idx] / (np.sum(spectrum[1:]) + 1e-8))
    return dominant, ratio


def _autocorr_peak(window: np.ndarray) -> float:
    centered = window - np.mean(window, axis=0, keepdims=True)
    signal = centered[:, int(np.argmax(_axis_energy(centered)))]
    denom = float(np.dot(signal, signal))
    if denom <= 1e-8:
        return 0.0
    corr = np.correlate(signal, signal, mode="full")[signal.size - 1 :]
    corr = corr / (denom + 1e-8)
    min_lag = max(2, signal.size // 20)
    max_lag = max(min_lag + 1, signal.size // 2)
    return float(np.max(corr[min_lag:max_lag])) if max_lag > min_lag else 0.0


def _peak_count(signal: np.ndarray) -> int:
    if signal.size < 3:
        return 0
    threshold = float(np.mean(signal) + 0.75 * np.std(signal))
    middle = signal[1:-1]
    peaks = (middle > signal[:-2]) & (middle > signal[2:]) & (middle > threshold)
    return int(np.sum(peaks))


def _trend_segments(signal: np.ndarray, parts: int = 3) -> list[str]:
    chunks = np.array_split(signal, parts)
    trends: list[str] = []
    for chunk in chunks:
        if chunk.size < 2:
            trends.append("stable")
            continue
        slope = float(chunk[-1] - chunk[0])
        scale = float(np.std(signal) + 1e-6)
        if slope > 0.20 * scale:
            trends.append("rise")
        elif slope < -0.20 * scale:
            trends.append("fall")
        else:
            trends.append("stable")
    return trends


@dataclass
class EvidenceCalibrator:
    thresholds: dict[str, tuple[float, float]]
    sample_rate: float = 50.0

    @classmethod
    def fit(
        cls,
        windows: Iterable[np.ndarray],
        sample_rate: float = 50.0,
        low_quantile: float = 0.33,
        high_quantile: float = 0.67,
    ) -> "EvidenceCalibrator":
        arrays = [_as_2d(window) for window in windows]
        if not arrays:
            raise ValueError("Cannot fit EvidenceCalibrator on an empty window list")
        rms_values = np.asarray([_rms_energy(window) for window in arrays], dtype=np.float32)
        burst_values = np.asarray(
            [_peak_count(np.linalg.norm(window, axis=1)) for window in arrays],
            dtype=np.float32,
        )
        freq_values = np.asarray(
            [_dominant_frequency(window, sample_rate)[0] for window in arrays],
            dtype=np.float32,
        )
        return cls(
            thresholds={
                "rms_energy": tuple(
                    float(x) for x in np.quantile(rms_values, [low_quantile, high_quantile])
                ),
                "peak_count": tuple(
                    float(x) for x in np.quantile(burst_values, [low_quantile, high_quantile])
                ),
                "dominant_frequency": tuple(
                    float(x) for x in np.quantile(freq_values, [low_quantile, high_quantile])
                ),
            },
            sample_rate=sample_rate,
        )

    def bucket(self, name: str, value: float, labels: tuple[str, str, str]) -> str:
        low, high = self.thresholds[name]
        if value <= low:
            return labels[0]
        if value >= high:
            return labels[2]
        return labels[1]

    def to_json_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "thresholds": {k: [float(v[0]), float(v[1])] for k, v in self.thresholds.items()},
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "EvidenceCalibrator":
        return cls(
            thresholds={k: (float(v[0]), float(v[1])) for k, v in data["thresholds"].items()},
            sample_rate=float(data.get("sample_rate", 50.0)),
        )


class EvidenceExtractor:
    def __init__(
        self,
        calibrator: EvidenceCalibrator,
        channel_names: list[str],
        sample_rate: float | None = None,
        dominant_axis_margin: float = 0.10,
    ) -> None:
        self.calibrator = calibrator
        self.channel_names = channel_names
        self.sample_rate = float(sample_rate or calibrator.sample_rate)
        self.dominant_axis_margin = dominant_axis_margin

    def extract(
        self,
        window: np.ndarray | SensorWindow,
        window_id: str | None = None,
        dataset_id: str | None = None,
        label: str | None = None,
        subject_id: str | None = None,
    ) -> EvidenceCard:
        if isinstance(window, SensorWindow):
            sensor = _as_2d(window.sensor)
            window_id = window.window_id
            dataset_id = window.dataset_id
            label = window.label
            subject_id = window.subject_id
        else:
            sensor = _as_2d(window)
        if sensor.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Expected {len(self.channel_names)} channels, got {sensor.shape[1]}"
            )

        rms = _rms_energy(sensor)
        axis_energy = _axis_energy(sensor)
        sorted_energy = np.sort(axis_energy)[::-1]
        top_idx = int(np.argmax(axis_energy))
        if len(sorted_energy) > 1 and sorted_energy[1] >= sorted_energy[0] * (
            1.0 - self.dominant_axis_margin
        ):
            dominant_axis = "uncertain"
        else:
            dominant_axis = self.channel_names[top_idx]

        dom_freq, fft_ratio = _dominant_frequency(sensor, self.sample_rate)
        autocorr = _autocorr_peak(sensor)
        periodic_score = max(autocorr, fft_ratio)
        if periodic_score >= 0.55:
            periodicity = "strong"
        elif periodic_score >= 0.25:
            periodicity = "weak"
        else:
            periodicity = "none"

        magnitude = np.linalg.norm(sensor, axis=1)
        peaks = _peak_count(magnitude)
        burstiness = "bursty" if peaks > self.calibrator.thresholds["peak_count"][1] else "smooth"

        centered = sensor - np.mean(sensor, axis=0, keepdims=True)
        active = np.std(centered, axis=0) > 1e-8
        corr = np.corrcoef(centered[:, active], rowvar=False) if int(np.sum(active)) > 1 else None
        relation = "no clear relation"
        if corr is not None and corr.ndim == 2 and corr.shape[0] > 1 and np.all(np.isfinite(corr)):
            active_names = [name for name, keep in zip(self.channel_names, active) if keep]
            pairs: list[tuple[float, int, int]] = []
            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    pairs.append((float(corr[i, j]), i, j))
            if pairs:
                corr_value, i, j = max(pairs, key=lambda item: abs(item[0]))
                if abs(corr_value) >= 0.45:
                    direction = "move synchronously" if corr_value > 0 else "move oppositely"
                    relation = f"{active_names[i]} and {active_names[j]} {direction}"

        trend_axis = sensor[:, top_idx]
        trend_segments = _trend_segments(trend_axis)
        intensity = self.calibrator.bucket("rms_energy", rms, ("low", "medium", "high"))
        if dom_freq <= 1e-8 or periodicity == "none":
            freq_bucket = "uncertain"
        else:
            freq_bucket = self.calibrator.bucket(
                "dominant_frequency", dom_freq, ("low", "mid", "high")
            )

        uncertain_count = sum(
            [
                dominant_axis == "uncertain",
                periodicity == "none",
                freq_bucket == "uncertain",
                relation == "no clear relation",
            ]
        )
        confidence = max(0.1, 1.0 - 0.15 * uncertain_count)

        return EvidenceCard(
            window_id=window_id or "unknown_window",
            dataset_id=dataset_id or "unknown_dataset",
            label=label,
            subject_id=subject_id,
            intensity=intensity,
            periodicity=periodicity,
            dominant_axis=dominant_axis,
            dominant_frequency=freq_bucket,
            cross_channel_relation=relation,
            lead_lag="no clear lag",
            burstiness=burstiness,
            trend_segments=trend_segments,
            confidence=confidence,
            numeric={
                "rms_energy": rms,
                "autocorr_peak": autocorr,
                "fft_dominant_ratio": fft_ratio,
                "dominant_frequency_hz": dom_freq,
                "peak_count": float(peaks),
            },
        )
