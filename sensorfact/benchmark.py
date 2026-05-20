from __future__ import annotations

import random
from dataclasses import replace

from sensorfact.schemas import EvidenceCard


def evidence_to_text(card: EvidenceCard, mode: str = "rich") -> str:
    if mode == "trend":
        trend = ", ".join(card.trend_segments) if card.trend_segments else "stable"
        return f"The sensor signal follows a {trend} trend pattern."
    parts = [
        f"The motion intensity is {card.intensity}.",
        f"The signal shows {card.periodicity} periodicity.",
        f"The dominant movement axis is {card.dominant_axis}.",
        f"The dominant frequency is {card.dominant_frequency}.",
        f"The movement is {card.burstiness}.",
    ]
    if card.cross_channel_relation != "no clear relation":
        parts.append(f"{card.cross_channel_relation}.")
    trend = ", ".join(card.trend_segments) if card.trend_segments else "stable"
    parts.append(f"The main trend pattern is {trend}.")
    return " ".join(parts)


class SensorFactBenchmarkBuilder:
    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def build_record(self, card: EvidenceCard) -> dict:
        positive = self._caption(card, supported=True, changed_fact=None)
        counterfactuals = self._counterfactuals(card)
        candidates = [positive] + counterfactuals[:3]
        self.random.shuffle(candidates)
        answer_index = next(i for i, item in enumerate(candidates) if item["supported"])
        return {
            "window_id": card.window_id,
            "dataset_id": card.dataset_id,
            "label": card.label,
            "subject_id": card.subject_id,
            "evidence": card.to_json_dict(),
            "positive": positive,
            "paraphrases": self._paraphrases(card),
            "counterfactuals": counterfactuals,
            "caption_selection": {
                "window_id": card.window_id,
                "candidates": candidates,
                "answer_index": answer_index,
            },
            "qa": self._qa(card),
        }

    def _caption(
        self,
        card: EvidenceCard,
        supported: bool,
        changed_fact: str | None,
    ) -> dict:
        text = evidence_to_text(card)
        return {
            "text": text,
            "supported": supported,
            "changed_fact": changed_fact,
        }

    def _paraphrases(self, card: EvidenceCard) -> list[dict]:
        trend = ", ".join(card.trend_segments) if card.trend_segments else "stable"
        return [
            {
                "text": (
                    f"This window contains {card.intensity} motion, {card.periodicity} rhythm, "
                    f"and strongest activity on {card.dominant_axis}."
                ),
                "supported": True,
            },
            {
                "text": (
                    f"The sensor evidence points to a {card.burstiness} movement pattern "
                    f"with {card.dominant_frequency} frequency and a {trend} trend."
                ),
                "supported": True,
            },
        ]

    def _counterfactuals(self, card: EvidenceCard) -> list[dict]:
        edits = [
            ("intensity", {"low": "high", "medium": "low", "high": "low"}.get(card.intensity, "high")),
            (
                "periodicity",
                {"none": "strong", "weak": "none", "strong": "none"}.get(card.periodicity, "strong"),
            ),
            (
                "dominant_axis",
                self._alternate(card.dominant_axis, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]),
            ),
            (
                "dominant_frequency",
                {"low": "high", "mid": "low", "high": "low", "uncertain": "mid"}.get(
                    card.dominant_frequency, "mid"
                ),
            ),
            ("burstiness", "smooth" if card.burstiness == "bursty" else "bursty"),
        ]
        rows: list[dict] = []
        for field, value in edits:
            changed = replace(card, **{field: value})
            rows.append(self._caption(changed, supported=False, changed_fact=field))
        return rows

    def _alternate(self, current: str, options: list[str]) -> str:
        for option in options:
            if option != current:
                return option
        return options[0]

    def _qa(self, card: EvidenceCard) -> list[dict]:
        return [
            {
                "question": "Which sensor axis carries the strongest movement?",
                "answer": card.dominant_axis,
                "fact": "dominant_axis",
            },
            {
                "question": "Is the motion evidence periodic?",
                "answer": card.periodicity,
                "fact": "periodicity",
            },
            {
                "question": "What intensity level is supported by the sensor evidence?",
                "answer": card.intensity,
                "fact": "intensity",
            },
            {
                "question": "Which statement type should be rejected if it contradicts one evidence field?",
                "answer": "counterfactual",
                "fact": "task_understanding",
            },
        ]
