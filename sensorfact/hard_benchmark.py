from __future__ import annotations

import random
from pathlib import Path

from sensorfact.io import read_jsonl, write_jsonl
from sensorfact.schemas import EvidenceCard
from sensorfact.axis_vocab import AXIS_PHRASES, axis_phrase, dataset_axis_options


INTENSITY_PHRASES = {
    "low": "small-amplitude movement",
    "medium": "moderate-amplitude movement",
    "high": "large-amplitude movement",
}

PERIODICITY_PHRASES = {
    "none": "little repeatable rhythm",
    "weak": "an irregular but noticeable rhythm",
    "strong": "a clearly repeated rhythm",
}

FREQUENCY_PHRASES = {
    "low": "a slow cadence",
    "mid": "a mid-paced cadence",
    "high": "a quick cadence",
    "uncertain": "no stable cadence estimate",
}

BURST_PHRASES = {
    "smooth": "changes remain fairly even rather than spiky",
    "bursty": "the trace contains short, spiky bursts",
}


def _alternate(current: str, options: list[str]) -> str:
    for option in options:
        if option != current:
            return option
    return options[0]


def _phrase(mapping: dict[str, str], key: str) -> str:
    return mapping.get(key, str(key).replace("_", " "))


def _trend_phrase(trends: list[str]) -> str:
    if not trends:
        return "the segments stay mostly level"
    names = {
        "rise": "rising",
        "fall": "falling",
        "stable": "steady",
    }
    return ", then ".join(names.get(item, item) for item in trends)


def _numeric(card: EvidenceCard, name: str, default: float = 0.0) -> float:
    return float(card.numeric.get(name, default))


def _band_hint(value: str) -> str:
    return {
        "low": "below the lower reference band",
        "medium": "inside the middle reference band",
        "high": "above the upper reference band",
        "none": "below the repeatability reference",
        "weak": "near the repeatability reference",
        "strong": "above the repeatability reference",
        "uncertain": "without a stable reference crossing",
    }.get(value, "outside the expected reference band")


def _axis_evidence_clause(axis: str) -> str:
    if axis == "uncertain":
        return "No single channel clearly dominates the activity"
    return f"The clearest activity is in the {axis_phrase(axis)}"


def _axis_positive_clause(axis: str) -> str:
    if axis == "uncertain":
        return "without a single clearly dominant channel"
    return f"strongest evidence from the {axis_phrase(axis)}"


def _v3_axis_clause(axis: str) -> str:
    if axis == "uncertain":
        return "energy is spread across channels without a clear leader"
    return f"the largest per-axis energy is on the {axis_phrase(axis)}"


def _v3_property_clause(field: str, value: str) -> str:
    if field == "intensity":
        return {
            "low": "subtle energy",
            "medium": "moderate energy",
            "high": "forceful energy",
        }.get(value, f"{value} energy")
    if field == "periodicity":
        return {
            "none": "little repeatable rhythm",
            "weak": "a loose rhythm",
            "strong": "a repeatable rhythm",
        }.get(value, f"{value} rhythm")
    if field == "dominant_axis":
        return _v3_axis_clause(value)
    if field == "dominant_frequency":
        return {
            "low": "slow cadence",
            "mid": "middle-rate cadence",
            "high": "fast cadence",
            "uncertain": "no stable cadence",
        }.get(value, f"{value} cadence")
    if field == "burstiness":
        return {
            "smooth": "an even profile",
            "bursty": "short spikes",
        }.get(value, value)
    return str(value).replace("_", " ")


def _numeric_evidence_text(card: EvidenceCard) -> str:
    relation = "" if card.cross_channel_relation == "no clear relation" else f" Pairwise channel check: {card.cross_channel_relation}."
    return (
        f"RMS energy is {_numeric(card, 'rms_energy'):.3f}, {_band_hint(card.intensity)}. "
        f"The repeatability scores are autocorrelation {_numeric(card, 'autocorr_peak'):.3f} "
        f"and FFT concentration {_numeric(card, 'fft_dominant_ratio'):.3f}, {_band_hint(card.periodicity)}. "
        f"Axis comparison: {_v3_axis_clause(card.dominant_axis)}. "
        f"The dominant spectral peak is {_numeric(card, 'dominant_frequency_hz'):.3f} Hz, "
        f"{_band_hint(card.dominant_frequency)}. Peak count is {_numeric(card, 'peak_count'):.0f}, "
        f"with the profile {_band_hint(card.burstiness)} for burstiness. "
        f"Segment slopes move {_trend_phrase(card.trend_segments)}.{relation}"
    )


def hard_evidence_text(card: EvidenceCard, variant: str = "v1") -> str:
    if variant == "v3":
        return _numeric_evidence_text(card)
    relation = "" if card.cross_channel_relation == "no clear relation" else f" A channel relation is visible: {card.cross_channel_relation}."
    return (
        f"The window shows {_phrase(INTENSITY_PHRASES, card.intensity)} with "
        f"{_phrase(PERIODICITY_PHRASES, card.periodicity)}. {_axis_evidence_clause(card.dominant_axis)}, "
        f"and the temporal pattern suggests {_phrase(FREQUENCY_PHRASES, card.dominant_frequency)}. "
        f"Across the window, {_phrase(BURST_PHRASES, card.burstiness)}. The coarse segment shape is "
        f"{_trend_phrase(card.trend_segments)}.{relation}"
    )


def _positive_text(card: EvidenceCard, variant: str = "v1") -> str:
    if variant == "v3":
        return (
            f"Overall, the motion shows {_v3_property_clause('intensity', card.intensity)}, "
            f"{_v3_property_clause('periodicity', card.periodicity)}, "
            f"{_v3_property_clause('dominant_axis', card.dominant_axis)}, "
            f"{_v3_property_clause('dominant_frequency', card.dominant_frequency)}, and "
            f"{_v3_property_clause('burstiness', card.burstiness)}."
        )
    if variant == "v2":
        return (
            f"The movement has {_phrase(INTENSITY_PHRASES, card.intensity)} and "
            f"{_phrase(PERIODICITY_PHRASES, card.periodicity)}, {_axis_positive_clause(card.dominant_axis)}, "
            f"with {_phrase(FREQUENCY_PHRASES, card.dominant_frequency)}."
        )
    return (
        f"A supported description is {_phrase(INTENSITY_PHRASES, card.intensity)} with "
        f"{_phrase(PERIODICITY_PHRASES, card.periodicity)}, {_axis_positive_clause(card.dominant_axis)}, "
        f"and {_phrase(FREQUENCY_PHRASES, card.dominant_frequency)}."
    )


def _counterfactual_text(card: EvidenceCard, field: str, value: str, variant: str = "v1") -> str:
    changed = {
        "intensity": _phrase(INTENSITY_PHRASES, value),
        "periodicity": _phrase(PERIODICITY_PHRASES, value),
        "dominant_axis": axis_phrase(value),
        "dominant_frequency": _phrase(FREQUENCY_PHRASES, value),
        "burstiness": _phrase(BURST_PHRASES, value),
    }
    if variant == "v2":
        if field == "intensity":
            return f"The movement has {changed[field]} while following the same rhythm and main channel."
        if field == "periodicity":
            return f"The trace shows {changed[field]} with otherwise similar amplitude and channel evidence."
        if field == "dominant_axis":
            return f"The strongest activity comes from the {changed[field]}, with the same cadence and amplitude."
        if field == "dominant_frequency":
            return f"The motion keeps the same channel and amplitude while showing {changed[field]}."
        return f"The trace contains short, spiky bursts while the other properties remain similar."
    if field == "intensity":
        return (
            f"This description claims {changed[field]}, while keeping the same rhythm and main channel."
        )
    if field == "periodicity":
        return (
            f"This description says the trace has {changed[field]} and otherwise matches the same amplitude and channel evidence."
        )
    if field == "dominant_axis":
        return (
            f"This description shifts the strongest activity to the {changed[field]}, while leaving the cadence and amplitude unchanged."
        )
    if field == "dominant_frequency":
        return (
            f"This description keeps the same channel and amplitude but says the motion has {changed[field]}."
        )
    return f"This description says {changed[field]}, despite the rest of the evidence being unchanged."


def _counterfactual_v3_text(card: EvidenceCard, edits: list[tuple[str, str]]) -> str:
    changed = {field: value for field, value in edits}
    clauses = [
        _v3_property_clause("intensity", changed.get("intensity", card.intensity)),
        _v3_property_clause("periodicity", changed.get("periodicity", card.periodicity)),
        _v3_property_clause("dominant_axis", changed.get("dominant_axis", card.dominant_axis)),
        _v3_property_clause("dominant_frequency", changed.get("dominant_frequency", card.dominant_frequency)),
        _v3_property_clause("burstiness", changed.get("burstiness", card.burstiness)),
    ]
    return (
        f"Overall, the motion shows {clauses[0]}, {clauses[1]}, {clauses[2]}, "
        f"{clauses[3]}, and {clauses[4]}."
    )


def build_hard_record(base_record: dict, seed: int = 42, variant: str = "v1") -> dict:
    rng = random.Random(f"{seed}:{base_record['window_id']}")
    card = EvidenceCard.from_json_dict(base_record["evidence"])
    edits = [
        ("intensity", {"low": "high", "medium": "high", "high": "low"}.get(card.intensity, "high")),
        ("periodicity", {"none": "strong", "weak": "strong", "strong": "none"}.get(card.periodicity, "strong")),
        (
            "dominant_axis",
            _alternate(card.dominant_axis, dataset_axis_options(card.dataset_id)),
        ),
        ("dominant_frequency", {"low": "high", "mid": "low", "high": "low", "uncertain": "mid"}.get(card.dominant_frequency, "mid")),
        ("burstiness", "smooth" if card.burstiness == "bursty" else "bursty"),
    ]
    positive = {"text": _positive_text(card, variant=variant), "supported": True, "changed_fact": None}
    if variant == "v3":
        v3_specs = [
            [edits[0], edits[3]],
            [edits[1], edits[4]],
            [edits[2], edits[1]],
            [edits[4], edits[3]],
            [edits[0]],
        ]
        counterfactuals = [
            {
                "text": _counterfactual_v3_text(card, spec),
                "supported": False,
                "changed_fact": "+".join(field for field, _ in spec),
                "changed_facts": [field for field, _ in spec],
            }
            for spec in v3_specs
        ]
    else:
        counterfactuals = [
            {
                "text": _counterfactual_text(card, field, value, variant=variant),
                "supported": False,
                "changed_fact": field,
            }
            for field, value in edits
        ]
    candidates = [positive] + counterfactuals[:3]
    rng.shuffle(candidates)
    answer_index = next(idx for idx, item in enumerate(candidates) if item["supported"])
    return {
        "window_id": base_record["window_id"],
        "dataset_id": base_record.get("dataset_id", "unknown"),
        "label": base_record.get("label"),
        "subject_id": base_record.get("subject_id"),
        "evidence": card.to_json_dict(),
        "evidence_text": hard_evidence_text(card, variant=variant),
        "positive": positive,
        "counterfactuals": counterfactuals,
        "caption_selection": {
            "window_id": base_record["window_id"],
            "candidates": candidates,
            "answer_index": answer_index,
        },
        "difficulty": {
            "v1": "natural_language_near_miss",
            "v2": "natural_language_near_miss_v2",
            "v3": "numeric_partial_contradiction_v3",
        }.get(variant, "natural_language_near_miss"),
    }


def build_hard_benchmark(
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    max_records: int | None = None,
    variant: str = "v1",
) -> int:
    records = list(read_jsonl(input_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    hard_records = [build_hard_record(record, seed=seed, variant=variant) for record in records]
    write_jsonl(output_path, hard_records)
    return len(hard_records)
