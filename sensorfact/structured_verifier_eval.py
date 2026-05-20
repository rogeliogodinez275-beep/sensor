from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sensorfact.benchmark import evidence_to_text
from sensorfact.axis_vocab import AXIS_TEXT_TO_CODE, AXIS_VALUES, axis_phrase, dataset_axis_options
from sensorfact.io import write_json, write_jsonl
from sensorfact.metrics import caption_selection_accuracy, counterfactual_rejection_metrics
from sensorfact.qwen_llm_eval import _extract_json, parse_support_labels, record_evidence_text
from sensorfact.schemas import EvidenceCard


class TextGenerator(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...


FIELDS = ["intensity", "periodicity", "dominant_axis", "dominant_frequency", "burstiness"]
FIELD_VALUES = {
    "intensity": ["low", "medium", "high"],
    "periodicity": ["none", "weak", "strong"],
    "dominant_axis": AXIS_VALUES,
    "dominant_frequency": ["low", "mid", "high", "uncertain"],
    "burstiness": ["smooth", "bursty"],
}


def extract_evidence_numeric_values(text: str) -> dict[str, float]:
    patterns = {
        "rms_energy": r"RMS energy is\s+([-+]?\d+(?:\.\d+)?)",
        "autocorr_peak": r"autocorrelation\s+([-+]?\d+(?:\.\d+)?)",
        "fft_dominant_ratio": r"FFT concentration\s+([-+]?\d+(?:\.\d+)?)",
        "dominant_frequency_hz": r"dominant spectral peak is\s+([-+]?\d+(?:\.\d+)?)\s*Hz",
        "peak_count": r"Peak count is\s+([-+]?\d+(?:\.\d+)?)",
    }
    out: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            out[key] = float(match.group(1))
    return out


@dataclass
class NumericEvidenceCalibrator:
    intensity_thresholds: tuple[float, float] | None = None
    periodicity_thresholds: tuple[float, float] | None = None
    dominant_frequency_thresholds: tuple[float, float] | None = None
    burstiness_threshold: float | None = None

    def fit(self, records: list[dict]) -> "NumericEvidenceCalibrator":
        self.intensity_thresholds = _fit_three_way_thresholds(records, "rms_energy", "intensity")
        self.periodicity_thresholds = _fit_three_way_thresholds(records, "autocorr_peak", "periodicity")
        self.dominant_frequency_thresholds = _fit_three_way_thresholds(
            records, "dominant_frequency_hz", "dominant_frequency"
        )
        smooth_values = [
            float(record["evidence"].get("numeric", {}).get("peak_count", 0.0))
            for record in records
            if record["evidence"].get("burstiness") == "smooth"
        ]
        bursty_values = [
            float(record["evidence"].get("numeric", {}).get("peak_count", 0.0))
            for record in records
            if record["evidence"].get("burstiness") == "bursty"
        ]
        if smooth_values and bursty_values:
            self.burstiness_threshold = float((np.mean(smooth_values) + np.mean(bursty_values)) / 2.0)
        return self

    def infer_field(self, field: str, numeric_values: dict[str, float]) -> str | None:
        if field == "intensity" and self.intensity_thresholds and "rms_energy" in numeric_values:
            low_mid, mid_high = self.intensity_thresholds
            value = numeric_values["rms_energy"]
            if value < low_mid:
                return "low"
            if value < mid_high:
                return "medium"
            return "high"
        if field == "periodicity" and self.periodicity_thresholds and "autocorr_peak" in numeric_values:
            none_weak, weak_strong = self.periodicity_thresholds
            value = numeric_values["autocorr_peak"]
            if value < none_weak:
                return "none"
            if value < weak_strong:
                return "weak"
            return "strong"
        if field == "dominant_frequency" and self.dominant_frequency_thresholds and "dominant_frequency_hz" in numeric_values:
            low_mid, mid_high = self.dominant_frequency_thresholds
            value = numeric_values["dominant_frequency_hz"]
            if value < low_mid:
                return "low"
            if value < mid_high:
                return "mid"
            return "high"
        if field == "burstiness" and self.burstiness_threshold is not None and "peak_count" in numeric_values:
            return "bursty" if numeric_values["peak_count"] >= self.burstiness_threshold else "smooth"
        return None


def _fit_three_way_thresholds(records: list[dict], numeric_key: str, field: str) -> tuple[float, float] | None:
    order = {
        "intensity": ["low", "medium", "high"],
        "periodicity": ["none", "weak", "strong"],
        "dominant_frequency": ["low", "mid", "high"],
    }.get(field)
    if not order:
        return None
    means = []
    for label in order:
        values = [
            float(record["evidence"].get("numeric", {}).get(numeric_key, 0.0))
            for record in records
            if record["evidence"].get(field) == label
        ]
        if not values:
            return None
        means.append(float(np.mean(values)))
    return ((means[0] + means[1]) / 2.0, (means[1] + means[2]) / 2.0)


def parse_evidence_fields_from_text(
    text: str,
    *,
    calibrator: NumericEvidenceCalibrator | None = None,
) -> dict[str, str]:
    lower = text.lower()
    fields: dict[str, str] = {}

    if "lower reference band" in lower:
        fields["intensity"] = "low"
    elif "middle reference band" in lower:
        fields["intensity"] = "medium"
    elif "upper reference band" in lower and "rms energy" in lower:
        fields["intensity"] = "high"

    if "below the repeatability reference" in lower:
        fields["periodicity"] = "none"
    elif "near the repeatability reference" in lower:
        fields["periodicity"] = "weak"
    elif "above the repeatability reference" in lower:
        fields["periodicity"] = "strong"

    axis_text_matches = sorted(
        AXIS_TEXT_TO_CODE.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for phrase, code in axis_text_matches:
        if phrase in lower:
            fields["dominant_axis"] = code
            break
    if "without a clear leader" in lower or "no single channel clearly dominates" in lower:
        fields["dominant_axis"] = "uncertain"

    if "dominant spectral peak" in lower:
        if "without a stable reference crossing" in lower:
            fields["dominant_frequency"] = "uncertain"
        elif "above the upper reference band" in lower:
            fields["dominant_frequency"] = "high"
        elif "middle reference band" in lower:
            fields["dominant_frequency"] = "mid"
        elif "lower reference band" in lower:
            fields["dominant_frequency"] = "low"

    if "spiky" in lower or "short spikes" in lower:
        fields["burstiness"] = "bursty"
    elif "even profile" in lower:
        fields["burstiness"] = "smooth"

    numeric_values = extract_evidence_numeric_values(text)
    if calibrator is not None:
        for field in FIELDS:
            if field not in fields:
                inferred = calibrator.infer_field(field, numeric_values)
                if inferred is not None:
                    fields[field] = inferred
    return fields


def build_structured_evidence_prompt(record: dict, prompt_style: str = "strict_json") -> str:
    evidence = record_evidence_text(record)
    dataset_id = str(record.get("dataset_id") or record.get("evidence", {}).get("dataset_id", ""))
    axis_options = dataset_axis_options(dataset_id)
    axis_legend = "\n".join(f"- {axis}: {axis_phrase(axis)}" for axis in axis_options)
    schema = {
        "intensity": "one of low/medium/high",
        "periodicity": "one of none/weak/strong",
        "dominant_axis": "one of " + "/".join(axis_options),
        "dominant_frequency": "one of low/mid/high/uncertain",
        "burstiness": "one of smooth/bursty",
    }
    if prompt_style == "terse":
        instruction = "Extract the five fields. Only output JSON with the schema keys."
    elif prompt_style == "chain_then_json":
        instruction = (
            "Extract the five fields, briefly reason about the numeric clues, "
            "then put the final JSON on the last line."
        )
    elif prompt_style == "fewshot_json":
        instruction = (
            "Extract the five fields. Follow this style example exactly:\n"
            '{"intensity":"medium","periodicity":"weak","dominant_axis":"ankle_mag_x","dominant_frequency":"high","burstiness":"smooth"}'
        )
    else:
        instruction = "Extract a normalized sensor evidence summary. Return JSON only with these keys and enum values."
    return (
        f"{instruction}\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"Axis code hints:\n{axis_legend}\n\n"
        f"Evidence:\n{evidence}\n"
    )


def _parse_model_fields(response: str, *, allowed_axes: list[str] | None = None) -> dict[str, str]:
    parsed = _extract_json(response)
    if not isinstance(parsed, dict):
        return {}
    fields: dict[str, str] = {}
    allowed_axis_values = set(allowed_axes or FIELD_VALUES["dominant_axis"])
    for field in FIELDS:
        value = parsed.get(field)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if field == "dominant_axis":
            if normalized in allowed_axis_values:
                fields[field] = normalized
                continue
            phrase_value = AXIS_TEXT_TO_CODE.get(value.strip().lower().replace("_", " "))
            if phrase_value in allowed_axis_values:
                fields[field] = phrase_value
            continue
        if normalized in FIELD_VALUES[field]:
            fields[field] = normalized
    return fields


def extract_claim_fields(text: str) -> dict[str, str]:
    lower = text.lower()
    fields: dict[str, str] = {}
    intensity_map = {
        "subtle energy": "low",
        "small-amplitude movement": "low",
        "moderate energy": "medium",
        "moderate-amplitude movement": "medium",
        "forceful energy": "high",
        "large-amplitude movement": "high",
    }
    periodicity_map = {
        "little repeatable rhythm": "none",
        "loose rhythm": "weak",
        "repeatable rhythm": "strong",
    }
    axis_map = sorted(
        ((phrase, code) for phrase, code in AXIS_TEXT_TO_CODE.items() if code != "uncertain"),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    frequency_map = {
        "slow cadence": "low",
        "middle-rate cadence": "mid",
        "mid-paced cadence": "mid",
        "fast cadence": "high",
        "quick cadence": "high",
        "no stable cadence": "uncertain",
    }
    burstiness_map = {
        "an even profile": "smooth",
        "smooth movement pattern": "smooth",
        "short spikes": "bursty",
        "spiky bursts": "bursty",
    }
    for phrase, value in intensity_map.items():
        if phrase in lower:
            fields["intensity"] = value
            break
    for phrase, value in periodicity_map.items():
        if phrase in lower:
            fields["periodicity"] = value
            break
    for phrase, value in axis_map:
        if phrase in lower:
            fields["dominant_axis"] = value
            break
    if "without a single clearly dominant channel" in lower:
        fields["dominant_axis"] = "uncertain"
    for phrase, value in frequency_map.items():
        if phrase in lower:
            fields["dominant_frequency"] = value
            break
    for phrase, value in burstiness_map.items():
        if phrase in lower:
            fields["burstiness"] = value
            break
    return fields


def verify_statement_against_fields(statement_text: str, evidence_fields: dict[str, str]) -> bool:
    claims = extract_claim_fields(statement_text)
    if not claims:
        return False
    for field, value in claims.items():
        if evidence_fields.get(field) != value:
            return False
    return True


def _caption_scores(record: dict, evidence_fields: dict[str, str]) -> list[float]:
    return [
        1.0 if verify_statement_against_fields(item["text"], evidence_fields) else 0.0
        for item in record["caption_selection"]["candidates"]
    ]


def _support_items(record: dict) -> list[dict]:
    return [
        {
            "kind": "positive",
            "text": record["positive"]["text"],
            "supported": True,
            "changed_fact": None,
            "changed_facts": [],
        }
    ] + [
        {
            "kind": "counterfactual",
            "text": item["text"],
            "supported": False,
            "changed_fact": item.get("changed_fact"),
            "changed_facts": item.get("changed_facts", []),
        }
        for item in record.get("counterfactuals", [])
    ]


def evaluate_structured_grounding(
    records: list[dict],
    *,
    parser_mode: str = "regex_evidence",
    generator: TextGenerator | None = None,
    calibrator: NumericEvidenceCalibrator | None = None,
    system_name: str = "structured_verifier",
    prompt_style: str = "strict_json",
) -> tuple[dict, list[dict]]:
    if parser_mode == "model_evidence" and generator is None:
        raise ValueError("generator is required when parser_mode=model_evidence")

    if calibrator is None:
        calibrator = NumericEvidenceCalibrator().fit(records)

    evidence_fields_list: list[dict[str, str]] = []
    model_responses: list[str | None] = [None for _ in records]
    if parser_mode == "model_evidence":
        prompts = [build_structured_evidence_prompt(record, prompt_style=prompt_style) for record in records]
        responses = generator.generate(prompts)
        model_responses = list(responses)
        evidence_fields_list = []
        for record, response in zip(records, responses):
            dataset_id = str(record.get("dataset_id") or record.get("evidence", {}).get("dataset_id", ""))
            evidence_fields_list.append(
                _parse_model_fields(response, allowed_axes=dataset_axis_options(dataset_id))
            )
    else:
        evidence_fields_list = [
            parse_evidence_fields_from_text(record_evidence_text(record), calibrator=calibrator) for record in records
        ]

    selection_examples = []
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    parse_successes = []

    for record, evidence_fields, model_response in zip(records, evidence_fields_list, model_responses):
        parse_successes.append(int(all(field in evidence_fields for field in FIELDS)))
        caption_scores = _caption_scores(record, evidence_fields)
        caption_prediction = int(np.argmax(caption_scores)) if caption_scores else None
        selection_examples.append(
            {
                "answer_index": record["caption_selection"]["answer_index"],
                "scores": caption_scores,
            }
        )
        support_items = _support_items(record)
        support_predictions = [
            verify_statement_against_fields(item["text"], evidence_fields) for item in support_items
        ]
        y_true.extend([1 if item["supported"] else 0 for item in support_items])
        y_score.extend([1.0 if pred else 0.0 for pred in support_predictions])
        rows.append(
            {
                "window_id": record["window_id"],
                "caption_prediction": caption_prediction,
                "caption_answer_index": record["caption_selection"]["answer_index"],
                "caption_scores": caption_scores,
                "support_predictions": support_predictions,
                "support_items": support_items,
                "evidence_fields": evidence_fields,
                "model_response": model_response,
            }
        )

    cf = counterfactual_rejection_metrics(y_true, y_score, threshold=0.5)
    metrics = {
        "system": system_name,
        "parser_mode": parser_mode,
        "caption_selection_accuracy": caption_selection_accuracy(selection_examples),
        "cf_reject_accuracy": cf["accuracy"],
        "cf_reject_precision": cf["precision"],
        "cf_reject_recall": cf["recall"],
        "cf_reject_f1": cf["f1"],
        "n_eval_records": len(records),
        "evidence_parse_complete_rate": float(np.mean(parse_successes)) if parse_successes else 0.0,
        "prompt_style": prompt_style,
    }
    return metrics, rows


def run_structured_verifier_eval(
    benchmark_path,
    output_metrics_path,
    output_rows_path,
    *,
    parser_mode: str = "regex_evidence",
    model_dir=None,
    max_records: int = -1,
    batch_size: int = 2,
    max_new_tokens: int = 96,
    device: str | None = None,
    prompt_style: str = "strict_json",
) -> dict:
    from sensorfact.io import read_jsonl
    from sensorfact.qwen_llm_eval import TransformersCausalLM

    records = list(read_jsonl(benchmark_path))
    if max_records is not None and max_records >= 0:
        records = records[:max_records]

    calibrator = NumericEvidenceCalibrator().fit(records)
    generator = None
    system_name = "regex_structured_verifier"
    if parser_mode == "model_evidence":
        if model_dir is None:
            raise ValueError("model_dir is required for parser_mode=model_evidence")
        generator = TransformersCausalLM(
            model_dir=model_dir,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        system_name = "model_structured_verifier"

    metrics, rows = evaluate_structured_grounding(
        records,
        parser_mode=parser_mode,
        generator=generator,
        calibrator=calibrator,
        system_name=system_name,
        prompt_style=prompt_style,
    )
    metrics["benchmark_path"] = str(benchmark_path)
    metrics["max_records"] = max_records
    metrics["model_dir"] = None if model_dir is None else str(model_dir)
    metrics["prompt_style"] = prompt_style
    write_json(output_metrics_path, metrics)
    write_jsonl(output_rows_path, rows)
    return metrics
