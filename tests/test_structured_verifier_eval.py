from sensorfact.structured_verifier_eval import (
    NumericEvidenceCalibrator,
    build_structured_evidence_prompt,
    evaluate_structured_grounding,
    extract_claim_fields,
    extract_evidence_numeric_values,
    parse_evidence_fields_from_text,
)


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompts):
        self.prompts.extend(prompts)
        return self.responses[: len(prompts)]


def make_record(
    window_id: str,
    *,
    intensity: str,
    periodicity: str,
    dominant_axis: str,
    dominant_frequency: str,
    burstiness: str,
    rms_energy: float,
    autocorr_peak: float,
    fft_dominant_ratio: float,
    dominant_frequency_hz: float,
    peak_count: float,
):
    evidence = {
        "window_id": window_id,
        "dataset_id": "toy",
        "intensity": intensity,
        "periodicity": periodicity,
        "dominant_axis": dominant_axis,
        "dominant_frequency": dominant_frequency,
        "cross_channel_relation": "acc_y and gyro_x move oppositely",
        "lead_lag": "no clear lag",
        "burstiness": burstiness,
        "trend_segments": ["fall", "fall", "fall"],
        "numeric": {
            "rms_energy": rms_energy,
            "autocorr_peak": autocorr_peak,
            "fft_dominant_ratio": fft_dominant_ratio,
            "dominant_frequency_hz": dominant_frequency_hz,
            "peak_count": peak_count,
        },
    }
    evidence_text = (
        f"RMS energy is {rms_energy:.3f}, inside the middle reference band. "
        f"The repeatability scores are autocorrelation {autocorr_peak:.3f} and FFT concentration "
        f"{fft_dominant_ratio:.3f}, near the repeatability reference. "
        f"Axis comparison: the largest per-axis energy is on the roll-rate gyroscope trace. "
        f"The dominant spectral peak is {dominant_frequency_hz:.3f} Hz, above the upper reference band. "
        f"Peak count is {peak_count:.0f}, with the profile outside the expected reference band for burstiness. "
        "Segment slopes move falling, then falling, then falling. "
        "Pairwise channel check: acc_y and gyro_x move oppositely."
    )
    positive = {
        "text": (
            f"Overall, the motion shows moderate energy, a loose rhythm, "
            f"the largest per-axis energy is on the roll-rate gyroscope trace, "
            f"fast cadence, and {'short spikes' if burstiness == 'bursty' else 'an even profile'}."
        )
    }
    wrong_axis = {
        "text": (
            "Overall, the motion shows moderate energy, a loose rhythm, "
            "the largest per-axis energy is on the forward-back acceleration trace, "
            f"fast cadence, and {'short spikes' if burstiness == 'bursty' else 'an even profile'}."
        ),
        "changed_fact": "dominant_axis",
    }
    wrong_burst = {
        "text": (
            "Overall, the motion shows moderate energy, a loose rhythm, "
            "the largest per-axis energy is on the roll-rate gyroscope trace, "
            f"fast cadence, and {'an even profile' if burstiness == 'bursty' else 'short spikes'}."
        ),
        "changed_fact": "burstiness",
    }
    return {
        "window_id": window_id,
        "evidence": evidence,
        "evidence_text": evidence_text,
        "positive": positive,
        "counterfactuals": [wrong_axis, wrong_burst],
        "caption_selection": {
            "answer_index": 0,
            "candidates": [positive, wrong_axis, wrong_burst],
        },
    }


def test_extract_evidence_numeric_values_reads_hard_v3_surface():
    values = extract_evidence_numeric_values(
        "RMS energy is 0.102, inside the middle reference band. "
        "The repeatability scores are autocorrelation 0.297 and FFT concentration 0.084, near the repeatability reference. "
        "Axis comparison: the largest per-axis energy is on the roll-rate gyroscope trace. "
        "The dominant spectral peak is 0.781 Hz, above the upper reference band. Peak count is 7."
    )

    assert values["rms_energy"] == 0.102
    assert values["autocorr_peak"] == 0.297
    assert values["fft_dominant_ratio"] == 0.084
    assert values["dominant_frequency_hz"] == 0.781
    assert values["peak_count"] == 7.0


def test_regex_parser_can_fill_missing_burstiness_from_numeric_calibrator():
    train_records = [
        make_record(
            "smooth_a",
            intensity="medium",
            periodicity="weak",
            dominant_axis="gyro_x",
            dominant_frequency="high",
            burstiness="smooth",
            rms_energy=0.10,
            autocorr_peak=0.29,
            fft_dominant_ratio=0.08,
            dominant_frequency_hz=0.78,
            peak_count=3,
        ),
        make_record(
            "bursty_b",
            intensity="medium",
            periodicity="weak",
            dominant_axis="gyro_x",
            dominant_frequency="high",
            burstiness="bursty",
            rms_energy=0.11,
            autocorr_peak=0.31,
            fft_dominant_ratio=0.09,
            dominant_frequency_hz=0.80,
            peak_count=11,
        ),
    ]
    calibrator = NumericEvidenceCalibrator().fit(train_records)

    fields = parse_evidence_fields_from_text(
        train_records[1]["evidence_text"],
        calibrator=calibrator,
    )

    assert fields["intensity"] == "medium"
    assert fields["periodicity"] == "weak"
    assert fields["dominant_axis"] == "gyro_x"
    assert fields["dominant_frequency"] == "high"
    assert fields["burstiness"] == "bursty"


def test_evaluate_structured_grounding_hits_perfect_score_with_model_json():
    record = make_record(
        "w1",
        intensity="medium",
        periodicity="weak",
        dominant_axis="gyro_x",
        dominant_frequency="high",
        burstiness="smooth",
        rms_energy=0.10,
        autocorr_peak=0.29,
        fft_dominant_ratio=0.08,
        dominant_frequency_hz=0.78,
        peak_count=3,
    )
    generator = FakeGenerator(
        [
            (
                '{"intensity":"medium","periodicity":"weak","dominant_axis":"gyro_x",'
                '"dominant_frequency":"high","burstiness":"smooth"}'
            )
        ]
    )

    metrics, rows = evaluate_structured_grounding(
        [record],
        parser_mode="model_evidence",
        generator=generator,
        system_name="toy_structured",
    )

    assert len(generator.prompts) == 1
    assert metrics["caption_selection_accuracy"] == 1.0
    assert metrics["cf_reject_accuracy"] == 1.0
    assert metrics["cf_reject_f1"] == 1.0
    assert metrics["evidence_parse_complete_rate"] == 1.0
    assert rows[0]["caption_prediction"] == 0
    assert rows[0]["support_predictions"] == [True, False, False]
    assert rows[0]["model_response"] == generator.responses[0]


def test_model_parser_rejects_axis_outside_dataset_schema():
    record = make_record(
        "wisdm_invalid_axis",
        intensity="medium",
        periodicity="weak",
        dominant_axis="acc_x",
        dominant_frequency="high",
        burstiness="smooth",
        rms_energy=0.12,
        autocorr_peak=0.30,
        fft_dominant_ratio=0.11,
        dominant_frequency_hz=0.90,
        peak_count=3,
    )
    record["dataset_id"] = "wisdm"
    record["evidence"]["dataset_id"] = "wisdm"
    generator = FakeGenerator(
        [
            (
                '{"intensity":"medium","periodicity":"weak","dominant_axis":"gyro_x",'
                '"dominant_frequency":"high","burstiness":"smooth"}'
            )
        ]
    )

    metrics, rows = evaluate_structured_grounding(
        [record],
        parser_mode="model_evidence",
        generator=generator,
    )

    assert "dominant_axis" not in rows[0]["evidence_fields"]
    assert metrics["evidence_parse_complete_rate"] == 0.0


def test_structured_verifier_supports_dataset_specific_mhealth_axes():
    record = make_record(
        "mhealth_axis",
        intensity="medium",
        periodicity="weak",
        dominant_axis="ankle_mag_x",
        dominant_frequency="high",
        burstiness="smooth",
        rms_energy=0.10,
        autocorr_peak=0.29,
        fft_dominant_ratio=0.08,
        dominant_frequency_hz=0.78,
        peak_count=3,
    )
    record["evidence"]["dataset_id"] = "mhealth"
    record["evidence"]["dominant_axis"] = "ankle_mag_x"
    record["evidence_text"] = record["evidence_text"].replace(
        "roll-rate gyroscope trace",
        "ankle magnetometer x trace",
    )
    for item in [record["positive"], *record["caption_selection"]["candidates"]]:
        item["text"] = item["text"].replace("roll-rate gyroscope trace", "ankle magnetometer x trace")

    bursty_record = make_record(
        "mhealth_axis_bursty",
        intensity="medium",
        periodicity="weak",
        dominant_axis="ankle_mag_x",
        dominant_frequency="high",
        burstiness="bursty",
        rms_energy=0.12,
        autocorr_peak=0.31,
        fft_dominant_ratio=0.09,
        dominant_frequency_hz=0.80,
        peak_count=11,
    )
    calibrator = NumericEvidenceCalibrator().fit([record, bursty_record])

    metrics, rows = evaluate_structured_grounding(
        [record],
        parser_mode="regex_evidence",
        calibrator=calibrator,
    )

    assert rows[0]["evidence_fields"]["dominant_axis"] == "ankle_mag_x"
    assert metrics["evidence_parse_complete_rate"] == 1.0
    assert metrics["caption_selection_accuracy"] == 1.0


def test_axis_phrase_matching_prefers_dataset_specific_long_phrases():
    fields = extract_claim_fields(
        "Overall, the motion shows moderate energy, a loose rhythm, "
        "the largest per-axis energy is on the chest forward-back acceleration trace, "
        "fast cadence, and an even profile."
    )

    assert fields["dominant_axis"] == "chest_acc_x"


def test_evidence_phrase_matching_prefers_dataset_specific_long_phrases():
    fields = parse_evidence_fields_from_text(
        "RMS energy is 0.120, inside the middle reference band. "
        "The repeatability scores are autocorrelation 0.300 and FFT concentration 0.100, "
        "near the repeatability reference. "
        "Axis comparison: the largest per-axis energy is on the ankle roll-rate gyroscope trace. "
        "The dominant spectral peak is 0.900 Hz, above the upper reference band. "
        "Peak count is 4, with an even profile."
    )

    assert fields["dominant_axis"] == "ankle_gyro_x"


def test_axis_phrase_matching_keeps_legacy_mhealth_mag_surface():
    fields = extract_claim_fields(
        "Overall, the motion shows moderate energy, a loose rhythm, "
        "the largest per-axis energy is on the ankle mag x, "
        "fast cadence, and an even profile."
    )

    assert fields["dominant_axis"] == "ankle_mag_x"


def test_structured_prompt_styles_change_instruction_surface():
    record = make_record(
        "w_prompt",
        intensity="medium",
        periodicity="weak",
        dominant_axis="gyro_x",
        dominant_frequency="high",
        burstiness="smooth",
        rms_energy=0.10,
        autocorr_peak=0.29,
        fft_dominant_ratio=0.08,
        dominant_frequency_hz=0.78,
        peak_count=3,
    )

    strict_prompt = build_structured_evidence_prompt(record, prompt_style="strict_json")
    terse_prompt = build_structured_evidence_prompt(record, prompt_style="terse")
    chain_prompt = build_structured_evidence_prompt(record, prompt_style="chain_then_json")
    fewshot_prompt = build_structured_evidence_prompt(record, prompt_style="fewshot_json")

    assert strict_prompt != terse_prompt
    assert "Return JSON only" in strict_prompt
    assert "Only output JSON" in terse_prompt
    assert "briefly reason" in chain_prompt
    assert '"intensity":"medium"' in fewshot_prompt
    assert "ankle_mag_x" in fewshot_prompt
    assert "Axis code hints" in strict_prompt
    assert "gyro_x: roll-rate gyroscope trace" in strict_prompt
