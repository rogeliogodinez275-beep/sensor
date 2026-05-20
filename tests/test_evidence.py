import numpy as np

from sensorfact.evidence import EvidenceCalibrator, EvidenceExtractor


def test_evidence_card_detects_periodic_dominant_axis_and_intensity():
    sample_rate = 20.0
    t = np.arange(80, dtype=np.float32) / sample_rate
    quiet = np.stack(
        [
            0.08 * np.sin(2 * np.pi * 1.5 * t),
            0.03 * np.sin(2 * np.pi * 1.5 * t),
            np.zeros_like(t),
        ],
        axis=1,
    )
    active = np.stack(
        [
            1.4 * np.sin(2 * np.pi * 1.5 * t),
            0.15 * np.sin(2 * np.pi * 1.5 * t),
            0.05 * np.cos(2 * np.pi * 1.5 * t),
        ],
        axis=1,
    )

    calibrator = EvidenceCalibrator.fit([quiet, active], sample_rate=sample_rate)
    extractor = EvidenceExtractor(
        calibrator=calibrator,
        channel_names=["acc_x", "acc_y", "acc_z"],
        sample_rate=sample_rate,
    )

    card = extractor.extract(active, window_id="w1", dataset_id="toy")

    assert card.window_id == "w1"
    assert card.intensity == "high"
    assert card.periodicity == "strong"
    assert card.dominant_axis == "acc_x"
    assert card.dominant_frequency in {"low", "mid"}
    assert card.numeric["rms_energy"] > calibrator.thresholds["rms_energy"][1]


def test_dominant_axis_is_uncertain_when_top_axes_are_too_close():
    sample_rate = 20.0
    t = np.arange(80, dtype=np.float32) / sample_rate
    window = np.stack(
        [
            np.sin(2 * np.pi * 1.0 * t),
            0.98 * np.sin(2 * np.pi * 1.0 * t),
            np.zeros_like(t),
        ],
        axis=1,
    )
    calibrator = EvidenceCalibrator.fit([window], sample_rate=sample_rate)
    extractor = EvidenceExtractor(
        calibrator=calibrator,
        channel_names=["acc_x", "acc_y", "acc_z"],
        sample_rate=sample_rate,
    )

    card = extractor.extract(window, window_id="w2", dataset_id="toy")

    assert card.dominant_axis == "uncertain"
