import pytest

from sensorfact.metrics import brier_score, expected_calibration_error, risk_coverage_auc


def test_brier_score_for_binary_probabilities():
    assert brier_score([1, 0], [0.9, 0.2]) == pytest.approx(((0.1**2) + (0.2**2)) / 2)


def test_expected_calibration_error_groups_confidence_bins():
    # Bin 0.5-1.0 has confidence mean 0.75 and accuracy 0.5, so contribution is 0.25.
    assert expected_calibration_error([1, 0], [0.9, 0.6], n_bins=2) == pytest.approx(0.25)


def test_risk_coverage_auc_rewards_high_confidence_correct_predictions():
    good = risk_coverage_auc([1, 0, 1], [0.95, 0.05, 0.9])
    bad = risk_coverage_auc([0, 1, 0], [0.95, 0.05, 0.9])

    assert good > bad
    assert 0.0 <= good <= 1.0
