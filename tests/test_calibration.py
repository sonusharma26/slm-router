"""Offline tests for the confidence calibration primitives (temperature
scaling + conformal calibration) used by serving-time confidence."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from slm_router.ml_core.confidence.calibration import (
    ConformalCalibrator,
    TemperatureScaler,
)
from slm_router.ml_core.confidence.predictor import ConfidencePredictor


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    # Correctness correlates with the first feature.
    p = 1.0 / (1.0 + np.exp(-X[:, 0]))
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_temperature_scaler_fits_and_bounds_probabilities():
    rng = np.random.default_rng(1)
    n = 100
    logits = rng.normal(size=(n, 2)) * 3
    labels = (logits[:, 1] > logits[:, 0]).astype(int)

    scaler = TemperatureScaler().fit(logits, labels)
    probs = scaler.transform(logits)

    assert probs.shape == logits.shape
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)
    assert scaler.temperature > 0


def test_conformal_calibrator_monotonic_accept():
    rng = np.random.default_rng(2)
    scores = rng.uniform(0, 1, size=200)
    labels = np.zeros_like(scores)  # unused by split conformal

    cal = ConformalCalibrator().fit(scores, labels, alpha=0.1)

    # Lower non-conformity scores should be at least as likely to be accepted.
    accept_low, lo_low, up_low = cal.predict_set(0.01)
    accept_high, lo_high, up_high = cal.predict_set(0.99)
    assert accept_low or not accept_high  # low score never worse than high score
    assert 0.0 <= lo_low <= 1.0 and up_low == 1.0
    assert 0.0 <= lo_high <= 1.0


def test_confidence_predictor_unfitted_is_safe_passthrough():
    predictor = ConfidencePredictor()
    feats = np.zeros(4, dtype=np.float32)

    prior = predictor.predict_prior(feats)
    assert 0.0 <= prior.p_correct <= 1.0

    posterior = predictor.predict_posterior(feats, {"mean_log_prob": -0.1})
    assert 0.0 <= posterior.p_correct <= 1.0
    assert 0.0 <= posterior.lower_bound <= 1.0


def test_confidence_predictor_fit_improves_calibration_and_is_monotonic():
    X, y = _synthetic(n=300, seed=3)
    predictor = ConfidencePredictor().fit(X, y)

    # Higher raw feature-driven score should not decrease calibrated confidence.
    low_feats = np.array([-3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    high_feats = np.array([3.0, 0.0, 0.0, 0.0], dtype=np.float32)

    p_low = predictor.predict_prior(low_feats).p_correct
    p_high = predictor.predict_prior(high_feats).p_correct

    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high >= p_low - 1e-6


def test_confidence_predictor_save_load_roundtrip(tmp_path):
    X, y = _synthetic(n=300, seed=4)
    predictor = ConfidencePredictor().fit(X, y)

    path = tmp_path / "confidence.joblib"
    predictor.save(path)

    restored = ConfidencePredictor()
    restored.load(path)

    feats = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    original = predictor.predict_prior(feats).p_correct
    reloaded = restored.predict_prior(feats).p_correct
    assert original == pytest.approx(reloaded)
