"""ConfidencePredictor: per-model-id confidence estimation with calibration."""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from slm_router.types import ConfidenceOut
from slm_router.ml_core.confidence.calibration import (
    ConformalCalibrator,
    EnsembleUncertainty,
    TemperatureScaler,
)
from slm_router.ml_core.confidence.metrics import expected_calibration_error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive sklearn imports
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import HistGradientBoostingClassifier as _HGBC  # type: ignore
    from sklearn.calibration import CalibratedClassifierCV as _CalCV  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _HGBC = None  # type: ignore
    _CalCV = None  # type: ignore
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# ConfidencePredictor
# ---------------------------------------------------------------------------

class ConfidencePredictor:
    """Predicts :class:`~slm_router.types.ConfidenceOut` for one model.

    Architecture
    ------------
    - **Prior**: uses ``base_ensemble`` (or a default
      ``HistGradientBoostingClassifier`` wrapped in
      ``CalibratedClassifierCV``) to estimate p_correct from difficulty
      features before the SLM runs.
    - **Posterior**: refines the estimate with raw signals (e.g. token-level
      log-probs, verifier score) after the SLM has produced a response.
    - **Calibration**: :class:`TemperatureScaler` + :class:`ConformalCalibrator`
      together give conformal_accept and lower_bound.

    One instance is created **per model_id** so that calibration is
    model-specific.

    Parameters
    ----------
    base_ensemble:
        Sklearn estimator with ``predict_proba``.  ``None`` → default.
    temp:
        :class:`TemperatureScaler` (may be unfitted).
    conformal:
        :class:`ConformalCalibrator` (may be unfitted).
    """

    def __init__(
        self,
        base_ensemble: object | None = None,
        temp: TemperatureScaler | None = None,
        conformal: ConformalCalibrator | None = None,
    ) -> None:
        self.temp = temp or TemperatureScaler()
        self.conformal = conformal or ConformalCalibrator()
        self._fitted = False

        if base_ensemble is not None:
            self._clf = base_ensemble
        elif _SKLEARN_AVAILABLE and _HGBC is not None and _CalCV is not None:
            base = _HGBC(max_iter=100, learning_rate=0.05, random_state=0)
            self._clf = _CalCV(base, cv=3, method="isotonic")
        else:
            self._clf = None
            logger.warning(
                "sklearn not available; ConfidencePredictor uses constant fallback."
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _raw_proba(self, feats: np.ndarray) -> np.ndarray:
        """Return p_correct per row."""
        if self._clf is None or not self._fitted:
            # Uninformative prior
            return np.full(len(feats), 0.5, dtype=np.float32)
        proba = self._clf.predict_proba(feats)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba.ravel().astype(np.float32)

    def _to_confidence_out(
        self,
        p_correct: float,
        epistemic_var: float,
        non_conformity: float,
    ) -> ConfidenceOut:
        conformal_accept, lower_bound, _ = self.conformal.predict_set(non_conformity)
        return ConfidenceOut(
            p_correct=float(np.clip(p_correct, 0.0, 1.0)),
            epistemic_var=float(np.clip(epistemic_var, 0.0, 1.0)),
            conformal_accept=conformal_accept,
            lower_bound=float(lower_bound),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_prior(self, feats: np.ndarray) -> ConfidenceOut:
        """Predict confidence before the SLM runs (prior).

        Parameters
        ----------
        feats:
            1-D feature vector from :class:`~slm_router.ml_core.features.DifficultyFeaturizer`.

        Returns
        -------
        ConfidenceOut
        """
        feats_2d = feats.reshape(1, -1)
        p = float(self._raw_proba(feats_2d)[0])
        # Rough epistemic: higher difficulty → higher uncertainty
        epistemic_var = float(np.clip(p * (1 - p) * 2, 0.0, 1.0))
        non_conformity = 1.0 - p
        return self._to_confidence_out(p, epistemic_var, non_conformity)

    def predict_posterior(
        self,
        feats: np.ndarray,
        raw_signals: dict[str, float],
    ) -> ConfidenceOut:
        """Update confidence after the SLM has answered.

        ``raw_signals`` may contain any subset of:
        - ``"mean_log_prob"``: mean token log-probability from the SLM.
        - ``"verifier_score"``: output of a lightweight verifier model [0,1].
        - ``"length_penalty"``: optional length-based penalty [0,1].

        Parameters
        ----------
        feats:
            Same feature vector as passed to :meth:`predict_prior`.
        raw_signals:
            Dict of optional posterior signals.

        Returns
        -------
        ConfidenceOut
        """
        prior = self.predict_prior(feats)
        p = prior.p_correct

        # Integrate available signals with simple weighted average
        signal_vals: list[float] = [p]
        if "mean_log_prob" in raw_signals:
            # Convert log-prob (typically −∞…0) to [0,1]
            lp = float(raw_signals["mean_log_prob"])
            signal_vals.append(float(np.clip(np.exp(lp), 0.0, 1.0)))
        if "verifier_score" in raw_signals:
            signal_vals.append(float(np.clip(raw_signals["verifier_score"], 0.0, 1.0)))
        if "length_penalty" in raw_signals:
            signal_vals.append(float(np.clip(1.0 - raw_signals["length_penalty"], 0.0, 1.0)))

        p_post = float(np.mean(signal_vals))
        epistemic_var = float(np.var(signal_vals)) if len(signal_vals) > 1 else prior.epistemic_var
        non_conformity = 1.0 - p_post
        return self._to_confidence_out(p_post, epistemic_var, non_conformity)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConfidencePredictor":
        """Train the classifier and calibrate.

        Also fits the conformal calibrator using a 20 % holdout and
        reports Expected Calibration Error.

        Parameters
        ----------
        X:
            Feature matrix shape ``(N, d)``.
        y:
            Binary correctness labels shape ``(N,)``.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(y)

        if n < 10:
            logger.warning("Too few samples (%d) to fit ConfidencePredictor.", n)
            return self

        # 80/20 split for conformal calibration
        split = max(int(0.8 * n), 1)
        X_train, X_cal = X[:split], X[split:]
        y_train, y_cal = y[:split], y[split:]

        if self._clf is not None:
            self._clf.fit(X_train, y_train.astype(int))
            self._fitted = True

            p_cal = self._raw_proba(X_cal)
            # Non-conformity score = 1 - p_correct
            nc_scores = 1.0 - p_cal
            self.conformal.fit(nc_scores, y_cal)

            # ECE report
            ece = expected_calibration_error(p_cal, y_cal)
            logger.info("ConfidencePredictor fitted: ECE=%.4f on %d cal samples.", ece, len(y_cal))
        else:
            logger.warning("No classifier; ConfidencePredictor in heuristic mode.")

        return self
