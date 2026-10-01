"""Calibration primitives: temperature scaling, conformal calibration, ensemble uncertainty."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive imports
# ---------------------------------------------------------------------------
try:
    from scipy.optimize import minimize as _sp_minimize  # type: ignore
    _SCIPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    _sp_minimize = None  # type: ignore
    _SCIPY_AVAILABLE = False

try:
    import torch  # type: ignore
    import torch.nn as _nn
    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    _nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    from sklearn.base import clone as _sk_clone  # type: ignore
    from sklearn.utils import resample as _sk_resample  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _sk_clone = None  # type: ignore
    _sk_resample = None  # type: ignore
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def _nll_scalar(log_temp: float | np.ndarray, logits: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood for a single temperature (numpy)."""
    # SciPy passes a one-element ndarray even for a scalar optimization
    # variable. Extract that scalar explicitly; direct ndarray-to-float
    # conversion is no longer supported by recent NumPy releases.
    log_temp_value = np.asarray(log_temp, dtype=np.float64).item()
    T = max(float(np.exp(log_temp_value)), 1e-6)
    probs = _softmax(logits / T)
    # Clip for numerical stability
    n = len(labels)
    idx = np.arange(n)
    nll = -np.mean(np.log(probs[idx, labels.astype(int)] + 1e-12))
    return float(nll)


# ---------------------------------------------------------------------------
# Temperature Scaler
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """1-parameter post-hoc calibration via temperature scaling.

    Fits the scalar T that minimises the NLL of held-out logits.
    Uses PyTorch L-BFGS when available, else scipy minimisation, else
    a simple grid search.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> "TemperatureScaler":
        """Fit temperature on validation logits/labels.

        Parameters
        ----------
        logits:
            Shape ``(N, C)`` raw (pre-softmax) scores.
        labels:
            Shape ``(N,)`` integer class indices.
        """
        logits = np.asarray(logits, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        if _TORCH_AVAILABLE and torch is not None:
            self._fit_torch(logits, labels)
        elif _SCIPY_AVAILABLE and _sp_minimize is not None:
            self._fit_scipy(logits, labels)
        else:
            self._fit_grid(logits, labels)

        logger.info("TemperatureScaler fitted: T=%.4f", self.temperature)
        return self

    def _fit_torch(self, logits: np.ndarray, labels: np.ndarray) -> None:
        t_logits = torch.tensor(logits, dtype=torch.float32)
        t_labels = torch.tensor(labels, dtype=torch.long)
        log_t = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.LBFGS([log_t], lr=0.01, max_iter=50)
        ce = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            T = torch.exp(log_t).clamp(min=1e-4)
            loss = ce(t_logits / T, t_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_t).item())

    def _fit_scipy(self, logits: np.ndarray, labels: np.ndarray) -> None:
        result = _sp_minimize(
            _nll_scalar,
            x0=[0.0],
            args=(logits, labels),
            method="L-BFGS-B",
            bounds=[(-4.0, 4.0)],
        )
        self.temperature = max(float(np.exp(result.x[0])), 1e-6)

    def _fit_grid(self, logits: np.ndarray, labels: np.ndarray) -> None:
        best_T, best_nll = 1.0, float("inf")
        for log_t in np.linspace(-3, 3, 61):
            nll = _nll_scalar(log_t, logits, labels)
            if nll < best_nll:
                best_nll = nll
                best_T = float(np.exp(log_t))
        self.temperature = best_T

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Parameters
        ----------
        logits:
            Shape ``(N, C)`` or ``(C,)``.

        Returns
        -------
        np.ndarray
            Probability array of same shape as input.
        """
        logits = np.asarray(logits, dtype=np.float64)
        return _softmax(logits / max(self.temperature, 1e-6))


# ---------------------------------------------------------------------------
# Conformal Calibrator
# ---------------------------------------------------------------------------

class ConformalCalibrator:
    """Split conformal prediction for classification.

    Fit on a held-out calibration set; at inference time returns a
    (accept, p_lower, p_upper) triple.

    ``score`` is assumed to be a *non-conformity* measure, e.g.
    ``1 - softmax_prob_of_true_class``.  Lower scores → more confident.
    """

    def __init__(self) -> None:
        self._threshold: float = 1.0  # conservative default
        self._alpha: float = 0.1

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        alpha: float = 0.1,
    ) -> "ConformalCalibrator":
        """Compute the conformal threshold from calibration scores.

        Parameters
        ----------
        scores:
            Non-conformity scores for calibration examples.
        labels:
            Not used by split conformal (marginal coverage guarantee).
            Kept for API symmetry.
        alpha:
            Desired marginal error rate.  Coverage ≥ 1 - alpha.
        """
        self._alpha = alpha
        scores = np.asarray(scores, dtype=np.float64)
        n = len(scores)
        # Finite-sample correction: ceil((n+1)*(1-alpha))/n quantile
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        self._threshold = float(np.quantile(scores, level))
        logger.info(
            "ConformalCalibrator fitted: alpha=%.2f, threshold=%.4f",
            alpha,
            self._threshold,
        )
        return self

    def predict_set(
        self,
        score: float,
    ) -> tuple[bool, float, float]:
        """Predict whether a new example is in the prediction set.

        Parameters
        ----------
        score:
            Non-conformity score for the new example.

        Returns
        -------
        accept:
            True if the example is accepted (score ≤ threshold).
        p_lower:
            Conservative lower bound on true probability (1 - threshold).
        p_upper:
            Upper bound (1.0).
        """
        accept = bool(score <= self._threshold)
        p_lower = float(np.clip(1.0 - self._threshold, 0.0, 1.0))
        p_upper = 1.0
        return accept, p_lower, p_upper


# ---------------------------------------------------------------------------
# Ensemble Uncertainty
# ---------------------------------------------------------------------------

class EnsembleUncertainty:
    """Epistemic uncertainty via bagged sklearn estimators.

    Parameters
    ----------
    base_estimator:
        Any sklearn estimator with ``predict_proba``.
    M:
        Number of bootstrap members.
    """

    def __init__(self, base_estimator: object, M: int = 10) -> None:
        self.base_estimator = base_estimator
        self.M = M
        self._members: list[object] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleUncertainty":
        """Fit M bootstrap members."""
        self._members = []
        if not _SKLEARN_AVAILABLE:
            logger.warning("sklearn unavailable; EnsembleUncertainty will return (0.5, 0.25).")
            return self
        rng = np.random.default_rng(42)
        X = np.asarray(X)
        y = np.asarray(y)
        for i in range(self.M):
            idx = rng.integers(0, len(X), size=len(X))
            m = _sk_clone(self.base_estimator)
            m.fit(X[idx], y[idx])
            self._members.append(m)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_probability, epistemic_variance) per sample.

        Parameters
        ----------
        X:
            Shape ``(N, d)``.

        Returns
        -------
        mean: np.ndarray shape (N,)
        var:  np.ndarray shape (N,)
        """
        X = np.asarray(X, dtype=np.float32)
        if not self._members:
            n = len(X)
            return np.full(n, 0.5, dtype=np.float32), np.full(n, 0.25, dtype=np.float32)

        preds = np.stack(
            [
                m.predict_proba(X)[:, 1]  # type: ignore[union-attr]
                for m in self._members
            ],
            axis=0,
        )  # (M, N)
        mean = preds.mean(axis=0).astype(np.float32)
        var = preds.var(axis=0).astype(np.float32)
        return mean, var
