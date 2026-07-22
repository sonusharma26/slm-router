"""Difficulty estimator: predicts query difficulty in [0, 1].

Silver-label strategy
---------------------
Because human difficulty annotations are expensive, we derive silver labels
as ``difficulty ≈ 1 - SLM_quality``, where ``SLM_quality`` is the correctness
score logged in :class:`~slm_router.trace.RunTrace` for a small SLM (e.g.,
Phi-2 or TinyLlama).  Queries where the SLM answers correctly (quality ≈ 1)
are treated as easy (difficulty ≈ 0); queries the SLM consistently fails are
treated as hard (difficulty ≈ 1).

In practice:
1. Run the SLM on a labelled evaluation corpus.
2. Compute ``label_i = 1 - mean_quality_i`` per query.
3. Call ``DifficultyEstimator.fit(queries, labels)``.
4. Save with ``estimator.save(path)`` and load at serving time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from slm_router.ml_core.features.difficulty_features import DifficultyFeaturizer
from slm_router.types import Query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive sklearn import
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import HistGradientBoostingRegressor as _HGBR  # type: ignore
    from sklearn.linear_model import LogisticRegression as _LR  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _HGBR = None  # type: ignore
    _LR = None  # type: ignore
    _SKLEARN_AVAILABLE = False

try:
    import joblib as _joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _joblib = None  # type: ignore
    _JOBLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_predict(feats: np.ndarray) -> float:
    """Mean of all features (all in [0,1]) as a naive difficulty proxy."""
    return float(np.clip(feats.mean(), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class DifficultyEstimator:
    """Predicts query difficulty in ``[0, 1]``.

    Wraps a sklearn regressor (``HistGradientBoostingRegressor`` by default)
    or falls back to a simple heuristic when sklearn is not available.

    Parameters
    ----------
    model:
        Any sklearn-compatible regressor exposing ``fit`` / ``predict``.
        Pass ``None`` to use the default model.
    featurizer:
        Pre-instantiated :class:`DifficultyFeaturizer`.
    """

    def __init__(
        self,
        model: object | None = None,
        featurizer: DifficultyFeaturizer | None = None,
    ) -> None:
        self.featurizer = featurizer
        self._fitted = False

        if model is not None:
            self._model = model
        elif _SKLEARN_AVAILABLE and _HGBR is not None:
            self._model = _HGBR(
                max_iter=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=0,
            )
        else:
            self._model = None
            logger.warning(
                "sklearn not available; DifficultyEstimator will use heuristic fallback."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _featurize(self, queries: Sequence[Query]) -> np.ndarray:
        if self.featurizer is None:
            raise RuntimeError("featurizer must be set before calling predict.")
        return np.stack([self.featurizer.vectorize(q) for q in queries])

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or not self._fitted:
            return np.array([_heuristic_predict(row) for row in X], dtype=np.float32)
        preds = np.asarray(self._model.predict(X), dtype=np.float32)
        return np.clip(preds, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, query: Query) -> float:
        """Predict difficulty for a single :class:`~slm_router.types.Query`.

        Returns
        -------
        float
            Difficulty score in ``[0, 1]``.
        """
        feats = self._featurize([query])
        return float(self._raw_predict(feats)[0])

    def predict_batch(self, queries: Sequence[Query]) -> np.ndarray:
        """Predict difficulty for a batch of queries.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` float32 array of difficulty scores in ``[0, 1]``.
        """
        if not queries:
            return np.array([], dtype=np.float32)
        X = self._featurize(queries)
        return self._raw_predict(X)

    def fit(
        self,
        queries: Sequence[Query],
        labels: Sequence[float],
    ) -> "DifficultyEstimator":
        """Train the estimator on silver-labelled data.

        Parameters
        ----------
        queries:
            Training queries.
        labels:
            Silver difficulty labels in ``[0, 1]``; typically
            ``1 - SLM_quality`` derived from logged traces.

        Returns
        -------
        self
        """
        X = self._featurize(queries)
        y = np.clip(np.asarray(labels, dtype=np.float32), 0.0, 1.0)

        if self._model is None:
            logger.warning("No model available; fit() is a no-op (heuristic mode).")
            self._fitted = False
            return self

        self._model.fit(X, y)
        self._fitted = True
        logger.info("DifficultyEstimator fitted on %d samples.", len(y))
        return self

    def save(self, path: str | Path) -> None:
        """Persist the estimator to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required for save/load.")
        _joblib.dump({"model": self._model, "fitted": self._fitted}, path)
        logger.info("DifficultyEstimator saved to %s.", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        featurizer: DifficultyFeaturizer | None = None,
    ) -> "DifficultyEstimator":
        """Load a previously saved estimator from disk.

        Parameters
        ----------
        path:
            Path written by :meth:`save`.
        featurizer:
            :class:`DifficultyFeaturizer` to attach (not serialised).
        """
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required for save/load.")
        data = _joblib.load(path)
        inst = cls.__new__(cls)
        inst._model = data["model"]
        inst._fitted = data["fitted"]
        inst.featurizer = featurizer
        logger.info("DifficultyEstimator loaded from %s.", path)
        return inst
