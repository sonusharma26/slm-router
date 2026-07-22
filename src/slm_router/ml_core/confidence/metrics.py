"""Calibration metrics: ECE, MCE, Brier score, reliability curve (numpy only)."""
from __future__ import annotations

import numpy as np


def reliability_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability (calibration) curve.

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class, shape ``(N,)``.
    labels:
        Binary ground-truth labels, shape ``(N,)``.
    n_bins:
        Number of equal-width confidence bins.

    Returns
    -------
    bin_confidences:
        Mean predicted probability per bin.
    bin_accuracies:
        Observed accuracy per bin.
    bin_counts:
        Number of samples per bin.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(probs, bins[1:-1])  # 0 … n_bins-1

    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            bin_confidences[b] = probs[mask].mean()
            bin_accuracies[b] = labels[mask].mean()
            bin_counts[b] = int(mask.sum())

    return bin_confidences, bin_accuracies, bin_counts


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class.
    labels:
        Binary ground-truth labels.
    n_bins:
        Number of confidence bins.

    Returns
    -------
    float
        Weighted mean absolute gap between confidence and accuracy.
    """
    conf, acc, counts = reliability_curve(probs, labels, n_bins)
    n = max(counts.sum(), 1)
    ece = float(np.sum(counts * np.abs(conf - acc)) / n)
    return ece


def max_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE).

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class.
    labels:
        Binary ground-truth labels.
    n_bins:
        Number of confidence bins.

    Returns
    -------
    float
        Maximum absolute gap between confidence and accuracy across bins.
    """
    conf, acc, counts = reliability_curve(probs, labels, n_bins)
    populated = counts > 0
    if not populated.any():
        return 0.0
    return float(np.max(np.abs(conf[populated] - acc[populated])))


def brier_score(
    probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Mean squared error between predicted probabilities and binary labels.

    Parameters
    ----------
    probs:
        Predicted probabilities for the positive class, shape ``(N,)``.
    labels:
        Binary ground-truth labels, shape ``(N,)``.

    Returns
    -------
    float
        Brier score in ``[0, 1]``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((probs - labels) ** 2))
