"""Calibration metrics for confidence-accuracy alignment."""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import numpy as np
    from numpy import ndarray
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    ndarray = None  # type: ignore[assignment,misc]
    _HAS_NUMPY = False


def _require_numpy() -> None:
    if not _HAS_NUMPY:
        raise ImportError(
            "numpy is required for calibration metrics. "
            "Install it with: pip install numpy"
        )


def _bin_data(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int,
) -> tuple["ndarray", "ndarray", "ndarray"]:
    """Bin confidence/correctness pairs into *n_bins* equal-width bins.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts) — arrays of length n_bins.
        Bins with no samples get confidence=midpoint, accuracy=0, count=0.
    """
    _require_numpy()
    confs = np.array(confidences, dtype=float)
    correct = np.array(correctness, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge only in last bin
        if i < n_bins - 1:
            mask = (confs >= lo) & (confs < hi)
        else:
            mask = (confs >= lo) & (confs <= hi)

        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            bin_confidences[i] = confs[mask].mean()
            bin_accuracies[i] = correct[mask].mean()
        else:
            bin_confidences[i] = (lo + hi) / 2.0
            bin_accuracies[i] = 0.0

    return bin_confidences, bin_accuracies, bin_counts


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |confidence_b - accuracy_b|

    Args:
        confidences: Predicted confidence values in [0, 1].
        correctness: Binary correctness indicators.
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]. Lower is better.
    """
    _require_numpy()
    n = len(confidences)
    if n == 0:
        return 0.0

    bin_confs, bin_accs, bin_counts = _bin_data(confidences, correctness, n_bins)
    weights = bin_counts / n
    ece = float(np.sum(weights * np.abs(bin_confs - bin_accs)))
    return ece


def max_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE).

    MCE = max_b |confidence_b - accuracy_b| over non-empty bins.

    Returns:
        MCE in [0, 1]. Lower is better.
    """
    _require_numpy()
    if not confidences:
        return 0.0

    bin_confs, bin_accs, bin_counts = _bin_data(confidences, correctness, n_bins)
    non_empty = bin_counts > 0
    if not non_empty.any():
        return 0.0

    mce = float(np.abs(bin_confs[non_empty] - bin_accs[non_empty]).max())
    return mce


def brier_score(
    confidences: list[float],
    correctness: list[bool],
) -> float:
    """Brier score: mean squared error between confidence and outcome.

    Brier = (1/N) * sum (p_i - y_i)^2

    Returns:
        Brier score in [0, 1]. Lower is better.
    """
    _require_numpy()
    if not confidences:
        return 0.0

    p = np.array(confidences, dtype=float)
    y = np.array(correctness, dtype=float)
    return float(np.mean((p - y) ** 2))


def reliability_curve(
    confidences: list[float],
    correctness: list[bool],
    n_bins: int = 10,
) -> "tuple[ndarray, ndarray, ndarray]":
    """Compute a reliability (calibration) curve.

    Args:
        confidences: Predicted confidence values in [0, 1].
        correctness: Binary correctness indicators.
        n_bins: Number of equal-width bins.

    Returns:
        (bin_confidences, bin_accuracies, bin_counts) — arrays of length n_bins.
        Suitable for plotting a calibration diagram.
    """
    _require_numpy()
    return _bin_data(confidences, correctness, n_bins)
