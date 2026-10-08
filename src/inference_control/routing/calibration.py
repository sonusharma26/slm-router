"""Pooled learn-then-test calibration of a routing threshold.

A routing rule certifies a request when some allowed endpoint's forecast is at least a
threshold ``t``. Among certified endpoints it picks the first in the planner's selection
order. For each candidate ``t`` the null hypothesis is

    H_t:  E[ quality of the chosen endpoint | request is certified at t ] < target.

Certification depends only on request features and training-derived forecasts, so the
certified calibration requests are an i.i.d. sample from that conditional population.
Each hypothesis is tested with the Hoeffding-Bentkus p-value for bounded scores
(Bates, Angelopoulos, Lei, Malik and Jordan, 2021). Thresholds are tested in a fixed,
pre-declared order from strict to loose and testing stops at the first non-rejection
(learn-then-test, Angelopoulos et al., 2021). The family-wise error is at most ``risk``,
so with probability at least ``1 - risk`` the mean quality over all certified traffic at
the selected threshold is at least ``target``. The guarantee concerns that population
mean, not any individual answer, and assumes future traffic is exchangeable with the
calibration requests.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

import numpy as np

METHOD = "hb-fixed-sequence-v1"
DEFAULT_FRACTIONS = tuple(round(.05 * i, 2) for i in range(1, 21))


def bernoulli_kl(p: float, q: float) -> float:
    if p == q:
        return 0.
    if q <= 0. or q >= 1.:
        return math.inf
    value = 0.
    if p > 0:
        value += p * math.log(p / q)
    if p < 1:
        value += (1 - p) * math.log((1 - p) / (1 - q))
    return max(0., value)


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k), summed exactly in log space."""
    if k < 0:
        return 0.
    if k >= n:
        return 1.
    if p <= 0.:
        return 1.
    if p >= 1.:
        return 0.
    log_p, log_q = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    terms = [base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q for i in range(k + 1)]
    peak = max(terms)
    return min(1., math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


def hb_pvalue(quality_sum: float, n: int, target: float) -> float:
    """Valid p-value for H0: mean < target, from n i.i.d. scores in [0, 1] summing to quality_sum."""
    if n <= 0:
        return 1.
    if target <= 0.:
        return 0.
    if target >= 1.:
        return 1.
    alpha = 1. - target                      # risk level on the loss 1 - quality
    risk = max(0., min(1., (n - quality_sum) / n))
    hoeffding = math.exp(-n * bernoulli_kl(min(risk, alpha), alpha))
    # Rounding the loss count up only makes the p-value larger, i.e. more conservative.
    bentkus = math.e * binomial_cdf(math.ceil(n - quality_sum - 1e-12), n, alpha)
    return min(1., hoeffding, bentkus)


@dataclass(frozen=True)
class ThresholdStep:
    threshold: float
    certified: int
    certified_mean: float | None
    p_value: float
    rejected: bool


@dataclass(frozen=True)
class ThresholdCalibration:
    threshold: float | None
    target: float
    risk: float
    calibration_requests: int
    minimum_certified: int
    steps: tuple[ThresholdStep, ...]
    method: str = METHOD

    @property
    def certified_mean(self) -> float | None:
        accepted = [s for s in self.steps if s.rejected]
        return accepted[-1].certified_mean if accepted else None

    @property
    def certified_count(self) -> int:
        accepted = [s for s in self.steps if s.rejected]
        return accepted[-1].certified if accepted else 0

    def identity(self) -> dict:
        value = asdict(self)
        value["steps"] = [asdict(step) for step in self.steps]
        return value


def threshold_grid(max_forecasts, fractions=DEFAULT_FRACTIONS, all_forecasts=None,
                   tail_steps: int = 40) -> tuple[float, ...]:
    """Strict-to-loose thresholds built only from outcome-independent reference forecasts.

    Phase one picks thresholds at which about each fraction of reference requests is
    certified (their best allowed forecast clears it). Once every request is certified,
    lowering the threshold still lets cheaper endpoints qualify, so phase two continues
    through quantiles of all allowed forecasts below the last phase-one threshold. The
    references must be independent of calibration outcomes (validation requests or
    leave-one-out training forecasts). A fixed sequence pays no multiplicity cost per
    extra step; it only stops earlier.
    """
    values = np.sort(np.asarray([v for v in np.asarray(max_forecasts, dtype=float) if np.isfinite(v)]))[::-1]
    if not len(values):
        return ()
    grid = []
    for fraction in sorted(set(fractions)):
        if not 0 < fraction <= 1:
            raise ValueError("certified fractions must lie in (0, 1]")
        index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
        threshold = float(values[index])
        if not grid or threshold < grid[-1] - 1e-12:
            grid.append(threshold)
    if all_forecasts is not None and tail_steps > 0:
        pool = np.asarray([v for v in np.asarray(all_forecasts, dtype=float).ravel() if np.isfinite(v)])
        lower = np.sort(pool[pool < grid[-1] - 1e-12])[::-1]
        if len(lower):
            for step in range(1, tail_steps + 1):
                index = min(len(lower) - 1, max(0, math.ceil(step / tail_steps * len(lower)) - 1))
                threshold = float(lower[index])
                if threshold < grid[-1] - 1e-12:
                    grid.append(threshold)
    return tuple(grid)


def first_certified(ordered_forecasts: np.ndarray, thresholds) -> np.ndarray:
    """Position of the first endpoint (in selection order) whose forecast >= threshold.

    ``ordered_forecasts`` is (requests x endpoints) in each request's selection order, with
    disallowed endpoints set to -inf. Returns (thresholds x requests); ``m`` means none.
    """
    forecasts = np.where(np.isnan(ordered_forecasts), -np.inf, ordered_forecasts)
    prefix = np.maximum.accumulate(forecasts, axis=1)
    thresholds = np.asarray(thresholds, dtype=float)
    return (prefix[None, :, :] < thresholds[:, None, None]).sum(axis=2)


def anchor_start(reference_forecasts: np.ndarray, reference_values: np.ndarray, thresholds, *,
                 calibration_size: int, target: float, risk: float, minimum_certified: int = 30,
                 z: float = 1.645, floor: int = 40) -> int:
    """Index of the first threshold to test, chosen without calibration outcomes.

    Strict thresholds certify few requests, so a fixed sequence that opens there can stop
    before reaching thresholds it could certify. Using reference requests independent of
    calibration (outcomes where known, otherwise forecasts), this predicts each threshold's
    certified count at the calibration size and a pessimistic certified mean (reference mean
    minus ``z`` standard errors at that count). It returns the first threshold whose
    pessimistic p-value is at most ``risk``; if none qualifies, the most promising one.
    Starting later never weakens the guarantee: only tested thresholds can be selected.
    """
    thresholds = tuple(thresholds)
    if not thresholds or not len(reference_forecasts):
        return 0
    n, m = reference_forecasts.shape
    positions = first_certified(reference_forecasts, thresholds)
    rows = np.arange(n)
    values = np.where(np.isnan(reference_values), reference_forecasts, reference_values)
    best_index, best_p = 0, math.inf
    for index in range(len(thresholds)):
        chosen = positions[index]
        certified = chosen < m
        if not certified.any():
            continue
        scores = values[rows[certified], chosen[certified]]
        scores = scores[np.isfinite(scores)]
        if not len(scores):
            continue
        expected = int(certified.mean() * calibration_size)
        if expected < max(minimum_certified, floor):
            continue
        average = float(np.clip(scores, 0, 1).mean())
        pessimistic = max(0., average - z * math.sqrt(max(average * (1 - average), 1e-4) / expected))
        p_value = hb_pvalue(pessimistic * expected, expected, target)
        if p_value <= risk:
            return index
        if p_value < best_p:
            best_index, best_p = index, p_value
    return best_index


def calibrate_threshold(ordered_forecasts: np.ndarray, ordered_quality: np.ndarray, thresholds, *,
                        target: float, risk: float, minimum_certified: int = 30) -> ThresholdCalibration:
    """Fixed-sequence learn-then-test over strict-to-loose thresholds."""
    if not 0 < risk < 1:
        raise ValueError("risk must lie in (0, 1)")
    if ordered_forecasts.shape != ordered_quality.shape:
        raise ValueError("forecast and quality matrices must align")
    thresholds = tuple(float(t) for t in thresholds)
    if any(b > a for a, b in pairwise(thresholds)):
        raise ValueError("thresholds must be ordered strict to loose")
    n, m = ordered_forecasts.shape
    steps, selected = [], None
    if n and thresholds:
        positions = first_certified(ordered_forecasts, thresholds)
        rows = np.arange(n)
        for index, threshold in enumerate(thresholds):
            chosen = positions[index]
            certified = chosen < m
            values = ordered_quality[rows[certified], chosen[certified]]
            if np.isnan(values).any():
                raise ValueError("calibration requests need outcomes for every allowed endpoint")
            count = int(certified.sum())
            total = float(values.sum()) if count else 0.
            p_value = hb_pvalue(total, count, target) if count >= minimum_certified else 1.
            rejected = p_value <= risk
            steps.append(ThresholdStep(threshold, count, total / count if count else None, p_value, rejected))
            if not rejected:
                break
            selected = threshold
    return ThresholdCalibration(selected, target, risk, n, minimum_certified, tuple(steps))


@dataclass(frozen=True)
class SequenceStep:
    parameter: float | str
    requests: int
    mean_quality: float | None
    p_value: float
    rejected: bool


@dataclass(frozen=True)
class SequenceCalibration:
    """Fixed-sequence learn-then-test over whole routing rules (every routed request counts).

    ``selected`` is the index of the last rejected rule in the declared order, or None.
    With probability at least ``1 - risk`` the mean quality over all requests routed by
    the selected rule is at least ``target``.
    """
    selected: int | None
    parameters: tuple
    target: float
    risk: float
    calibration_requests: int
    minimum_requests: int
    steps: tuple[SequenceStep, ...]
    kind: str = "utility"
    method: str = "hb-fixed-sequence-rules-v1"

    @property
    def threshold(self):  # threshold-rule compatibility: rule calibrations certify by parameter, not threshold
        return None

    @property
    def parameter(self):
        return self.parameters[self.selected] if self.selected is not None else None

    @property
    def certified_mean(self) -> float | None:
        return self.steps[self.selected].mean_quality if self.selected is not None else None

    @property
    def certified_count(self) -> int:
        return self.steps[self.selected].requests if self.selected is not None else 0

    def identity(self) -> dict:
        value = asdict(self)
        value["steps"] = [asdict(step) for step in self.steps]
        return value


def calibrate_sequence(values, parameters, *, target: float, risk: float, minimum_requests: int = 30,
                       kind: str = "utility") -> SequenceCalibration:
    """Test rules in their declared order (safest first); stop at the first non-rejection.

    ``values[k]`` holds the realized quality of every calibration request under rule k.
    The order must be fixed without calibration outcomes.
    """
    if not 0 < risk < 1:
        raise ValueError("risk must lie in (0, 1)")
    if len(values) != len(parameters):
        raise ValueError("one value array per rule required")
    steps, selected, n = [], None, 0
    for index, (rule_values, parameter) in enumerate(zip(values, parameters)):
        array = np.asarray(rule_values, dtype=float)
        if np.isnan(array).any():
            raise ValueError("calibration needs a realized quality for every routed request")
        n = len(array)
        total = float(array.sum())
        p_value = hb_pvalue(total, n, target) if n >= minimum_requests else 1.
        rejected = p_value <= risk
        steps.append(SequenceStep(parameter, n, total / n if n else None, p_value, rejected))
        if not rejected:
            break
        selected = index
    return SequenceCalibration(selected, tuple(parameters), target, risk, n, minimum_requests, tuple(steps), kind)


def utility_choice(forecasts: np.ndarray, costs: np.ndarray, lam: float) -> np.ndarray:
    """Position of argmax(forecast - lam * cost) per row; ties resolve to the earliest position.

    Disallowed positions carry forecast -inf. Rows with no allowed position return -1.
    """
    score = np.where(np.isfinite(forecasts), forecasts - lam * np.nan_to_num(costs), -np.inf)
    choice = np.argmax(score, axis=1)
    choice[~np.isfinite(score).any(axis=1)] = -1
    return choice


def utility_grid(forecasts, costs, steps: int = 120) -> tuple[float, ...]:
    """Price penalties from 0 (quality first) upward, at the points where choices switch.

    For every reference request and pair of allowed endpoints where the dearer one has the
    higher forecast, the rule switches between them at (f_a - f_b) / (c_a - c_b). The grid
    takes quantiles of those switching points. It uses forecasts and expected costs of
    outcome-independent references only; ascending, so the sequence runs from the most
    expensive rule to the cheapest.
    """
    F = np.asarray(forecasts, dtype=float)
    C = np.nan_to_num(np.asarray(costs, dtype=float), nan=np.inf)
    if F.size == 0:
        return (0.,)
    df = F[:, :, None] - F[:, None, :]
    dc = C[:, :, None] - C[:, None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where((df > 0) & (dc > 0) & np.isfinite(df) & np.isfinite(dc), df / dc, np.nan)
    points = ratio[np.isfinite(ratio)]
    if not len(points):
        return (0.,)
    grid = np.unique(np.quantile(points, np.linspace(0, 1, steps)))
    return (0.,) + tuple(float(x) * (1 + 1e-9) for x in grid if x > 0)
