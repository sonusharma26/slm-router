"""Live answer-agreement cascade: the gated replay that serving and calibration share.

``routing.cascade`` evaluates ``(lam, accept)`` configurations on full-information matrices. Serving
adds hard limits that the offline functions cannot see: the compound plan (first and checker
concurrently, then the quality-first endpoint on disagreement) must satisfy the same spend,
call-limit and deadline gates as any other plan. A request whose compound plan fails a gate does not
check; its first endpoint answers directly. Those gates are evaluated by a caller-supplied ``gate``
function, and this module applies it identically when

* ordering configurations on training references,
* running the fixed-sequence test on calibration requests, and
* deciding one live request (a one-row reference),

so the calibrated population is exactly the population served. Where no gate binds the results equal
``cascade.order_configs`` / ``cascade.calibrate`` (a test asserts this). Configurations are
``CascadeConfig(lam, accept)`` only: vote configurations are never served live.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from inference_control.routing import cascade
from inference_control.routing.calibration import hb_pvalue

Gate = Callable[["CascadeReference", int, int, int, int], bool]


@dataclass
class CascadeReference:
    """Per-request matrices in each request's selection order (position 0 is the best-ranked allowed endpoint)."""
    requests: list
    ids: list                      # per request: allowed endpoint ids by position
    forecasts: np.ndarray          # (n, m) quality forecasts, -inf where there is no endpoint
    costs: np.ndarray              # (n, m) expected costs, NaN where there is no endpoint
    columns: np.ndarray            # (n, m) evidence column of the endpoint at each position, -1 if none
    absolute: np.ndarray           # (n, m) absolute spend bound of the endpoint at each position
    numbers: list                  # per request: pooled_numbers dict by position
    quality: np.ndarray | None = None
    realized: np.ndarray | None = None
    agree: np.ndarray | None = None
    complete: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.requests)

    def subset(self, index) -> CascadeReference:
        index = np.asarray(index, dtype=int)
        pick = lambda a: None if a is None else a[index]
        return CascadeReference([self.requests[i] for i in index], [self.ids[i] for i in index],
                                self.forecasts[index], self.costs[index], self.columns[index], self.absolute[index],
                                [self.numbers[i] for i in index], pick(self.quality), pick(self.realized),
                                pick(self.agree), pick(self.complete))


def gated_plan(reference: CascadeReference, config, gate: Gate):
    """first, checker, final positions, the post-gate check flag and the pre-gate check flag."""
    first, checker, final, raw = cascade.plan(reference.forecasts, reference.costs, config)
    check = raw.copy()
    for i in np.flatnonzero(raw):
        if not gate(reference, int(i), int(first[i]), int(checker[i]), int(final[i])):
            check[i] = False
    return first, checker, final, check, raw


def gated_run(reference: CascadeReference, config, gate: Gate):
    """Realized quality, cost, escalation and check flag per request; unroutable requests give NaN."""
    q, c, escalated = cascade.run(reference.forecasts, reference.costs, reference.quality, reference.realized,
                                  reference.agree, config)
    _, _, _, check, raw = gated_plan(reference, config, gate)
    dropped = np.flatnonzero(raw & ~check)
    if len(dropped):
        # A gated-off request is served by its first endpoint alone: the plain price-penalty rule.
        plain = cascade.CascadeConfig(config.lam, -1.)
        dq, dc, _ = cascade.run(reference.forecasts[dropped], reference.costs[dropped], reference.quality[dropped],
                                reference.realized[dropped], reference.agree[dropped], plain)
        q[dropped], c[dropped], escalated[dropped] = dq, dc, False
    return q, c, escalated, check


def order_gated(configs, reference: CascadeReference, gate: Gate):
    """Order by reference quality (highest first); same key as ``cascade.order_configs``."""
    scored = []
    for position, config in enumerate(configs):
        q, c, _, _ = gated_run(reference, config, gate)
        routed = ~np.isnan(q)
        scored.append((-(float(q[routed].mean()) if routed.any() else 0.),
                       float(c[routed].mean()) if routed.any() else 0., position, config))
    return tuple(item[3] for item in sorted(scored, key=lambda item: item[:3]))


def calibrate_gated(configs, reference: CascadeReference, gate: Gate, *, target: float, risk: float,
                    minimum_requests: int = 30) -> cascade.CascadeCalibration:
    """Fixed-sequence test in the given order; select the cheapest rejected configuration.

    Same procedure as ``cascade.calibrate`` with the compound gates applied to every request.
    """
    if not 0 < risk < 1:
        raise ValueError("risk must lie in (0, 1)")
    steps, n = [], 0
    for config in configs:
        q, c, _, _ = gated_run(reference, config, gate)
        routed = ~np.isnan(q)
        if np.isnan(c[routed]).any():
            raise ValueError("calibration needs a realized cost for every routed request")
        n = int(routed.sum())
        total = float(q[routed].sum())
        p_value = hb_pvalue(total, n, target) if n >= minimum_requests else 1.
        rejected = p_value <= risk
        steps.append(cascade.CascadeStep(config, n, total / n if n else None,
                                         float(c[routed].mean()) if n else None, p_value, rejected))
        if not rejected:
            break
    passed = [s for s in steps if s.rejected]
    selected = min(passed, key=lambda s: (s.mean_cost, -s.mean_quality)).config if passed else None
    return cascade.CascadeCalibration(selected, target, risk, n, tuple(steps))


@dataclass(frozen=True)
class LiveCascade:
    """A calibrated cascade the planner serves: the test result plus what serving needs to replay it."""
    calibration: cascade.CascadeCalibration
    agreement: str                                  # name of the registered agreement function
    endpoint_ids: tuple[str, ...]                   # evidence column order
    disagreement: tuple[tuple[float, ...], ...]     # training P(answers differ), by evidence column pair
    kind: str = "cascade"

    def __post_init__(self):
        if getattr(self.calibration.selected, "voters", None):
            raise ValueError("vote configurations are calibrated offline only and are never served live")

    @property
    def config(self):
        return self.calibration.selected

    # Compatibility with the direct calibrations the planner and estimates already read.
    @property
    def threshold(self):
        return None

    @property
    def parameter(self):
        return self.config.lam if self.config is not None else None

    @property
    def certified_mean(self):
        return self.calibration.certified_mean

    @property
    def calibration_requests(self) -> int:
        return self.calibration.calibration_requests

    @property
    def certified_count(self) -> int:
        step = next((s for s in self.calibration.steps if s.config == self.config), None)
        return step.requests if step else 0

    @property
    def method(self) -> str:
        return self.calibration.method

    @property
    def steps(self):
        return self.calibration.steps

    def identity(self) -> dict:
        return {"kind": self.kind, "agreement": self.agreement, "endpoints": list(self.endpoint_ids),
                "disagreement": [list(row) for row in self.disagreement], "calibration": self.calibration.identity()}
