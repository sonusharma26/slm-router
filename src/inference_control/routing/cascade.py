"""Answer-agreement cascades with a calibrated quality guarantee (offline evaluation).

A cascade configuration ``(lam, accept)`` routes each request as follows:

1. ``first`` = argmax(forecast - lam * cost) among allowed endpoints (the price-penalty rule);
   ``final`` = argmax(forecast) (quality first).
2. If ``first`` is ``final``, or ``first``'s forecast is at least ``accept``, serve ``first``.
3. Otherwise also run a checker, the next best endpoint by the same score. If the two
   answers agree after normalization, serve ``first``'s answer; if not, escalate to
   ``final`` (no extra call when ``final`` is the checker).

``accept < 0`` never checks (the plain price-penalty rule); ``accept > 1`` always checks.

Calibration: configurations are ordered by quality on training references (leave-one-out
forecasts with training outcomes and answers), which never see calibration outcomes. A
fixed-sequence Hoeffding-Bentkus test then runs over that order on full-information
calibration requests (every allowed endpoint scored and its answer stored). All rejected
hypotheses hold simultaneously with probability at least ``1 - risk``, so the cheapest
rejected configuration (by calibration cost) can be selected without weakening the
guarantee: mean quality over all routed requests is at least the target.

Agreement is only informative when answers can be compared (extracted final answers,
choices, numbers). Free-form outputs rarely agree, so checks on them mostly escalate.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from inference_control.routing.calibration import hb_pvalue, utility_choice

_SPACE = re.compile(r"\s+")


def normalize_answer(value) -> str | None:
    """Comparable form of an extracted answer; None when there is nothing to compare."""
    if value is None:
        return None
    text = _SPACE.sub(" ", str(value)).strip().lower().rstrip(".")
    return text or None


def agreement_matrix(answers: list[list]) -> np.ndarray:
    """(requests x endpoints x endpoints) boolean: two endpoints gave the same answer."""
    n = len(answers)
    m = len(answers[0]) if n else 0
    codes = np.full((n, m), -1, dtype=np.int64)
    vocabulary: dict[str, int] = {}
    for i, row in enumerate(answers):
        for j, value in enumerate(row):
            text = normalize_answer(value)
            if text is not None:
                codes[i, j] = vocabulary.setdefault(text, len(vocabulary))
    valid = codes >= 0
    return (codes[:, :, None] == codes[:, None, :]) & valid[:, :, None] & valid[:, None, :]


@dataclass(frozen=True)
class CascadeConfig:
    lam: float
    accept: float

    @property
    def checks(self) -> bool:
        return self.accept >= 0


def default_configs(lambdas, accepts=(-1., .9, .8, .7, 2.)) -> tuple[CascadeConfig, ...]:
    return tuple(CascadeConfig(float(lam), float(a)) for lam in lambdas for a in accepts)


def plan(forecasts: np.ndarray, costs: np.ndarray, config: CascadeConfig):
    """Per request: first, checker and final positions (-1 when absent) and whether to check."""
    rows = np.arange(len(forecasts))
    first = utility_choice(forecasts, costs, config.lam)
    final = utility_choice(forecasts, costs, 0.)
    score = np.where(np.isfinite(forecasts), forecasts - config.lam * np.nan_to_num(costs), -np.inf)
    masked = score.copy()
    ok = first >= 0
    masked[rows[ok], first[ok]] = -np.inf
    checker = np.argmax(masked, axis=1)
    checker[~np.isfinite(masked).any(axis=1)] = -1
    forecast_first = np.where(ok, forecasts[rows, np.maximum(first, 0)], -np.inf)
    check = ok & config.checks & (first != final) & (forecast_first < config.accept) & (checker >= 0)
    return first, checker, final, check


def run(forecasts, costs, quality, realized_cost, agree, config: CascadeConfig):
    """Realized quality, cost and escalation per request; unroutable requests give NaN."""
    first, checker, final, check = plan(forecasts, costs, config)
    n = len(forecasts)
    rows = np.arange(n)
    routed = first >= 0
    q = np.full(n, np.nan)
    c = np.full(n, np.nan)
    escalated = np.zeros(n, dtype=bool)
    f, k, z = np.maximum(first, 0), np.maximum(checker, 0), np.maximum(final, 0)
    q[routed] = quality[rows[routed], f[routed]]
    c[routed] = realized_cost[rows[routed], f[routed]]
    idx = rows[check]
    if len(idx):
        agreed = agree[idx, f[idx], k[idx]]
        c[idx] += realized_cost[idx, k[idx]]
        up = idx[~agreed]
        escalated[up] = True
        extra = (z[up] != f[up]) & (z[up] != k[up])
        c[up] += np.where(extra, realized_cost[up, z[up]], 0.)
        q[up] = quality[up, z[up]]
    return q, c, escalated


@dataclass(frozen=True)
class CascadeStep:
    config: CascadeConfig
    requests: int
    mean_quality: float | None
    mean_cost: float | None
    p_value: float
    rejected: bool


@dataclass(frozen=True)
class CascadeCalibration:
    selected: CascadeConfig | None
    target: float
    risk: float
    calibration_requests: int
    steps: tuple[CascadeStep, ...]
    method: str = "hb-fixed-sequence-cascade-v1"

    @property
    def certified_mean(self) -> float | None:
        step = next((s for s in self.steps if s.config == self.selected), None)
        return step.mean_quality if step else None

    def identity(self) -> dict:
        return {"selected": asdict(self.selected) if self.selected else None, "target": self.target,
                "risk": self.risk, "calibration_requests": self.calibration_requests, "method": self.method,
                "steps": [dict(asdict(s), config=asdict(s.config)) for s in self.steps]}


def order_configs(configs, forecasts, costs, quality, realized_cost, agree):
    """Order by reference quality (highest first); references must exclude calibration outcomes."""
    scored = []
    for position, config in enumerate(configs):
        q, c, _ = run(forecasts, costs, quality, realized_cost, agree, config)
        routed = ~np.isnan(q)
        scored.append((-(float(q[routed].mean()) if routed.any() else 0.),
                       float(c[routed].mean()) if routed.any() else 0., position, config))
    return tuple(item[3] for item in sorted(scored))


def calibrate(configs, forecasts, costs, quality, realized_cost, agree, *, target: float, risk: float,
              minimum_requests: int = 30) -> CascadeCalibration:
    """Fixed-sequence test in the given order; select the cheapest rejected configuration."""
    if not 0 < risk < 1:
        raise ValueError("risk must lie in (0, 1)")
    steps, n = [], 0
    for config in configs:
        q, c, _ = run(forecasts, costs, quality, realized_cost, agree, config)
        routed = ~np.isnan(q)
        if np.isnan(c[routed]).any():
            raise ValueError("calibration needs a realized cost for every routed request")
        n = int(routed.sum())
        total = float(q[routed].sum())
        p_value = hb_pvalue(total, n, target) if n >= minimum_requests else 1.
        rejected = p_value <= risk
        steps.append(CascadeStep(config, n, total / n if n else None,
                                 float(c[routed].mean()) if n else None, p_value, rejected))
        if not rejected:
            break
    passed = [s for s in steps if s.rejected]
    selected = min(passed, key=lambda s: (s.mean_cost, -s.mean_quality)).config if passed else None
    return CascadeCalibration(selected, target, risk, n, tuple(steps))
