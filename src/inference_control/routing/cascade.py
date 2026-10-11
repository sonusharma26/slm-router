"""Answer-agreement cascades and majority votes with a calibrated quality guarantee (offline).

A cascade configuration ``(lam, accept)`` routes each request as follows:

1. ``first`` = argmax(forecast - lam * cost) among allowed endpoints (the price-penalty rule);
   ``final`` = argmax(forecast) (quality first).
2. If ``first`` is ``final``, or ``first``'s forecast is at least ``accept``, serve ``first``.
3. Otherwise also run a checker, the next best endpoint by the same score. If the two
   answers agree after normalization, serve ``first``'s answer; if not, escalate to
   ``final`` (no extra call when ``final`` is the checker).

``accept < 0`` never checks (the plain price-penalty rule); ``accept > 1`` always checks.

Majority voting: ``CascadeConfig(lam, accept, voters=k, quorum=q)`` with ``2 <= q <= k``.

1. Voters are the top-``k`` allowed endpoints ranked by ``forecast - lam * cost`` (ties go to
   the earlier position), so rank 1 is ``first``. If ``first``'s forecast is at least
   ``accept`` (or fewer than two endpoints are allowed), serve ``first`` alone; ``accept > 1``
   always votes. Unlike the agreement check, a vote is also held when ``first`` is ``final``.
2. All voters are called. The vote counts normalized answers among voters whose answer is
   comparable (not missing or empty). The plurality answer wins; ties between answers go to the
   one whose best-ranked voter ranks higher.
3. If the winner has at least ``q`` votes, serve it, with the realized quality of its
   best-ranked voter. Otherwise escalate to ``final``: if ``final`` is one of the voters, its
   own answer is served at no extra cost, otherwise ``final`` is called as well. When no
   answers are comparable the count is zero, so the request escalates; a vote never guesses.
4. Cost is the sum of the voters' costs, plus ``final``'s cost when it is called in addition.

With ``sequential=True`` voters are called in rank order and calling stops as soon as one
answer reaches ``q`` votes (that answer is served) or no answer can reach ``q`` any more
(escalate). The served answer is then the first to reach ``q``; it equals the plurality winner
whenever ``2q > k``. Calls are saved at the price of latency.

With parallel calls, ``voters=2, quorum=2`` is the agreement cascade with the check always on,
except that it also votes when ``first`` is ``final``.

Calibration: configurations are ordered by quality on training references (leave-one-out
forecasts with training outcomes and answers), which never see calibration outcomes. A
fixed-sequence Hoeffding-Bentkus test then runs over that order on full-information
calibration requests (every allowed endpoint scored and its answer stored). All rejected
hypotheses hold simultaneously with probability at least ``1 - risk``, so the cheapest
rejected configuration (by calibration cost) can be selected without weakening the
guarantee: mean quality over all routed requests is at least the target. Plain routing,
agreement cascades and votes are all deterministic functions of the stored matrices, so one
mixed family is covered by the same argument.

Agreement is only informative when answers can be compared (extracted final answers,
choices, numbers). Free-form outputs rarely agree, so checks and votes on them mostly escalate.
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
    """One routing configuration.

    ``voters == 0`` is the agreement cascade (``accept < 0``: plain price-penalty routing).
    ``voters >= 2`` is a majority vote that needs ``quorum`` matching answers (default: a
    strict majority); see the module docstring. ``sequential`` only applies to votes.
    """
    lam: float
    accept: float
    voters: int = 0
    quorum: int = 0
    sequential: bool = False

    def __post_init__(self):
        if self.voters == 0:
            if self.quorum or self.sequential:
                raise ValueError("quorum and sequential only apply to votes (voters >= 2)")
            return
        if self.voters < 2:
            raise ValueError("a vote needs at least two voters")
        quorum = self.quorum or self.voters // 2 + 1
        if not 2 <= quorum <= self.voters:
            raise ValueError("quorum must lie between 2 and the number of voters")
        if self.accept < 0:
            raise ValueError("a vote needs accept >= 0 (accept > 1 always votes)")
        object.__setattr__(self, "quorum", quorum)

    @property
    def checks(self) -> bool:
        return self.accept >= 0

    @property
    def kind(self) -> str:
        return "vote" if self.voters else "agree" if self.checks else "plain"


def config_dict(config: CascadeConfig) -> dict:
    """Serializable form; agreement and plain configurations keep their original two fields."""
    if config.voters:
        return asdict(config)
    return {"lam": config.lam, "accept": config.accept}


def vote_lambdas(lambdas, count: int = 8) -> tuple[float, ...]:
    """Evenly spaced subset (by position) of a penalty grid, always keeping both ends."""
    values = tuple(float(x) for x in lambdas)
    if len(values) <= count:
        return values
    positions = np.unique(np.round(np.linspace(0, len(values) - 1, count)).astype(int))
    return tuple(values[i] for i in positions)


def default_quorums(voters: int) -> tuple[int, ...]:
    """Strict majority and the next stricter quorum (3 voters: 2, 3; 5 voters: 3, 4)."""
    majority = voters // 2 + 1
    return tuple(q for q in (majority, majority + 1) if q <= voters)


def vote_configs(lambdas, voters=(3, 5), *, quorums=None, accepts=(2.,), sequential=(True,),
                 count: int = 8) -> tuple[CascadeConfig, ...]:
    """Vote configurations over ``count`` evenly spaced penalties (invalid quorums are skipped).

    Sequential calling is the default for families: with a strict-majority quorum it serves the same
    answer as calling all voters at once and never costs more. Pass ``sequential=(False,)`` (or both)
    to evaluate parallel calls, which trade cost for latency.
    """
    chosen = vote_lambdas(lambdas, count)
    configs = []
    for lam in chosen:
        for k in voters:
            for quorum in (quorums or default_quorums(k)):
                if not 2 <= quorum <= k:
                    continue
                for accept in accepts:
                    for seq in sequential:
                        configs.append(CascadeConfig(float(lam), float(accept), int(k), int(quorum), bool(seq)))
    return tuple(configs)


def default_configs(lambdas, accepts=(-1., .9, .8, .7, 2.), *, voters=(), vote_accepts=(2.,),
                    vote_quorums=None, vote_sequential=(True,), vote_count: int = 8) -> tuple[CascadeConfig, ...]:
    """Agreement and plain configurations over ``lambdas``; ``voters`` appends vote configurations.

    With the default ``voters=()`` the result is exactly the agreement/plain family.
    """
    base = tuple(CascadeConfig(float(lam), float(a)) for lam in lambdas for a in accepts)
    if not voters:
        return base
    return base + vote_configs(lambdas, voters, quorums=vote_quorums, accepts=vote_accepts,
                               sequential=vote_sequential, count=vote_count)


def plan(forecasts: np.ndarray, costs: np.ndarray, config: CascadeConfig):
    """Per request: first, checker and final positions (-1 when absent) and whether to check.

    For a vote ``check`` means a vote is held (``first`` is then rank 1 and ``checker`` rank 2).
    """
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
    check = ok & config.checks & (forecast_first < config.accept) & (checker >= 0)
    if not config.voters:
        check &= first != final
    return first, checker, final, check


@dataclass(frozen=True)
class CascadeOutcome:
    """Per-request realized results of one configuration (NaN quality/cost when unroutable)."""
    quality: np.ndarray
    cost: np.ndarray
    escalated: np.ndarray   # served by ``final`` after a disagreement or a missed quorum
    calls: np.ndarray       # endpoints invoked (0 when unroutable)
    checked: np.ndarray     # an agreement check or a vote was held
    consensus: np.ndarray   # served answer backed by an agreeing checker or a quorum


def _vote(forecasts, costs, quality, realized_cost, agree, config, final, idx, out):
    """Resolve votes for the requests in ``idx`` (see the module docstring); fills ``out`` in place."""
    q, c, escalated, calls, consensus = out
    m = len(idx)
    k = min(config.voters, forecasts.shape[1])
    f, w = forecasts[idx], costs[idx]
    score = np.where(np.isfinite(f), f - config.lam * np.nan_to_num(w), -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    valid = np.isfinite(np.take_along_axis(score, order, axis=1))  # invalid voters sort last
    same = (agree[idx[:, None, None], order[:, :, None], order[:, None, :]]
            & valid[:, :, None] & valid[:, None, :])
    rows = np.arange(m)
    if config.sequential:
        called = np.zeros((m, k), dtype=bool)
        served = np.zeros(m, dtype=bool)
        done = np.zeros(m, dtype=bool)
        lead = np.zeros(m, dtype=np.int64)
        best = np.zeros(m, dtype=np.int64)
        available = valid.sum(1)
        for j in range(k):
            live = valid[:, j] & ~done
            called[:, j] = live
            group = same[:, j, : j + 1]
            size = group.sum(1)
            best = np.where(live, np.maximum(best, size), best)
            hit = live & (size >= config.quorum)
            lead = np.where(hit, group.argmax(1), lead)
            served |= hit
            done |= hit | (live & (best + (available - j - 1) < config.quorum))
    else:
        called = valid
        votes = same.sum(2)                 # voters behind each voter's answer (0 if not comparable)
        lead = votes.argmax(1)              # best-ranked voter among the leading answers
        best = votes[rows, lead]
        served = (best >= config.quorum) & (best > 0)
    winner = order[rows, lead]
    fin = final[idx]
    final_called = ((order == fin[:, None]) & called).any(1)
    extra = ~served & ~final_called
    spent = np.where(called, realized_cost[idx[:, None], order], 0.).sum(1)
    c[idx] = spent + np.where(extra, realized_cost[idx, fin], 0.)
    q[idx] = np.where(served, quality[idx, winner], quality[idx, fin])
    escalated[idx] = ~served
    calls[idx] = called.sum(1) + extra
    consensus[idx] = served


def outcome(forecasts, costs, quality, realized_cost, agree, config: CascadeConfig) -> CascadeOutcome:
    """Realized quality, cost, calls and decision flags per request; unroutable requests give NaN."""
    first, checker, final, check = plan(forecasts, costs, config)
    n = len(forecasts)
    rows = np.arange(n)
    routed = first >= 0
    q = np.full(n, np.nan)
    c = np.full(n, np.nan)
    escalated = np.zeros(n, dtype=bool)
    calls = np.zeros(n, dtype=np.int64)
    consensus = np.zeros(n, dtype=bool)
    f, k, z = np.maximum(first, 0), np.maximum(checker, 0), np.maximum(final, 0)
    q[routed] = quality[rows[routed], f[routed]]
    c[routed] = realized_cost[rows[routed], f[routed]]
    calls[routed] = 1
    idx = rows[check]
    if len(idx) and config.voters:
        _vote(forecasts, costs, quality, realized_cost, agree, config, final, idx,
              (q, c, escalated, calls, consensus))
    elif len(idx):
        agreed = agree[idx, f[idx], k[idx]]
        c[idx] += realized_cost[idx, k[idx]]
        calls[idx] += 1
        consensus[idx[agreed]] = True
        up = idx[~agreed]
        escalated[up] = True
        extra = (z[up] != f[up]) & (z[up] != k[up])
        c[up] += np.where(extra, realized_cost[up, z[up]], 0.)
        calls[up] += extra
        q[up] = quality[up, z[up]]
    return CascadeOutcome(q, c, escalated, calls, check, consensus)


def run(forecasts, costs, quality, realized_cost, agree, config: CascadeConfig):
    """Realized quality, cost and escalation per request; unroutable requests give NaN."""
    result = outcome(forecasts, costs, quality, realized_cost, agree, config)
    return result.quality, result.cost, result.escalated


def diagnostics(result: CascadeOutcome, config: CascadeConfig) -> dict:
    """Reporting summary over routed requests: kind, vote shape and decision rates.

    ``check_rate``: share with an agreement check or a vote; ``consensus_rate``: share served by
    an agreeing checker or a quorum; ``vote_rate``: share served by a quorum (votes only);
    ``escalation_rate``: share served by ``final`` after disagreement or a missed quorum;
    ``mean_calls``: endpoints invoked per request.
    """
    routed = ~np.isnan(result.quality)

    def rate(flags) -> float:
        return float(flags[routed].mean()) if routed.any() else 0.
    consensus = rate(result.consensus)
    return {"kind": config.kind, "lam": config.lam, "accept": config.accept, "voters": config.voters,
            "quorum": config.quorum, "sequential": config.sequential, "check_rate": rate(result.checked),
            "consensus_rate": consensus, "vote_rate": consensus if config.voters else 0.,
            "escalation_rate": rate(result.escalated),
            "mean_calls": float(result.calls[routed].mean()) if routed.any() else 0.}


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
        return {"selected": config_dict(self.selected) if self.selected else None, "target": self.target,
                "risk": self.risk, "calibration_requests": self.calibration_requests, "method": self.method,
                "steps": [dict(asdict(s), config=config_dict(s.config)) for s in self.steps]}


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
