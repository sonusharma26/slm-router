"""Request-conditioned, auditable kNN with separate calibration evidence.

Bounds concern the local traffic population under the recorded assumptions, not an
individual prompt. No test label, oracle answer or post-response feature enters routing.
"""
from __future__ import annotations
from datetime import datetime, timezone
import math
from statistics import mean
from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from inference_control.contracts import FrozenContract, RequestContext, Prediction, EndpointSnapshot
from inference_control.util import digest


@lru_cache(maxsize=1024)
def quantile_tolerance_rank(n: int, quantile: float, alpha: float) -> int | None:
    """Exact binomial one-sided tolerance bound for an IID population quantile.

    Return the 1-based order statistic whose chance of lying BELOW the population
    quantile is at most alpha. None means even the maximum lacks enough evidence.
    """
    if n < 1: return None
    cdf = 0.
    for i in range(n):
        log_p = (math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                 + i*math.log(quantile)+(n-i)*math.log1p(-quantile))
        cdf += math.exp(log_p)
        if cdf >= 1-alpha+1e-12:
            return i+1
    return None


def stratum(request: RequestContext) -> str:
    """Exact semantic/governance fields and coarse length buckets; no output leakage."""
    return digest({
        "application": request.application_id, "task": request.task_hint,
        "slices": request.traffic_slices, "tags": request.tags,
        "modality": request.modality, "capabilities": request.required_capabilities,
        "feature_version": request.feature_version, "privacy": request.privacy_classification,
        "input_bucket": int(math.log2(max(1, request.input_tokens))),
        "output_bucket": int(math.log2(max(1, request.max_output_tokens))),
        "json_schema": request.structured_output_schema_hash, "tools": request.tool_schema_hash,
    })


class EvidenceObservation(FrozenContract):
    sample_id: str
    request: RequestContext
    target_id: str
    plan_type: str = "direct"
    endpoint_ids: tuple[str, ...]
    endpoint_revisions: tuple[str, ...]
    quality: float = Field(ge=0, le=1)
    cost: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    output_tokens: int = Field(default=0, ge=0)
    failed: bool = False
    split: Literal["train", "calibration"]
    evaluator_type: Literal["deterministic", "application", "human", "calibrated_judge", "proxy", "benchmark"]
    evaluator_version: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    trusted: bool = True
    trusted_until: datetime | None = None
    certification_eligible: bool = True
    disputed: bool = False
    outcome_id: str | None = None
    metrics: frozenset[str] = frozenset({"quality", "cost", "latency", "failure"})

    @model_validator(mode="after")
    def shape(self):
        if len(self.endpoint_ids) != len(self.endpoint_revisions) or not self.endpoint_ids:
            raise ValueError("evidence needs every endpoint and its capability revision")
        if self.observed_at.tzinfo is None:
            raise ValueError("observation timestamp must have a timezone")
        if self.evaluator_type == "proxy" and self.trusted:
            raise ValueError("proxy evidence cannot certify")
        if self.evaluator_type == "calibrated_judge" and self.trusted and self.trusted_until is None:
            raise ValueError("trusted judge evidence requires a trust expiration")
        if self.trusted_until is not None and self.trusted_until.tzinfo is None:
            raise ValueError("trust expiration must have a timezone")
        if self.evaluator_type == "benchmark" and self.certification_eligible:
            raise ValueError("benchmark-only labels cannot certify production decisions")
        return self


class EvidenceLineage(FrozenContract):
    target_id: str
    traffic_stratum: str
    feature_version: str
    sample_size: int
    calibration_size: int
    training_hash: str
    calibration_hash: str
    artifact_hash: str
    endpoint_revisions: tuple[str, ...]
    measured_at: datetime
    age_seconds: float
    calibrated: bool
    trusted_until: datetime | None = None
    reasons: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = (
        "Quality scores are bounded in [0,1].",
        "Distinct calibration requests are independent and exchangeable within this neighborhood.",
        "Neighborhood selection depends on pre-response features, not labels.",
        "Traffic, evaluator and endpoint configuration remain stable until expiry.",
        "Latency calibration samples are IID within the selected local population; p95 uses a binomial tolerance bound.",
        "Confidence applies to a local population mean, not guaranteed per-prompt correctness.",
        "One fixed evidence review; repeated adaptive policy searches require fresh calibration.",
    )


class ConditionalCapabilityMap:
    def __init__(self, *, k: int = 512, min_samples: int = 30, max_age_seconds: int = 86400):
        if k < 1 or min_samples < 1 or max_age_seconds < 1:
            raise ValueError("positive estimator bounds required")
        self.k, self.min_samples, self.max_age_seconds = k, min_samples, max_age_seconds
        self.rows: dict[str, EvidenceObservation] = {}
        self.invalidations: list[dict] = []
        self._version: str | None = None
        self._index: dict[tuple, list[EvidenceObservation]] | None = None
        self._fingerprints: dict[str, tuple[EvidenceObservation, str]] = {}

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = "map:" + digest({"format":"content-addressed-knn-v2", "k":self.k,
                "min_samples":self.min_samples,"max_age_seconds":self.max_age_seconds,
                "rows":{key:self._row_hash(self.rows[key]) for key in sorted(self.rows)},
                "invalidations":self.invalidations})
        return self._version

    def _row_hash(self, row: EvidenceObservation) -> str:
        cached=self._fingerprints.get(row.sample_id)
        if cached is None or cached[0] is not row:
            cached=(row,digest(row))
            self._fingerprints[row.sample_id]=cached
        return cached[1]

    def _lineage_hash(self, rows) -> str:
        return digest([(r.sample_id,self._row_hash(r)) for r in rows])

    def add(self, row: EvidenceObservation) -> None:
        # A prompt cannot migrate between training and calibration, even across endpoints.
        for prior in self.rows.values():
            if prior.request.request_id == row.request.request_id and prior.split != row.split:
                raise ValueError("request appears in training and calibration")
        prior = self.rows.get(row.sample_id)
        if prior and prior != row:
            raise ValueError("immutable sample_id conflict; use supersede_outcome")
        self.rows[row.sample_id] = row
        self._version = self._index = None

    def add_many(self, rows: list[EvidenceObservation]) -> None:
        splits = {r.request.request_id: r.split for r in self.rows.values()}
        samples = dict(self.rows)
        for row in rows:
            if row.request.request_id in splits and splits[row.request.request_id] != row.split:
                raise ValueError("request appears in training and calibration")
            splits[row.request.request_id] = row.split
            prior = samples.get(row.sample_id)
            if prior and prior != row:
                raise ValueError("immutable sample_id conflict")
            samples[row.sample_id] = row
        self.rows.update({r.sample_id: r for r in rows})
        self._version = self._index = None

    def supersede_outcome(self, outcome_id: str) -> None:
        for key, row in tuple(self.rows.items()):
            if row.outcome_id == outcome_id:
                self.rows[key] = row.model_copy(update={"disputed": True, "trusted": False})
        self._version = self._index = None

    def invalidate(self, endpoint_id: str, metrics: tuple[str, ...], *,
                   at: datetime, slice_id: str | None = None, reason: str = "drift") -> None:
        self.invalidations.append({"endpoint_id": endpoint_id, "metrics": list(metrics),
                                   "at": at.isoformat(), "slice_id": slice_id, "reason": reason})
        self._version = None

    def export(self) -> dict:
        return {"k": self.k, "min_samples": self.min_samples, "max_age_seconds": self.max_age_seconds,
                "rows": [self.rows[k].model_dump(mode="json") for k in sorted(self.rows)],
                "invalidations": self.invalidations}

    @classmethod
    def restore(cls, data: dict) -> ConditionalCapabilityMap:
        result = cls(k=data["k"], min_samples=data["min_samples"], max_age_seconds=data["max_age_seconds"])
        result.add_many([EvidenceObservation.model_validate(r) for r in data["rows"]])
        result.invalidations = list(data.get("invalidations", []))
        return result

    def _valid(self, row: EvidenceObservation, metric: str, at: datetime) -> bool:
        if metric not in row.metrics or row.disputed:
            return False
        if metric in {"quality", "failure"} and (not row.trusted or (row.trusted_until is not None and at >= row.trusted_until)):
            return False
        for item in self.invalidations:
            if item["endpoint_id"] not in row.endpoint_ids or metric not in item["metrics"]:
                continue
            if item["slice_id"] and item["slice_id"] not in row.request.traffic_slices:
                continue
            if row.observed_at <= datetime.fromisoformat(item["at"]):
                return False
        return True

    def _neighbors(self, request, target_id, revisions, at, max_age):
        if self._index is None:
            self._index = {}
            for row in self.rows.values():
                key = (row.target_id, stratum(row.request), row.endpoint_revisions)
                self._index.setdefault(key, []).append(row)
        candidates = self._index.get((target_id, stratum(request), tuple(revisions)), [])
        candidates = [r for r in candidates if
                      0 <= (at - r.observed_at).total_seconds() <= max_age
                      and len(r.request.query_features) == len(request.query_features)]
        def distance(row):
            return (sum((x-y)**2 for x,y in zip(row.request.query_features, request.query_features)),
                    row.request.request_id, row.sample_id)
        return sorted(candidates, key=distance)

    def estimate(self, request: RequestContext, target_id: str,
                 endpoints: tuple[EndpointSnapshot, ...], *, at: datetime | None = None,
                 quality_risk: float = 0.05, latency_risk: float = 0.05,
                 min_samples: int | None = None, max_age_seconds: int | None = None,
                 comparisons: int = 1):
        from inference_control.planning.planner import CapabilityEstimate
        at = at or datetime.now(timezone.utc)
        revisions = tuple(e.capability_revision for e in endpoints)
        rows = self._neighbors(request, target_id, revisions, at,
                               min(self.max_age_seconds, max_age_seconds or self.max_age_seconds))
        if not rows:
            return None
        minimum = max(self.min_samples, min_samples or self.min_samples)
        def subset(split, metric):
            seen, chosen = set(), []
            for row in rows:
                if row.split != split or not self._valid(row, metric, at):
                    continue
                if row.request.request_id in seen:
                    continue
                seen.add(row.request.request_id)
                chosen.append(row)
                if len(chosen) == self.k:
                    break
            return chosen
        train = subset("train", "quality")
        calibration = subset("calibration", "quality")
        if not train and not calibration:
            return None
        alpha = quality_risk / max(1, comparisons)
        n = len(calibration)
        center = mean(r.quality for r in (train or calibration))
        if n:
            local_mean = mean(r.quality for r in calibration)
            radius = math.sqrt(math.log(2 / alpha) / (2*n))
            lower, upper = max(0., local_mean-radius), min(1., local_mean+radius)
        else:
            lower, upper = 0., 1.
        quality = Prediction(mean=center, lower=min(lower, center), upper=max(upper, center))
        # Direct cost is repriced from current immutable token prices; no stale-dollar reuse.
        absolute = sum(((request.input_tokens + e.input_token_overhead) * e.input_price_per_million
                        + request.max_output_tokens * e.output_price_per_million) / 1e6 for e in endpoints)
        cost_rows = subset("train", "cost") or subset("calibration", "cost")
        if len(endpoints) == 1:
            output = request.expected_output_tokens
            if output is None:
                output = mean(r.output_tokens for r in cost_rows) if cost_rows else request.max_output_tokens
            e = endpoints[0]
            expected = ((request.input_tokens + e.input_token_overhead) * e.input_price_per_million
                        + min(request.max_output_tokens, output) * e.output_price_per_million) / 1e6
        else:
            # Joint cost rows depend on price snapshots; callers invalidate cost on price drift.
            expected = mean(r.cost for r in cost_rows) if cost_rows else absolute
        cost = Prediction(mean=expected, lower=min(expected, 0.), upper=max(absolute, expected))
        latency_cal = subset("calibration", "latency")
        latency_train = subset("train", "latency")
        latency_rows = latency_train or latency_cal
        lat_mean = mean(r.latency_ms for r in latency_rows) if latency_rows else 0.
        values = sorted(r.latency_ms for r in latency_cal)
        # A confidence bound ON the population p95, not merely a marginal 95% prediction interval.
        rank = quantile_tolerance_rank(len(values),.95,latency_risk/max(1, comparisons))
        tail_supported = rank is not None
        lat_upper = values[rank-1] if tail_supported else max((r.latency_ms for r in latency_rows), default=5e299) * 2
        latency = Prediction(mean=lat_mean, lower=0, upper=max(lat_mean, lat_upper))
        failure_rows = subset("calibration", "failure")
        nf = len(failure_rows)
        pf = mean(float(r.failed) for r in failure_rows) if nf else 0.5
        fr = math.sqrt(math.log(2/alpha)/(2*nf)) if nf else 1.
        failure = Prediction(mean=pf, lower=max(0., pf-fr), upper=min(1., pf+fr))
        reasons = []
        if not latency_rows: reasons.append("LATENCY_MISSING")
        if len(train) < minimum: reasons.append("TRAINING_COVERAGE")
        if n < minimum: reasons.append("CALIBRATION_COVERAGE")
        if not tail_supported: reasons.append("LATENCY_TAIL_COVERAGE")
        if nf < minimum: reasons.append("FAILURE_COVERAGE")
        if len(endpoints) > 1 and not cost_rows: reasons.append("JOINT_COST_COVERAGE")
        used = {r.sample_id: r for r in train + calibration + latency_cal + failure_rows + cost_rows}
        measured = min(r.observed_at for r in used.values())
        if any(not r.certification_eligible for r in used.values()): reasons.append("BENCHMARK_ONLY_EVIDENCE")
        trust_expirations = [r.trusted_until for r in used.values() if r.trusted_until is not None]
        lineage = EvidenceLineage(
            target_id=target_id, traffic_stratum=stratum(request), feature_version=request.feature_version,
            sample_size=len(train), calibration_size=n,
            training_hash=self._lineage_hash(train), calibration_hash=self._lineage_hash(calibration + latency_cal + failure_rows),
            artifact_hash=digest({"method": "local-knn-hoeffding-binomial-p95-v2", "rows": {key:self._row_hash(row) for key,row in used.items()},
                                  "request_features":request.query_features,"request_stratum":stratum(request),
                                  "training_order":[r.sample_id for r in train],
                                  "calibration_order":[r.sample_id for r in calibration],
                                  "k": self.k, "alpha": alpha, "latency_alpha": latency_risk / max(1, comparisons)}),
            endpoint_revisions=revisions, measured_at=measured,
            age_seconds=(at-measured).total_seconds(), calibrated=not reasons, reasons=tuple(reasons),
            trusted_until=min(trust_expirations) if trust_expirations else None)
        return CapabilityEstimate(target_id, quality, cost, latency, failure, lineage)
