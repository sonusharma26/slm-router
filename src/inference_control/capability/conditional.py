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


QualityMethod = Literal["local-mean-hoeffding", "local-mean-kl"]
QualityPredictor = Literal["local", "cohort-mean"]


def binary_kl(observed: float, candidate: float) -> float:
    if observed == candidate:
        return 0.
    if candidate in (0., 1.):
        return math.inf
    value = 0.
    if observed > 0:
        value += observed * (math.log(observed) - math.log(candidate))
    if observed < 1:
        value += (1-observed) * (math.log1p(-observed) - math.log1p(-candidate))
    return max(0., value)


def quality_interval(score_mean: float, n: int, alpha: float,
                     method: QualityMethod = "local-mean-kl") -> tuple[float, float]:
    if not 0 <= score_mean <= 1 or n < 0 or not 0 < alpha < 1:
        raise ValueError("bounded mean, nonnegative sample count and risk in (0,1) required")
    if method not in {"local-mean-hoeffding", "local-mean-kl"}:
        raise ValueError("unknown quality bound method")
    if n == 0:
        return 0., 1.
    if method == "local-mean-hoeffding":
        radius = math.sqrt(math.log(2/alpha)/(2*n))
        return max(0., score_mean-radius), min(1., score_mean+radius)
    threshold = (math.log(2.) - math.log(alpha)) / n
    # Spend less risk to keep floating-point inversion outside the exact confidence set.
    threshold += max(1e-12, 32*math.ulp(threshold))
    def lower_root(observed):
        if observed == 0:
            return 0.
        if observed == 1:
            return math.nextafter(math.exp(-threshold), 0.)
        low, high = 0., observed
        for _ in range(64):
            middle = (low+high)/2
            if middle == low or middle == high:
                break
            if binary_kl(observed, middle) > threshold:
                low = middle
            else:
                high = middle
        return math.nextafter(low, 0.)
    return lower_root(score_mean), min(1., math.nextafter(1-lower_root(1-score_mean), 1.))


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


def stratum(request: RequestContext, *, include_lengths: bool = True) -> str:
    """Exact semantic/governance fields and coarse length buckets; no output leakage."""
    fields = {
        "application": request.application_id, "task": request.task_hint,
        "slices": request.traffic_slices, "tags": request.tags,
        "modality": request.modality, "capabilities": request.required_capabilities,
        "feature_version": request.feature_version, "privacy": request.privacy_classification,
        "json_schema": request.structured_output_schema_hash, "tools": request.tool_schema_hash,
    }
    if include_lengths:
        fields.update(input_bucket=int(math.log2(max(1, request.input_tokens))),
                      output_bucket=int(math.log2(max(1, request.max_output_tokens))))
    return digest(fields)


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


class QualityBoundDiagnostics(FrozenContract):
    """Population confidence endpoints and the separate raw point forecast."""
    method: QualityMethod = "local-mean-hoeffding"
    predictor: QualityPredictor = "local"
    raw_prediction: float | None = None
    prediction_sample_size: int = 0
    prediction_projection: float = 0.
    lower_tail_risk: float | None = None
    upper_tail_risk: float | None = None
    training_mean: float | None
    calibration_mean: float | None
    hoeffding_radius: float | None
    calibration_lower: float
    training_mean_clamp: float
    comparisons: int
    effective_risk: float


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
    quality_diagnostics: QualityBoundDiagnostics | None = None
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
    def __init__(self, *, k: int = 512, min_samples: int = 30, max_age_seconds: int = 86400,
                 quality_method: QualityMethod = "local-mean-kl",
                 quality_predictors: dict[str, QualityPredictor] | None = None):
        if k < 1 or min_samples < 1 or max_age_seconds < 1:
            raise ValueError("positive estimator bounds required")
        self.k, self.min_samples, self.max_age_seconds = k, min_samples, max_age_seconds
        if quality_method not in {"local-mean-kl", "local-mean-hoeffding"}:
            raise ValueError("unknown quality bound method")
        if any(mode not in {"local", "cohort-mean"} for mode in (quality_predictors or {}).values()):
            raise ValueError("unknown quality point predictor")
        self._quality_method = quality_method
        self._quality_predictors = tuple(sorted((quality_predictors or {}).items()))
        self.rows: dict[str, EvidenceObservation] = {}
        self.invalidations: list[dict] = []
        self._version: str | None = None
        self._index: dict[tuple, list[EvidenceObservation]] | None = None
        self._fingerprints: dict[str, tuple[EvidenceObservation, str]] = {}

    @property
    def quality_method(self) -> QualityMethod:
        return self._quality_method

    @property
    def quality_predictors(self) -> tuple[tuple[str, QualityPredictor], ...]:
        return self._quality_predictors

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = "map:" + digest({"format":"content-addressed-capability-v3", "k":self.k,
                "min_samples":self.min_samples,"max_age_seconds":self.max_age_seconds,
                "quality_method":self.quality_method,"quality_predictors":self.quality_predictors,
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

    def reconcile_outcome_time(self, sample_id: str, observed_at: datetime | None) -> bool:
        row = self.rows.get(sample_id)
        if row is None or row.outcome_id is None or sample_id != f"outcome:{row.outcome_id}":
            return False
        if observed_at is not None and observed_at.tzinfo is None:
            raise ValueError("observation timestamp must have a timezone")
        updates = {"observed_at":observed_at} if observed_at is not None else {"disputed":True,"trusted":False}
        corrected = row.model_copy(update=updates)
        if corrected == row:
            return False
        self.rows[sample_id] = corrected
        self._fingerprints.pop(sample_id, None)
        self._version = self._index = None
        return True

    def invalidate(self, endpoint_id: str, metrics: tuple[str, ...], *,
                   at: datetime, slice_id: str | None = None, reason: str = "drift") -> None:
        self.invalidations.append({"endpoint_id": endpoint_id, "metrics": list(metrics),
                                   "at": at.isoformat(), "slice_id": slice_id, "reason": reason})
        self._version = None

    def export(self) -> dict:
        return {"k": self.k, "min_samples": self.min_samples, "max_age_seconds": self.max_age_seconds,
                "quality_method":self.quality_method,"quality_predictors":dict(self.quality_predictors),
                "rows": [self.rows[k].model_dump(mode="json") for k in sorted(self.rows)],
                "invalidations": self.invalidations}

    @classmethod
    def restore(cls, data: dict) -> ConditionalCapabilityMap:
        result = cls(k=data["k"], min_samples=data["min_samples"], max_age_seconds=data["max_age_seconds"],
                     quality_method=data.get("quality_method", "local-mean-hoeffding"),
                     quality_predictors=data.get("quality_predictors", {}))
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
            if item["slice_id"] and item["slice_id"] not in (row.request.traffic_slices or {"default"}):
                continue
            if row.observed_at <= datetime.fromisoformat(item["at"]):
                return False
        return True

    def _neighbors(self, request, target_id, revisions, at, max_age, *, cohort=False):
        if self._index is None:
            self._index = {}
            for row in self.rows.values():
                for include_lengths in (True, False):
                    key = (include_lengths, row.target_id, stratum(row.request, include_lengths=include_lengths), row.endpoint_revisions)
                    self._index.setdefault(key, []).append(row)
        key = (not cohort, target_id, stratum(request, include_lengths=not cohort), tuple(revisions))
        candidates = self._index.get(key, [])
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
        predictor = dict(self.quality_predictors).get(target_id, "local")
        prediction_rows = train
        if predictor == "cohort-mean":
            seen = set()
            prediction_rows = []
            for row in self._neighbors(request, target_id, revisions, at,
                                       min(self.max_age_seconds, max_age_seconds or self.max_age_seconds), cohort=True):
                if row.split == "train" and self._valid(row, "quality", at) and row.request.request_id not in seen:
                    seen.add(row.request.request_id)
                    prediction_rows.append(row)
        center = mean(r.quality for r in (prediction_rows or train or calibration))
        if n:
            local_mean = mean(r.quality for r in calibration)
            lower, upper = quality_interval(local_mean, n, alpha, self.quality_method)
        else:
            lower, upper = 0., 1.
        if self.quality_method == "local-mean-hoeffding":
            quality = Prediction(mean=center, lower=min(lower, center), upper=max(upper, center))
        else:
            quality = Prediction(mean=min(upper, max(lower, center)), lower=lower, upper=upper)
        quality_diagnostics = QualityBoundDiagnostics(
            method=self.quality_method, predictor=predictor, raw_prediction=center if prediction_rows or train else None,
            prediction_sample_size=len(prediction_rows or train), prediction_projection=quality.mean-center,
            lower_tail_risk=alpha/2, upper_tail_risk=alpha/2,
            training_mean=mean(r.quality for r in train) if train else None,
            calibration_mean=local_mean if n else None,
            hoeffding_radius=math.sqrt(math.log(2/alpha)/(2*n)) if n and self.quality_method == "local-mean-hoeffding" else None,
            calibration_lower=lower, training_mean_clamp=lower-quality.lower,
            comparisons=max(1, comparisons), effective_risk=alpha)
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
        used = {r.sample_id: r for r in train + prediction_rows + calibration + latency_cal + failure_rows + cost_rows}
        measured = min(r.observed_at for r in used.values())
        if any(not r.certification_eligible for r in used.values()): reasons.append("BENCHMARK_ONLY_EVIDENCE")
        trust_expirations = [r.trusted_until for r in used.values() if r.trusted_until is not None]
        lineage = EvidenceLineage(
            target_id=target_id, traffic_stratum=stratum(request), feature_version=request.feature_version,
            sample_size=len(train), calibration_size=n,
            training_hash=self._lineage_hash(prediction_rows or train), calibration_hash=self._lineage_hash(calibration + latency_cal + failure_rows),
            artifact_hash=digest({"method": self.quality_method, "predictor":predictor,
                                  "prediction_order":[r.sample_id for r in prediction_rows],
                                  "rows": {key:self._row_hash(row) for key,row in used.items()},
                                  "request_features":request.query_features,"request_stratum":stratum(request),
                                  "training_order":[r.sample_id for r in train],
                                  "calibration_order":[r.sample_id for r in calibration],
                                  "k": self.k, "alpha": alpha, "latency_alpha": latency_risk / max(1, comparisons)}),
            endpoint_revisions=revisions, measured_at=measured,
            age_seconds=(at-measured).total_seconds(), calibrated=not reasons, reasons=tuple(reasons),
            quality_diagnostics=quality_diagnostics,
            trusted_until=min(trust_expirations) if trust_expirations else None)
        return CapabilityEstimate(target_id, quality, cost, latency, failure, lineage)
