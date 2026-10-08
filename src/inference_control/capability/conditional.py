"""Request-conditioned, auditable kNN with separate calibration evidence.

Bounds concern the local traffic population under the recorded assumptions, not an
individual prompt. No test label, oracle answer or post-response feature enters routing.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import math
from statistics import mean
from functools import lru_cache
from typing import Literal
from pydantic import Field, model_validator
from inference_control.contracts import FrozenContract, RequestContext, Prediction, EndpointSnapshot
from inference_control.util import JsonNative, digest, digest_native
from inference_control.routing.pooled import (POOLED_ASSUMPTIONS, POOLED_METHOD, EndpointEvidence,
                                              PooledConfig, PooledEvidence)


QualityMethod = Literal["local-mean-hoeffding", "local-mean-kl", "pooled-ltt"]
QualityPredictor = Literal["local", "cohort-mean", "learned"]


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


@lru_cache(maxsize=8192)
def _feature_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum((x-y)**2 for x,y in zip(left, right))


def _json_native(value) -> bool:
    kind = type(value)
    if kind is float:
        return math.isfinite(value)
    if kind in (str, int, bool, type(None)):
        return True
    if kind is dict:
        return all(type(k) is str and _json_native(v) for k, v in value.items())
    if kind is list:
        return all(_json_native(v) for v in value)
    return False


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
                 quality_method: QualityMethod = POOLED_METHOD,
                 quality_predictors: dict[str, QualityPredictor] | None = None,
                 pooled: PooledConfig | dict | None = None):
        if k < 1 or min_samples < 1 or max_age_seconds < 1:
            raise ValueError("positive estimator bounds required")
        self.k, self.min_samples, self.max_age_seconds = k, min_samples, max_age_seconds
        if quality_method not in {"local-mean-kl", "local-mean-hoeffding", POOLED_METHOD}:
            raise ValueError("unknown quality bound method")
        if any(mode not in {"local", "cohort-mean"} for mode in (quality_predictors or {}).values()):
            raise ValueError("unknown quality point predictor")
        self._quality_method = quality_method
        self._quality_predictors = tuple(sorted((quality_predictors or {}).items()))
        self._pooled_config = pooled if isinstance(pooled, PooledConfig) else PooledConfig.from_identity(pooled)
        self._pooled_cache: dict[tuple, PooledEvidence] = {}
        self._base_version: str | None = None
        self.rows: dict[str, EvidenceObservation] = {}
        self.invalidations: list[dict] = []
        self._version: str | None = None
        self._index: dict[tuple, list[EvidenceObservation]] | None = None
        self._fingerprints: dict[str, tuple[EvidenceObservation, str]] = {}
        # Rows are immutable; their strata are pure functions of the request.
        self._strata: dict[str, tuple[EvidenceObservation, str, str]] = {}
        self._row_dumps: dict[str, tuple[EvidenceObservation, JsonNative]] = {}
        self._request_strata: dict[str, tuple[RequestContext, dict[bool, str]]] = {}

    @property
    def quality_method(self) -> QualityMethod:
        return self._quality_method

    @property
    def quality_predictors(self) -> tuple[tuple[str, QualityPredictor], ...]:
        return self._quality_predictors

    @property
    def pooled_config(self) -> PooledConfig:
        return self._pooled_config

    def configure_pooled(self, config: PooledConfig) -> None:
        """Set learned-routing hyperparameters and (optionally) an explicit threshold grid."""
        if not isinstance(config, PooledConfig):
            raise TypeError("PooledConfig required")
        self._pooled_config = config
        self._version = None

    @property
    def version(self) -> str:
        if self._version is None:
            identity = {"format":"content-addressed-capability-v3", "k":self.k,
                "min_samples":self.min_samples,"max_age_seconds":self.max_age_seconds,
                "quality_method":self.quality_method,"quality_predictors":self.quality_predictors,
                "rows":{key:self._row_hash(self.rows[key]) for key in sorted(self.rows)},
                "invalidations":self.invalidations}
            if self.quality_method == POOLED_METHOD:
                identity["pooled"] = self._pooled_config.identity()
            self._version = "map:" + digest(identity)
        return self._version

    @property
    def evidence_version(self) -> str:
        """Rows and invalidations only; unchanged by estimator configuration."""
        if self._base_version is None:
            self._base_version = digest({"rows":{key:self._row_hash(self.rows[key]) for key in sorted(self.rows)},
                                         "invalidations":self.invalidations})
        return self._base_version

    def _row_hash(self, row: EvidenceObservation) -> str:
        cached=self._fingerprints.get(row.sample_id)
        if cached is None or cached[0] is not row:
            cached=(row,digest(row))
            self._fingerprints[row.sample_id]=cached
        return cached[1]

    def _request_stratum(self, request: RequestContext, include_lengths: bool = True) -> str:
        """``stratum`` memoized per immutable request object (estimates repeat it per plan)."""
        cached = self._request_strata.get(request.request_id)
        if cached is None or cached[0] is not request:
            if len(self._request_strata) >= 4096:
                self._request_strata.clear()
            cached = (request, {})
            self._request_strata[request.request_id] = cached
        values = cached[1]
        if include_lengths not in values:
            values[include_lengths] = stratum(request, include_lengths=include_lengths)
        return values[include_lengths]

    def _row_json(self, row: EvidenceObservation) -> JsonNative:
        """Cached JSON dump of an immutable row; snapshots reuse it instead of re-dumping."""
        cached = self._row_dumps.get(row.sample_id)
        if cached is None or cached[0] is not row:
            dumped = row.model_dump(mode="json")
            if not _json_native(dumped):
                raise ValueError("non-finite evidence cannot be serialized")
            cached = (row, JsonNative(dumped))
            self._row_dumps[row.sample_id] = cached
        return cached[1]

    def _lineage_hash(self, rows) -> str:
        return digest_native([[r.sample_id,self._row_hash(r)] for r in rows])

    def add(self, row: EvidenceObservation) -> None:
        # A prompt cannot migrate between training and calibration, even across endpoints.
        for prior in self.rows.values():
            if prior.request.request_id == row.request.request_id and prior.split != row.split:
                raise ValueError("request appears in training and calibration")
        prior = self.rows.get(row.sample_id)
        if prior and prior != row:
            raise ValueError("immutable sample_id conflict; use supersede_outcome")
        self.rows[row.sample_id] = row
        self._version = self._index = self._base_version = None

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
        self._version = self._index = self._base_version = None

    def supersede_outcome(self, outcome_id: str) -> None:
        for key, row in tuple(self.rows.items()):
            if row.outcome_id == outcome_id:
                self.rows[key] = row.model_copy(update={"disputed": True, "trusted": False})
        self._version = self._index = self._base_version = None

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
        self._version = self._index = self._base_version = None
        return True

    def invalidate(self, endpoint_id: str, metrics: tuple[str, ...], *,
                   at: datetime, slice_id: str | None = None, reason: str = "drift") -> None:
        self.invalidations.append({"endpoint_id": endpoint_id, "metrics": list(metrics),
                                   "at": at.isoformat(), "slice_id": slice_id, "reason": reason})
        self._version = self._base_version = None

    def export(self) -> dict:
        result = {"k": self.k, "min_samples": self.min_samples, "max_age_seconds": self.max_age_seconds,
                "quality_method":self.quality_method,"quality_predictors":dict(self.quality_predictors),
                "rows": [self._row_json(self.rows[k]) for k in sorted(self.rows)],
                "invalidations": self.invalidations}
        if self.quality_method == POOLED_METHOD:
            result["pooled"] = self._pooled_config.identity()
        return result

    @classmethod
    def restore(cls, data: dict) -> ConditionalCapabilityMap:
        result = cls(k=data["k"], min_samples=data["min_samples"], max_age_seconds=data["max_age_seconds"],
                     quality_method=data.get("quality_method", "local-mean-hoeffding"),
                     quality_predictors=data.get("quality_predictors", {}),
                     pooled=data.get("pooled"))
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
                cached = self._strata.get(row.sample_id)
                if cached is None or cached[0] is not row:
                    cached = (row, stratum(row.request, include_lengths=True),
                              stratum(row.request, include_lengths=False))
                    self._strata[row.sample_id] = cached
                for include_lengths, key_stratum in ((True, cached[1]), (False, cached[2])):
                    key = (include_lengths, row.target_id, key_stratum, row.endpoint_revisions)
                    self._index.setdefault(key, []).append(row)
        key = (not cohort, target_id, self._request_stratum(request, not cohort), tuple(revisions))
        candidates = self._index.get(key, [])
        candidates = [r for r in candidates if
                      0 <= (at - r.observed_at).total_seconds() <= max_age
                      and len(r.request.query_features) == len(request.query_features)]
        def distance(row):
            return (_feature_distance(row.request.query_features, request.query_features),
                    row.request.request_id, row.sample_id)
        return sorted(candidates, key=distance)

    # ----- pooled learned routing ---------------------------------------------------------
    def pooled_evidence(self, endpoints, feature_version: str, at: datetime, max_age: float) -> PooledEvidence:
        """Train/calibration matrices and a fitted predictor for direct plans over ``endpoints``.

        Rows must be direct single-endpoint evidence for the endpoint's current capability
        revision and the request's feature version, valid (age, trust, disputes and
        invalidations) at ``at``. Cached until evidence or configuration changes or a row
        enters or leaves its validity window.
        """
        revisions = self._revision_signature(endpoints)
        key = (self.evidence_version, digest(self._pooled_config.predictor.identity()), revisions,
               feature_version, float(max_age))
        cached = self._pooled_cache.get(key)
        if cached is not None and cached.usable(at):
            return cached
        previous = next((v for k, v in self._pooled_cache.items() if k[2:] == key[2:]), None)
        evidence = self._build_pooled_evidence(dict(revisions), feature_version, at, max_age, previous)
        self._pooled_cache = {k: v for k, v in self._pooled_cache.items() if k[2:] != key[2:]}
        self._pooled_cache[key] = evidence
        return evidence

    def _revision_signature(self, endpoints) -> tuple:
        """(endpoint_id, capability_revision) pairs, memoized per snapshot object (revisions are digests)."""
        cache = self.__dict__.setdefault("_signature_cache", {})
        key = tuple(id(e) for e in endpoints)
        cached = cache.get(key)
        if cached is None or any(a is not b for a, b in zip(cached[0], endpoints)):
            if len(cache) >= 256:
                cache.clear()
            cached = (tuple(endpoints), tuple(sorted((e.endpoint_id, e.capability_revision) for e in endpoints)))
            cache[key] = cached
        return cached[1]

    def _build_pooled_evidence(self, revisions: dict, feature_version: str, at: datetime,
                               max_age: float, previous: PooledEvidence | None) -> PooledEvidence:
        import numpy as np
        from inference_control.routing.predictor import QualityPredictor
        ids = tuple(sorted(revisions))
        column = {e: i for i, e in enumerate(ids)}
        splits = {"train": {}, "calibration": {}}
        stats = {e: {"train": 0, "lat_cal": [], "lat_all": [], "fail": 0, "fail_n": 0, "out": [], "eligible": True}
                 for e in ids}
        changes, used = [], {"train": [], "calibration": []}
        for row in self.rows.values():
            if row.plan_type != "direct" or len(row.endpoint_ids) != 1:
                continue
            endpoint = row.endpoint_ids[0]
            if endpoint not in revisions or row.endpoint_revisions != (revisions[endpoint],):
                continue
            if row.request.feature_version != feature_version:
                continue
            age = (at - row.observed_at).total_seconds()
            if age < 0:
                changes.append(row.observed_at)
                continue
            if age > max_age:
                continue
            changes.append(row.observed_at + timedelta(seconds=max_age, microseconds=1))
            if row.trusted_until is not None and row.trusted_until > at:
                changes.append(row.trusted_until)
            info, bucket = stats[endpoint], splits[row.split]
            entry = bucket.setdefault(row.request.request_id, [row.request, {}, {}])
            touched = False
            for metric, slot in (("quality", 1), ("cost", 2)):
                if self._valid(row, metric, at):
                    prior = entry[slot].get(endpoint)
                    order = (row.observed_at, row.sample_id)
                    if prior is None or order > prior[0]:
                        entry[slot][endpoint] = (order, row)
                    touched = True
            if touched:
                used[row.split].append(row)
            if self._valid(row, "latency", at):
                info["lat_all"].append(row.latency_ms)
                if row.split == "calibration":
                    info["lat_cal"].append(row.latency_ms)
            if row.split == "calibration" and self._valid(row, "failure", at):
                info["fail_n"] += 1
                info["fail"] += int(row.failed)
            if self._valid(row, "cost", at):
                info["out"].append(row.output_tokens)
        def matrices(split):
            entries = [splits[split][rid] for rid in sorted(splits[split])]
            entries = [e for e in entries if e[1]]  # at least one valid quality outcome
            if not entries:
                return [], np.zeros((0, len(ids))), np.zeros((0, len(ids)))
            dims = {len(e[0].query_features) for e in entries}
            width = max(dims, key=lambda d: (sum(len(e[0].query_features) == d for e in entries), d))
            entries = [e for e in entries if len(e[0].query_features) == width]
            quality = np.full((len(entries), len(ids)), np.nan)
            cost = np.full((len(entries), len(ids)), np.nan)
            for i, (request, q_rows, c_rows) in enumerate(entries):
                for endpoint, (_, row) in q_rows.items():
                    quality[i, column[endpoint]] = row.quality
                for endpoint, (_, row) in c_rows.items():
                    cost[i, column[endpoint]] = row.cost
            return [e[0] for e in entries], quality, cost
        train_requests, train_quality, train_cost = matrices("train")
        calibration_requests, calibration_quality, _ = matrices("calibration")
        for endpoint in ids:
            stats[endpoint]["train"] = int((~np.isnan(train_quality[:, column[endpoint]])).sum()) if len(train_requests) else 0
        for row in used["train"] + used["calibration"]:
            if not row.certification_eligible and self._valid(row, "quality", at):
                stats[row.endpoint_ids[0]]["eligible"] = False
        endpoint_evidence = {e: EndpointEvidence(
            training_outcomes=v["train"], latency_calibration=tuple(sorted(v["lat_cal"])),
            latency_mean=mean(v["lat_all"]) if v["lat_all"] else None,
            latency_max=max(v["lat_all"]) if v["lat_all"] else None,
            failures=v["fail"], failure_observations=v["fail_n"],
            output_tokens=mean(v["out"]) if v["out"] else None,
            certification_eligible=v["eligible"]) for e, v in stats.items()}
        predictor = None
        config = self._pooled_config.predictor
        if len(train_requests) and not np.isnan(train_quality).all():
            features = [r.query_features for r in train_requests]
            slices = [r.task_hint for r in train_requests]
            reuse = None
            if previous is not None and previous.predictor is not None:
                old = previous.predictor
                if ((old.config.clusters, old.config.iterations, old.config.seed) ==
                        (config.clusters, config.iterations, config.seed)
                        and old._centroids.shape[1] == len(features[0])
                        and old.training_size >= .8 * len(train_requests)):
                    reuse = old._centroids  # still a function of training evidence only
            predictor = QualityPredictor(config).fit(features, train_quality, train_cost, slices, centroids=reuse)
        rows_used = used["train"] + used["calibration"]
        measured = min((r.observed_at for r in rows_used), default=None)
        trust = [r.trusted_until for r in rows_used if r.trusted_until is not None]
        training_hash = self._lineage_hash(sorted(used["train"], key=lambda r: r.sample_id))
        calibration_hash = self._lineage_hash(sorted(used["calibration"], key=lambda r: r.sample_id))
        future = [t for t in changes if t > at]
        identity = digest_native({"training": training_hash, "calibration": calibration_hash,
            "predictor": predictor.artifact_digest() if predictor is not None else None,
            "endpoints": [[e, revisions[e]] for e in ids], "feature_version": feature_version,
            "max_age": float(max_age)})
        return PooledEvidence(endpoint_ids=ids, feature_version=feature_version, fitted_at=at,
            next_change=min(future) if future else None, train_requests=train_requests,
            train_quality=train_quality, train_cost=train_cost, calibration_requests=calibration_requests,
            calibration_quality=calibration_quality, endpoints=endpoint_evidence, measured_at=measured,
            trusted_until=min(trust) if trust else None, training_hash=training_hash,
            calibration_hash=calibration_hash, predictor=predictor, identity=identity)

    @staticmethod
    def pooled_numbers(evidence: PooledEvidence, request: RequestContext, endpoint: EndpointSnapshot,
                       cost_forecast: float, *, quality_risk: float, latency_risk: float,
                       comparisons: int, minimum: int) -> dict:
        """Cost, latency and failure predictions shared by serving and calibration."""
        info = evidence.endpoints[endpoint.endpoint_id]
        absolute = ((request.input_tokens + endpoint.input_token_overhead) * endpoint.input_price_per_million
                    + request.max_output_tokens * endpoint.output_price_per_million) / 1e6
        if cost_forecast == cost_forecast:  # finite learned forecast
            expected = float(cost_forecast)
        else:
            output = request.expected_output_tokens
            if output is None:
                output = info.output_tokens if info.output_tokens is not None else request.max_output_tokens
            expected = ((request.input_tokens + endpoint.input_token_overhead) * endpoint.input_price_per_million
                        + min(request.max_output_tokens, output) * endpoint.output_price_per_million) / 1e6
        comparisons = max(1, comparisons)
        values = info.latency_calibration
        rank = quantile_tolerance_rank(len(values), .95, latency_risk / comparisons) if values else None
        lat_mean = info.latency_mean if info.latency_mean is not None else 0.
        lat_upper = values[rank-1] if rank is not None else (info.latency_max if info.latency_max is not None else 5e299) * 2
        nf = info.failure_observations
        pf = info.failures / nf if nf else .5
        radius = math.sqrt(math.log(2/(quality_risk / comparisons))/(2*nf)) if nf else 1.
        reasons = []
        if info.latency_mean is None: reasons.append("LATENCY_MISSING")
        if rank is None: reasons.append("LATENCY_TAIL_COVERAGE")
        if info.training_outcomes < minimum: reasons.append("TRAINING_COVERAGE")
        if nf < minimum: reasons.append("FAILURE_COVERAGE")
        if not info.certification_eligible: reasons.append("BENCHMARK_ONLY_EVIDENCE")
        return {"cost": (expected, min(expected, 0.), max(absolute, expected)),
                "latency": (lat_mean, 0., max(lat_mean, lat_upper)),
                "failure": (pf, max(0., pf - radius), min(1., pf + radius)),
                "reasons": tuple(reasons)}

    def pooled_estimate(self, evidence: PooledEvidence, request: RequestContext, endpoint: EndpointSnapshot,
                        quality_forecast: float, numbers: dict, *, at: datetime, quality_risk: float,
                        target: float | None = None, calibration=None, artifact_base: str | None = None):
        """Assemble a CapabilityEstimate; ``quality.lower`` carries the certified population bound."""
        from inference_control.planning.planner import CapabilityEstimate
        if quality_forecast != quality_forecast:  # NaN: no learned forecast
            return None
        q = float(min(1., max(0., quality_forecast)))
        certified = (calibration is not None and calibration.threshold is not None and target is not None
                     and q >= calibration.threshold)
        if certified:
            quality = Prediction(mean=min(1., max(target, q)), lower=target, upper=1.)
        else:
            quality = Prediction(mean=q, lower=0., upper=1.)
        reasons = list(numbers["reasons"])
        if not certified:
            reasons.insert(0, "POOLED_NOT_CERTIFIED" if calibration is not None and calibration.threshold is not None
                           else "POOLED_UNCALIBRATED")
        info = evidence.endpoints[endpoint.endpoint_id]
        diagnostics = QualityBoundDiagnostics(method=POOLED_METHOD, predictor="learned", raw_prediction=q,
            prediction_sample_size=info.training_outcomes, prediction_projection=quality.mean - q,
            lower_tail_risk=quality_risk, upper_tail_risk=0., training_mean=evidence.training_mean(endpoint.endpoint_id),
            calibration_mean=calibration.certified_mean if calibration is not None else None,
            hoeffding_radius=None, calibration_lower=quality.lower, training_mean_clamp=0.,
            comparisons=1, effective_risk=quality_risk)
        base = artifact_base or digest_native([evidence.identity, None])
        measured = evidence.measured_at or at
        lineage = EvidenceLineage(target_id=endpoint.endpoint_id, traffic_stratum=self._request_stratum(request),
            feature_version=request.feature_version, sample_size=info.training_outcomes,
            calibration_size=calibration.calibration_requests if calibration is not None else 0,
            training_hash=evidence.training_hash, calibration_hash=evidence.calibration_hash,
            artifact_hash=digest_native([base, endpoint.endpoint_id]), endpoint_revisions=(endpoint.capability_revision,),
            measured_at=measured, age_seconds=(at - measured).total_seconds(), calibrated=not reasons,
            reasons=tuple(reasons), quality_diagnostics=diagnostics, trusted_until=evidence.trusted_until,
            assumptions=POOLED_ASSUMPTIONS)
        return CapabilityEstimate(endpoint.endpoint_id, quality,
                                  Prediction(mean=numbers["cost"][0], lower=numbers["cost"][1], upper=numbers["cost"][2]),
                                  Prediction(mean=numbers["latency"][0], lower=numbers["latency"][1], upper=numbers["latency"][2]),
                                  Prediction(mean=numbers["failure"][0], lower=numbers["failure"][1], upper=numbers["failure"][2]),
                                  lineage)

    def estimate(self, request: RequestContext, target_id: str,
                 endpoints: tuple[EndpointSnapshot, ...], *, at: datetime | None = None,
                 quality_risk: float = 0.05, latency_risk: float = 0.05,
                 min_samples: int | None = None, max_age_seconds: int | None = None,
                 comparisons: int = 1):
        from inference_control.planning.planner import CapabilityEstimate
        at = at or datetime.now(timezone.utc)
        if self.quality_method == POOLED_METHOD:
            # Uncertified forecast: certification is a property of the planner's routing rule.
            if len(endpoints) != 1 or endpoints[0].endpoint_id != target_id:
                return None
            evidence = self.pooled_evidence(endpoints, request.feature_version, at,
                                            min(self.max_age_seconds, max_age_seconds or self.max_age_seconds))
            quality, cost = evidence.forecast([request])
            numbers = self.pooled_numbers(evidence, request, endpoints[0], float(cost[0, 0]),
                quality_risk=quality_risk, latency_risk=latency_risk, comparisons=comparisons,
                minimum=max(self.min_samples, min_samples or self.min_samples))
            return self.pooled_estimate(evidence, request, endpoints[0], float(quality[0, 0]), numbers,
                                        at=at, quality_risk=quality_risk)
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
            target_id=target_id, traffic_stratum=self._request_stratum(request), feature_version=request.feature_version,
            sample_size=len(train), calibration_size=n,
            training_hash=self._lineage_hash(prediction_rows or train), calibration_hash=self._lineage_hash(calibration + latency_cal + failure_rows),
            artifact_hash=digest({"method": self.quality_method, "predictor":predictor,
                                  "prediction_order":[r.sample_id for r in prediction_rows],
                                  "rows": {key:self._row_hash(row) for key,row in used.items()},
                                  "request_features":request.query_features,"request_stratum":self._request_stratum(request),
                                  "training_order":[r.sample_id for r in train],
                                  "calibration_order":[r.sample_id for r in calibration],
                                  "k": self.k, "alpha": alpha, "latency_alpha": latency_risk / max(1, comparisons)}),
            endpoint_revisions=revisions, measured_at=measured,
            age_seconds=(at-measured).total_seconds(), calibrated=not reasons, reasons=tuple(reasons),
            quality_diagnostics=quality_diagnostics,
            trusted_until=min(trust_expirations) if trust_expirations else None)
        return CapabilityEstimate(target_id, quality, cost, latency, failure, lineage)
