"""Learned per-endpoint quality and cost forecasts from training evidence only.

For request x and endpoint m the forecast is hierarchical:

* base: the endpoint's mean on x's slice (when that slice was seen in training),
  shrunk toward the endpoint's global mean;
* cluster: a soft average over the nearest spherical k-means clusters of
  ``(sum_c + cluster_prior * base) / (count_c + cluster_prior)``;
* final: ``(sum_neighbors w * Q + neighbor_prior * cluster) / (sum w + neighbor_prior)``,
  where neighbors are the most similar training requests (cosine similarity) and
  ``w = exp(sharpness * (s - s_max))``.

Missing outcomes (NaN) are ignored per endpoint, so sparse evidence is supported.
Forecasts are point predictions; statistical guarantees come from separate
calibration (``inference_control.routing.calibration``), never from this module.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class PredictorConfig:
    neighbors: int = 24
    neighbor_prior: float = 4.0
    sharpness: float = 20.0
    clusters: int = 16
    cluster_prior: float = 8.0
    cluster_top: int = 3
    slice_prior: float = 16.0
    slice_neighbors: bool = True
    iterations: int = 30
    seed: int = 0

    def __post_init__(self):
        if self.neighbors < 0 or self.clusters < 1 or self.cluster_top < 1 or self.iterations < 1:
            raise ValueError("predictor sizes must be positive (neighbors may be zero)")
        if min(self.neighbor_prior, self.cluster_prior, self.slice_prior) <= 0 or self.sharpness <= 0:
            raise ValueError("priors and sharpness must be positive")

    def identity(self) -> dict:
        return asdict(self)

    @classmethod
    def from_identity(cls, value: dict | None) -> PredictorConfig:
        return cls(**(value or {}))


def normalize(rows) -> np.ndarray:
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.
    return matrix / norms


def spherical_kmeans(features: np.ndarray, clusters: int, iterations: int, seed: int) -> np.ndarray:
    """Deterministic k-means++ initialization and cosine k-means on unit vectors."""
    n = len(features)
    count = max(1, min(clusters, n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    centers = [features[first]]
    closest = np.maximum(1. - features @ features[first], 0.)
    for _ in range(1, count):
        total = float(closest.sum())
        index = int(rng.integers(n)) if total <= 1e-12 else int(rng.choice(n, p=closest / total))
        centers.append(features[index])
        closest = np.minimum(closest, np.maximum(1. - features @ features[index], 0.))
    centroids = np.array(centers)
    for _ in range(iterations):
        assign = np.argmax(features @ centroids.T, axis=1)
        updated = np.zeros_like(centroids)
        np.add.at(updated, assign, features)
        norms = np.linalg.norm(updated, axis=1)
        empty = norms == 0
        updated[~empty] /= norms[~empty, None]
        updated[empty] = centroids[empty]
        if np.allclose(updated, centroids, atol=1e-10):
            centroids = updated
            break
        centroids = updated
    return centroids


def _array_digest(*arrays) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(np.nan_to_num(np.asarray(array, dtype=float), nan=-7.0))
        hasher.update(str(value.shape).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


class QualityPredictor:
    """Fit on a (requests x endpoints) training matrix; predict quality and cost matrices."""

    def __init__(self, config: PredictorConfig | None = None):
        self.config = config or PredictorConfig()
        self._fitted = False

    # ----- fitting -------------------------------------------------------------------------
    def fit(self, features, quality, cost=None, slices=None, *, centroids=None) -> QualityPredictor:
        X = normalize(features)
        Q = np.asarray(quality, dtype=float)
        if Q.ndim != 2 or len(Q) != len(X) or len(X) == 0:
            raise ValueError("quality must be a nonempty (requests x endpoints) matrix")
        finite = Q[~np.isnan(Q)]
        if finite.size and (finite.min() < 0 or finite.max() > 1):
            raise ValueError("quality must be normalized to [0,1]")
        C = np.full(Q.shape, np.nan) if cost is None else np.asarray(cost, dtype=float)
        if C.shape != Q.shape:
            raise ValueError("cost must match the quality matrix")
        self._X, self._Q, self._C = X, Q, C
        self._qmask, self._cmask = ~np.isnan(Q), ~np.isnan(C)
        self._Q0, self._C0 = np.where(self._qmask, Q, 0.), np.where(self._cmask, C, 0.)
        qcount, ccount = self._qmask.sum(0), self._cmask.sum(0)
        self._global_q = np.where(qcount > 0, self._Q0.sum(0) / np.maximum(qcount, 1), .5)
        self._global_c = np.where(ccount > 0, self._C0.sum(0) / np.maximum(ccount, 1), np.nan)
        self._train_slices = np.array([s or "" for s in (slices if slices is not None else [None]*len(X))], dtype=object)
        self._slice_q, self._slice_c = {}, {}
        prior = self.config.slice_prior
        for name in sorted(set(self._train_slices) - {""}):
            rows = self._train_slices == name
            qs, qn = self._Q0[rows].sum(0), self._qmask[rows].sum(0)
            cs, cn = self._C0[rows].sum(0), self._cmask[rows].sum(0)
            self._slice_q[name] = (qs + prior * self._global_q) / (qn + prior)
            with np.errstate(invalid="ignore", divide="ignore"):
                self._slice_c[name] = np.where(np.isnan(self._global_c), np.where(cn > 0, cs / np.maximum(cn, 1), np.nan),
                                               (cs + prior * np.nan_to_num(self._global_c)) / (cn + prior))
        self._centroids = (np.asarray(centroids, dtype=float) if centroids is not None else
                           spherical_kmeans(X, self.config.clusters, self.config.iterations, self.config.seed))
        assign = np.argmax(X @ self._centroids.T, axis=1)
        self._assign = assign
        clusters, m = len(self._centroids), Q.shape[1]
        self._cq_sum, self._cq_cnt = np.zeros((clusters, m)), np.zeros((clusters, m))
        self._cc_sum, self._cc_cnt = np.zeros((clusters, m)), np.zeros((clusters, m))
        np.add.at(self._cq_sum, assign, self._Q0)
        np.add.at(self._cq_cnt, assign, self._qmask.astype(float))
        np.add.at(self._cc_sum, assign, self._C0)
        np.add.at(self._cc_cnt, assign, self._cmask.astype(float))
        self._fitted = True
        self._digest = None
        return self

    def with_config(self, config: PredictorConfig) -> QualityPredictor:
        """Refit with new hyperparameters, reusing clusters when their settings are unchanged."""
        same = (config.clusters, config.iterations, config.seed) == (
            self.config.clusters, self.config.iterations, self.config.seed)
        slices = [s or None for s in self._train_slices]
        return QualityPredictor(config).fit(self._X, self._Q, self._C, slices,
                                            centroids=self._centroids if same else None)

    @property
    def endpoints(self) -> int:
        return self._Q.shape[1]

    @property
    def training_size(self) -> int:
        return len(self._X)

    @property
    def global_quality(self) -> np.ndarray:
        return self._global_q.copy()

    def artifact_digest(self) -> str:
        if not self._fitted:
            raise ValueError("predictor is not fitted")
        if self._digest is None:
            self._digest = hashlib.sha256((repr(sorted(self.config.identity().items())) + _array_digest(
                self._X, self._Q, self._C, self._centroids)).encode()).hexdigest()
        return self._digest

    # ----- prediction ----------------------------------------------------------------------
    def _base(self, slices, table, fallback) -> np.ndarray:
        rows = [table.get(s or "", fallback) if s else fallback for s in slices]
        return np.array(rows, dtype=float)

    def predict(self, features, slices=None, *, exclude=None, batch: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Return (quality, cost) forecasts with shape (requests x endpoints).

        ``exclude`` optionally gives, per query, a training row index to leave out of the
        neighbor search (leave-one-out scoring of training rows).
        """
        if not self._fitted:
            raise ValueError("predictor is not fitted")
        X = normalize(features)
        n = len(X)
        slices = list(slices) if slices is not None else [None] * n
        cfg = self.config
        exclude = None if exclude is None else np.asarray(exclude, dtype=int)
        base_q = self._base(slices, self._slice_q, self._global_q)
        base_c = self._base(slices, self._slice_c, self._global_c)
        if exclude is not None:
            # Leave-one-out slice means: remove each excluded row from its own slice table.
            prior = cfg.slice_prior
            for i, j in enumerate(exclude):
                name = slices[i] or ""
                if j < 0 or not name or name not in self._slice_q or self._train_slices[j] != name:
                    continue
                rows = self._train_slices == name
                qs = self._Q0[rows].sum(0) - self._Q0[j]
                qn = self._qmask[rows].sum(0) - self._qmask[j]
                base_q[i] = (qs + prior * self._global_q) / (qn + prior)
        sims = X @ self._centroids.T
        top_n = min(cfg.cluster_top, sims.shape[1])
        top = np.argpartition(-sims, top_n - 1, axis=1)[:, :top_n]
        top_sims = np.take_along_axis(sims, top, axis=1)
        weights = np.exp(cfg.sharpness * (top_sims - top_sims.max(axis=1, keepdims=True)))
        weights /= weights.sum(axis=1, keepdims=True)
        kc = cfg.cluster_prior
        cq_sum, cq_cnt = self._cq_sum[top], self._cq_cnt[top]
        cc_sum, cc_cnt = self._cc_sum[top], self._cc_cnt[top]
        if exclude is not None:
            # Leave-one-out cluster tables: remove each excluded row from its own cluster.
            valid = exclude >= 0
            own = np.zeros(top.shape, dtype=bool)
            own[valid] = top[valid] == self._assign[exclude[valid]][:, None]
            safe = np.where(valid, exclude, 0)
            cq_sum = cq_sum - own[:, :, None] * self._Q0[safe][:, None, :]
            cq_cnt = cq_cnt - own[:, :, None] * self._qmask[safe][:, None, :]
            cc_sum = cc_sum - own[:, :, None] * self._C0[safe][:, None, :]
            cc_cnt = cc_cnt - own[:, :, None] * self._cmask[safe][:, None, :]
        cluster_q = ((cq_sum + kc * base_q[:, None, :]) / (cq_cnt + kc) * weights[:, :, None]).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            no_base = np.isnan(base_c)[:, None, :]
            raw_c = np.where(no_base, np.where(cc_cnt > 0, cc_sum / np.maximum(cc_cnt, 1), np.nan),
                             (cc_sum + kc * np.nan_to_num(base_c)[:, None, :]) / (cc_cnt + kc))
            known = ~np.isnan(raw_c)
            cw = weights[:, :, None] * known
            cluster_c = np.where(cw.sum(1) > 0, np.nansum(raw_c * cw, axis=1) / np.maximum(cw.sum(1), 1e-300), np.nan)
        k = min(cfg.neighbors, len(self._X) - (1 if exclude is not None else 0))
        if k <= 0:
            return np.clip(cluster_q, 0., 1.), cluster_c
        quality, cost = np.empty_like(cluster_q), np.empty_like(cluster_c)
        query_slices = np.array([s or "" for s in slices], dtype=object)
        use_slices = cfg.slice_neighbors and bool(self._slice_q) and any(query_slices != "")
        for start in range(0, n, batch):
            stop = min(n, start + batch)
            S = X[start:stop] @ self._X.T
            if use_slices:
                same = (query_slices[start:stop, None] == self._train_slices[None, :]) & (query_slices[start:stop, None] != "")
                S = S + same.astype(float)
            if exclude is not None:
                rows = np.arange(stop - start)
                valid = exclude[start:stop] >= 0
                S[rows[valid], exclude[start:stop][valid]] = -np.inf
            index = np.argpartition(-S, k - 1, axis=1)[:, :k]
            near = np.take_along_axis(S, index, axis=1)
            w = np.exp(cfg.sharpness * (near - near.max(axis=1, keepdims=True)))
            qm = self._qmask[index]
            num = (w[:, :, None] * self._Q0[index]).sum(1)
            den = (w[:, :, None] * qm).sum(1)
            kk = cfg.neighbor_prior
            quality[start:stop] = (num + kk * cluster_q[start:stop]) / (den + kk)
            cm = self._cmask[index]
            cnum = (w[:, :, None] * self._C0[index]).sum(1)
            cden = (w[:, :, None] * cm).sum(1)
            prior_c = cluster_c[start:stop]
            with np.errstate(invalid="ignore", divide="ignore"):
                cost[start:stop] = np.where(np.isnan(prior_c), np.where(cden > 0, cnum / np.maximum(cden, 1e-300), np.nan),
                                            (cnum + kk * np.nan_to_num(prior_c)) / (cden + kk))
        return np.clip(quality, 0., 1.), cost

    def predict_training(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out neighbor forecasts for the training rows themselves."""
        slices = [s or None for s in self._train_slices]
        return self.predict(self._X, slices, exclude=np.arange(len(self._X)))


def brier(prediction: np.ndarray, quality: np.ndarray) -> float:
    observed = ~np.isnan(quality)
    if not observed.any():
        return float("nan")
    return float(np.mean((prediction[observed] - quality[observed]) ** 2))


def tune(train_features, train_quality, train_cost, train_slices, valid_features, valid_quality, valid_slices,
         *, seed: int = 0) -> tuple[PredictorConfig, list[dict]]:
    """Select hyperparameters by validation Brier score in two stages (clusters, then neighbors).

    Only training and validation data are used. The returned trial log is descriptive.
    """
    n = len(train_features)
    has_slices = any(s for s in (train_slices or [])) and any(s for s in (valid_slices or []))
    sizes = sorted({c for c in (4, 8, 16, 32, 64) if c <= max(4, n // 10)}) or [1]
    trials, best = [], None
    fitted: dict[int, QualityPredictor] = {}
    for clusters in sizes:
        for cluster_prior in (4., 16.):
            config = PredictorConfig(neighbors=0, clusters=clusters, cluster_prior=cluster_prior, seed=seed)
            if clusters not in fitted:
                fitted[clusters] = QualityPredictor(config).fit(train_features, train_quality, train_cost, train_slices)
                model = fitted[clusters]
            else:
                model = fitted[clusters].with_config(config)
            score = brier(model.predict(valid_features, valid_slices)[0], np.asarray(valid_quality, dtype=float))
            trials.append({"stage": "clusters", "config": config.identity(), "brier": score})
            if best is None or score < best[0] - 1e-12:
                best = (score, config, clusters)
    _, chosen, clusters = best
    for neighbors in (8, 16, 32, 64):
        for neighbor_prior in (1., 4., 16.):
            for sharpness in (10., 30.):
                for slice_neighbors in ((True, False) if has_slices else (False,)):
                    config = replace(chosen, neighbors=neighbors, neighbor_prior=neighbor_prior,
                                     sharpness=sharpness, slice_neighbors=slice_neighbors)
                    model = fitted[clusters].with_config(config)
                    score = brier(model.predict(valid_features, valid_slices)[0], np.asarray(valid_quality, dtype=float))
                    trials.append({"stage": "neighbors", "config": config.identity(), "brier": score})
                    if score < best[0] - 1e-12:
                        best = (score, config, clusters)
    return best[1], trials
