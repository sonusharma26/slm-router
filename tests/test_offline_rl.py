"""Robustness tests for OfflineRLPolicy.select() and fqe_value()."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from slm_router.ml_core.routing.offline_rl import OfflineRLPolicy
from slm_router.ml_core.routing.trace_adapter import TrainingSample
from slm_router.types import ModelSpec


def _model(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        name=model_id,
        tier="SLM",
        price_in_per_m=0.1,
        price_out_per_m=0.1,
        context_len=8192,
    )


def _sample(context, arm_index, quality, cost_usd=0.001, latency_ms=100.0):
    return TrainingSample(
        context=np.asarray(context, dtype=np.float32),
        arm_index=arm_index,
        quality=quality,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )


def _fitted_policy(n_arms=2, dim=3, seed=0):
    rng = np.random.default_rng(seed)
    traces = []
    for arm in range(n_arms):
        for _ in range(20):
            ctx = rng.normal(size=dim)
            # Arm 0 is reliably good, arm 1 is reliably bad, so the policy
            # learns a clear preference.
            quality = 0.9 if arm == 0 else 0.1
            traces.append(_sample(ctx, arm, quality))
    policy = OfflineRLPolicy(n_arms=n_arms, dim=dim)
    policy.fit(traces)
    return policy


class TestSelectArmCountGuard:
    def test_select_raises_on_fewer_candidates(self):
        policy = _fitted_policy(n_arms=3)
        candidates = [_model("m0"), _model("m1")]
        with pytest.raises(ValueError):
            policy.select(np.zeros(3, dtype=np.float32), candidates)

    def test_select_raises_on_more_candidates(self):
        policy = _fitted_policy(n_arms=2)
        candidates = [_model("m0"), _model("m1"), _model("m2")]
        with pytest.raises(ValueError):
            policy.select(np.zeros(2, dtype=np.float32), candidates)

    def test_select_succeeds_when_counts_match(self):
        policy = _fitted_policy(n_arms=2)
        candidates = [_model("m0"), _model("m1")]
        decision = policy.select(np.zeros(3, dtype=np.float32), candidates)
        assert decision.chosen_model in {"m0", "m1"}
        assert set(decision.expected_reward.keys()) == {"m0", "m1"}


class TestFqeValueDoublyRobust:
    def test_fqe_value_returns_finite_float(self):
        policy = _fitted_policy(n_arms=2)
        rng = np.random.default_rng(1)
        traces = [
            _sample(rng.normal(size=3), arm, 0.9 if arm == 0 else 0.1)
            for arm in [0, 1, 0, 1]
        ]
        value = policy.fqe_value(traces)
        assert isinstance(value, float)
        assert np.isfinite(value)

    def test_fqe_value_empty_traces_is_zero(self):
        policy = _fitted_policy(n_arms=2)
        assert policy.fqe_value([]) == 0.0

    def test_fqe_value_matches_direct_method_when_no_greedy_match(self):
        # Construct a policy where arm 0 is always greedy (its Q-function
        # predicts high reward everywhere), and log all traces under arm 1.
        # Since a_log != a_greedy for every trace, the correction term is
        # always zero, so fqe_value must equal the pure direct-method mean
        # of Q(x, a_greedy).
        n_arms, dim = 2, 3
        traces_fit = []
        rng = np.random.default_rng(2)
        for _ in range(20):
            traces_fit.append(_sample(rng.normal(size=dim), 0, 0.95))
        for _ in range(20):
            traces_fit.append(_sample(rng.normal(size=dim), 1, 0.05))
        policy = OfflineRLPolicy(n_arms=n_arms, dim=dim)
        policy.fit(traces_fit)

        eval_traces = [_sample(rng.normal(size=dim), 1, 0.05) for _ in range(10)]

        # Sanity check: for every eval trace, greedy arm must differ from
        # logged arm (1), otherwise the test setup doesn't isolate the DM
        # term.
        for tr in eval_traces:
            q_vals = [policy._q_value(i, np.asarray(tr.context, dtype=np.float32)) for i in range(n_arms)]
            assert int(np.argmax(q_vals)) == 0
            assert tr.arm_index == 1

        expected_dm = float(np.mean([
            max(policy._q_value(i, np.asarray(tr.context, dtype=np.float32)) for i in range(n_arms))
            for tr in eval_traces
        ]))
        actual = policy.fqe_value(eval_traces)
        assert actual == pytest.approx(expected_dm, abs=1e-6)
