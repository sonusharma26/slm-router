"""Append-only policy lifecycle with atomic promotion/rollback (V2-506/V2-604/V2-605)."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum


class PolicyState(StrEnum):
    DRAFT = "draft"
    OFFLINE_VALIDATED = "offline_validated"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    STALE = "stale"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


ALLOWED = {
    PolicyState.DRAFT: {PolicyState.OFFLINE_VALIDATED},
    PolicyState.OFFLINE_VALIDATED: {PolicyState.SHADOW},
    PolicyState.SHADOW: {PolicyState.CANARY},
    PolicyState.CANARY: {PolicyState.ACTIVE, PolicyState.REJECTED},
    PolicyState.ACTIVE: {PolicyState.STALE, PolicyState.ROLLED_BACK},
    PolicyState.STALE: {PolicyState.DRAFT},
}


@dataclass(frozen=True)
class Transition:
    sequence: int
    policy_version: str
    from_state: PolicyState
    to_state: PolicyState
    actor: str
    reason: str
    idempotency_key: str


class PolicyLifecycle:
    def __init__(self):
        self.states: dict[str, PolicyState] = {}
        self.transitions: list[Transition] = []
        self.active: str | None = None
        self.rollback_target: str | None = None
        self._keys = set()

    def create(self, version: str):
        if version in self.states:
            raise ValueError("policy already exists")
        self.states[version] = PolicyState.DRAFT

    def transition(
        self, version: str, target: PolicyState, *, actor: str, reason: str, idempotency_key: str
    ) -> Transition:
        if idempotency_key in self._keys:
            return next(t for t in self.transitions if t.idempotency_key == idempotency_key)
        current = self.states[version]
        if target not in ALLOWED.get(current, set()):
            raise ValueError(f"invalid lifecycle transition {current}->{target}")
        if target == PolicyState.ACTIVE:
            self.rollback_target = self.active
            self.active = version
        if target == PolicyState.ROLLED_BACK:
            if version != self.active or not self.rollback_target:
                raise ValueError("no configured rollback target")
            self.active = self.rollback_target
        event = Transition(
            len(self.transitions) + 1, version, current, target, actor, reason, idempotency_key
        )
        self.states[version] = target
        self.transitions.append(event)
        self._keys.add(idempotency_key)
        return event


@dataclass(frozen=True)
class SequentialStopRule:
    max_violations: int
    max_failure_rate: float
    minimum_samples: int

    def stop(self, violations: int, samples: int) -> bool:
        return samples >= self.minimum_samples and (
            violations >= self.max_violations or violations / samples > self.max_failure_rate
        )
