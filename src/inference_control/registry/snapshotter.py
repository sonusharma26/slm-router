"""Content-addressed identities; configuration/pricing changes create new snapshots."""
from __future__ import annotations
from dataclasses import dataclass
from inference_control.contracts import EndpointSnapshot
from inference_control.util import digest


@dataclass(frozen=True)
class SnapshotChange:
    before: str | None
    after: str
    endpoint_id: str
    changed_metrics: tuple[str, ...]


class Snapshotter:
    def __init__(self, state):
        self.state = state

    def refresh(self, snapshot: EndpointSnapshot) -> SnapshotChange:
        self.state.refresh()
        prior = self.state.endpoints.get(snapshot.endpoint_id)
        metrics = set()
        if prior:
            if prior.capability_revision != snapshot.capability_revision:
                metrics.update(("quality", "cost", "latency", "failure"))
            if (prior.input_price_per_million, prior.output_price_per_million, prior.price_version) != (
                snapshot.input_price_per_million, snapshot.output_price_per_million, snapshot.price_version):
                metrics.add("cost")
            if (prior.available, prior.healthy) != (snapshot.available, snapshot.healthy):
                metrics.add("availability")
        if prior != snapshot:
            self.state.write("endpoint_snapshot", snapshot.model_dump(mode="json"),
                             key=f"snapshot-activation:{snapshot.endpoint_id}:{self.state.sequence}:{snapshot.snapshot_id}")
        return SnapshotChange(prior.snapshot_id if prior else None, snapshot.snapshot_id,
                              snapshot.endpoint_id, tuple(sorted(metrics)))

    def refresh_from_adapter(self, endpoint_id: str, adapter) -> SnapshotChange:
        """Only adapters exposing verified metadata can refresh; no invented alias revision."""
        if not hasattr(adapter, "snapshot"):
            raise ValueError("adapter has no authoritative metadata; provide a manual snapshot")
        snapshot = adapter.snapshot(endpoint_id)
        if not isinstance(snapshot, EndpointSnapshot) or snapshot.endpoint_id != endpoint_id:
            raise ValueError("invalid endpoint metadata")
        return self.refresh(snapshot)
