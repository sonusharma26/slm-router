"""Immutable endpoint and price snapshot registry (V2-200)."""

from __future__ import annotations
from dataclasses import dataclass
from inference_control.contracts import EndpointSnapshot


@dataclass(frozen=True)
class PriceSnapshot:
    price_version: str
    endpoint_id: str
    input_price_per_million: float
    output_price_per_million: float
    effective_at: str


class EndpointRegistry:
    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], EndpointSnapshot] = {}
        self._prices: dict[tuple[str, str], PriceSnapshot] = {}

    def register(self, snapshot: EndpointSnapshot) -> EndpointSnapshot:
        key = (snapshot.endpoint_id, snapshot.revision)
        prior = self._snapshots.get(key)
        if prior is not None and prior != snapshot:
            raise ValueError("immutable endpoint snapshot conflict")
        self._snapshots[key] = snapshot
        return snapshot

    def register_price(self, price: PriceSnapshot) -> PriceSnapshot:
        key = (price.endpoint_id, price.price_version)
        prior = self._prices.get(key)
        if prior is not None and prior != price:
            raise ValueError("immutable price snapshot conflict")
        self._prices[key] = price
        return price

    def get(self, endpoint_id: str, revision: str) -> EndpointSnapshot:
        return self._snapshots[(endpoint_id, revision)]

    def snapshots(self) -> tuple[EndpointSnapshot, ...]:
        return tuple(sorted(self._snapshots.values(), key=lambda x: (x.endpoint_id, x.revision)))
