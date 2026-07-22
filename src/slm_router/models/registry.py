"""ModelRegistry — loads models.yaml and provides lookups + cost computation."""

from __future__ import annotations

from pathlib import Path

import yaml

from slm_router.types import CostBreakdown, ModelSpec, Tier, Usage


class ModelRegistry:
    """In-memory registry of ModelSpec objects keyed by model id."""

    def __init__(self, specs: list[ModelSpec]) -> None:
        self._by_id: dict[str, ModelSpec] = {s.id: s for s in specs}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelRegistry":
        """Load from a YAML file with a top-level ``models:`` list."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        specs = [ModelSpec.model_validate(m) for m in raw.get("models", [])]
        return cls(specs)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> ModelSpec:
        """Return the ModelSpec for *model_id*, or raise KeyError."""
        try:
            return self._by_id[model_id]
        except KeyError:
            known = ", ".join(sorted(self._by_id)) or "<empty registry>"
            raise KeyError(
                f"Unknown model id {model_id!r}. Known ids: {known}"
            ) from None

    def by_tier(self, tier: Tier) -> list[ModelSpec]:
        """Return all models with the given tier, sorted by price_in_per_m."""
        return sorted(
            (s for s in self._by_id.values() if s.tier == tier),
            key=lambda s: s.price_in_per_m,
        )

    def all(self) -> list[ModelSpec]:
        """Return all registered ModelSpec objects."""
        return list(self._by_id.values())

    # ------------------------------------------------------------------
    # Cost
    # ------------------------------------------------------------------

    def cost(self, model_id: str, usage: Usage) -> CostBreakdown:
        """Compute CostBreakdown from registry pricing and actual token usage."""
        spec = self.get(model_id)
        input_usd = spec.price_in_per_m * usage.prompt_tokens / 1_000_000
        output_usd = spec.price_out_per_m * usage.completion_tokens / 1_000_000
        return CostBreakdown(
            input_usd=input_usd,
            output_usd=output_usd,
            total_usd=input_usd + output_usd,
        )
