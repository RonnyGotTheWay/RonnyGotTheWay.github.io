from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    version: str
    owner: str
    dependencies: tuple[str, ...]


REGISTRY = {
    "momentum_20d": FeatureDefinition("momentum_20d", "1", "research", ("close_forward_adjusted",)),
    "volatility_20d": FeatureDefinition("volatility_20d", "1", "research", ("daily_return",)),
    "turnover_proxy": FeatureDefinition("turnover_proxy", "1", "research", ("amount",)),
}
