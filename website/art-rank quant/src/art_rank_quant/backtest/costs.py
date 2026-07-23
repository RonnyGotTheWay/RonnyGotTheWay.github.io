from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CostModel:
    commission_bps: float = 3.0
    stamp_duty_bps_sell: float = 5.0
    slippage_bps: float = 5.0
    impact_coefficient: float = 0.1

    def estimate(self, notional: float, side: str, participation: float = 0.0) -> float:
        bps = self.commission_bps + self.slippage_bps
        if side == "sell":
            bps += self.stamp_duty_bps_sell
        return abs(notional) * (bps / 10_000 + self.impact_coefficient * participation**2)
