from __future__ import annotations

from dataclasses import dataclass


def _bps_to_ratio(bps: float) -> float:
    return max(0.0, bps) / 10_000.0


@dataclass(frozen=True)
class ExecutionCostModel:
    """Applies optional trading costs (slippage, fees, and taxes)."""

    enabled: bool = False
    buy_bps: float = 0.0
    sell_bps: float = 0.0
    tax_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("buy_bps", "sell_bps", "tax_bps"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")

    @classmethod
    def disabled(cls) -> "ExecutionCostModel":
        return cls(enabled=False)

    def net_entry_capital(self, capital: float) -> float:
        if not self.enabled or capital <= 0:
            return float(capital)
        return float(capital * (1.0 - _bps_to_ratio(self.buy_bps)))

    def net_exit_capital(self, gross_exit: float, tradable_fraction: float = 1.0) -> float:
        if not self.enabled or gross_exit <= 0:
            return float(gross_exit)
        fraction = min(max(tradable_fraction, 0.0), 1.0)
        exit_bps = self.sell_bps + self.tax_bps
        factor = max(0.0, 1.0 - _bps_to_ratio(exit_bps) * fraction)
        return float(gross_exit * factor)

    def total_round_trip_bps(self) -> float:
        if not self.enabled:
            return 0.0
        return float(self.buy_bps + self.sell_bps + self.tax_bps)

    def is_enabled(self) -> bool:
        return self.enabled and self.total_round_trip_bps() > 0
