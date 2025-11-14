from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.root import DATA_ROOT, PROJECT_ROOT, SCORES_ROOT


def _default_scores_path() -> Path:
    return SCORES_ROOT / "price_trends_score_test_i20_r20.parquet"


def _default_close_path() -> Path:
    return DATA_ROOT / "close.parquet"


def _default_output_path() -> Path:
    return PROJECT_ROOT / "bt"


@dataclass(frozen=True)
class BacktestConfig:
    scores_path: Path = field(default_factory=_default_scores_path)
    close_path: Path = field(default_factory=_default_close_path)
    output_dir: Path = field(default_factory=_default_output_path)

    initial_capital: float = 100_000_000.0
    quantiles: int = 5
    rebalance_frequency: str = "M"
    min_assets: int = 30
    active_quantiles: Optional[Sequence[int]] = None

    allow_partial_buckets: bool = False

    apply_trading_costs: bool = False
    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    tax_bps: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scores_path", Path(self.scores_path))
        object.__setattr__(self, "close_path", Path(self.close_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        self._validate_numeric_fields()

    def _validate_numeric_fields(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.quantiles < 2:
            raise ValueError("quantiles must be at least 2.")
        if self.min_assets < self.quantiles:
            raise ValueError("min_assets must be >= quantiles to form buckets.")
        for field_name in ("buy_cost_bps", "sell_cost_bps", "tax_bps"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def quantile_ids(self) -> Tuple[int, ...]:
        if not self.active_quantiles:
            return tuple(range(self.quantiles))
        unique_sorted = tuple(sorted(set(int(q) for q in self.active_quantiles)))
        for q in unique_sorted:
            if q < 0 or q >= self.quantiles:
                raise ValueError(f"Quantile id {q} is outside the range [0, {self.quantiles}).")
        return unique_sorted

    def with_overrides(self, **updates: Any) -> "BacktestConfig":
        current = {f.name: getattr(self, f.name) for f in fields(self)}
        current.update(updates)
        return BacktestConfig(**current)

    def ensure_io_paths(self) -> None:
        if not self.scores_path.exists():
            raise FileNotFoundError(f"Scores parquet not found: {self.scores_path}")
        if not self.close_path.exists():
            raise FileNotFoundError(f"Close price parquet not found: {self.close_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
