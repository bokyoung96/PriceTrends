from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.root import DATA_ROOT, PROJECT_ROOT, SCORES_ROOT  # noqa: E402


def _default_scores_path() -> Path:
    return SCORES_ROOT / "price_trends_score_test_i20_r20.parquet"


def _default_close_path() -> Path:
    return DATA_ROOT / "close.parquet"


def _default_output_path() -> Path:
    return PROJECT_ROOT / "bt"


@dataclass(frozen=True)
class BacktestConfig:
    scores_path: Path | Sequence[Path] = field(default_factory=_default_scores_path)
    close_path: Path = field(default_factory=_default_close_path)
    output_dir: Path = field(default_factory=_default_output_path)
    score_paths: Tuple[Path, ...] = field(init=False, repr=False)

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
    entry_lag: int = 0
    min_price_relative: float = 0.05
    max_price_relative: float = 20.0

    def __post_init__(self) -> None:
        normalized = self._normalize_scores(self.scores_path)
        object.__setattr__(self, "score_paths", normalized)
        object.__setattr__(self, "scores_path", normalized[0])
        object.__setattr__(self, "close_path", self._resolve_path(self.close_path))
        object.__setattr__(self, "output_dir", self._resolve_path(self.output_dir))
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
        if self.entry_lag < 0:
            raise ValueError("entry_lag must be non-negative.")
        if not (0 < self.min_price_relative < 1):
            raise ValueError("min_price_relative must be in the interval (0, 1).")
        if self.max_price_relative <= 1:
            raise ValueError("max_price_relative must be greater than 1.")
        if self.min_price_relative >= self.max_price_relative:
            raise ValueError("min_price_relative must be less than max_price_relative.")

    def quantile_ids(self) -> Tuple[int, ...]:
        if not self.active_quantiles:
            return tuple(range(self.quantiles))
        unique_sorted = tuple(sorted(set(int(q) for q in self.active_quantiles)))
        for q in unique_sorted:
            if q < 0 or q >= self.quantiles:
                raise ValueError(f"Quantile id {q} is outside the range [0, {self.quantiles}).")
        return unique_sorted

    def with_overrides(self, **updates: Any) -> "BacktestConfig":
        current = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        current.update(updates)
        return BacktestConfig(**current)

    def ensure_io_paths(self, *, scores_in_memory: bool = False, prices_in_memory: bool = False) -> None:
        if not scores_in_memory:
            for path in self.score_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Scores parquet not found: {path}")
        if not prices_in_memory and not self.close_path.exists():
            raise FileNotFoundError(f"Close price parquet not found: {self.close_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate

    def _normalize_scores(self, raw: Path | Sequence[Path]) -> Tuple[Path, ...]:
        if isinstance(raw, (str, Path)):
            return (self._resolve_path(raw),)
        return tuple(self._resolve_path(p) for p in raw)

    def resolve_score_paths(self, explicit: Sequence[Path | str] | None = None) -> list[Path]:
        if explicit:
            return [self._resolve_path(p) for p in explicit]
        return list(self.score_paths)
