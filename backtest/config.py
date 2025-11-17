from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.const import MarketUniverse
from utils.root import DATA_ROOT, PROJECT_ROOT, SCORES_ROOT


def _default_scores_path() -> Path:
    return SCORES_ROOT / "price_trends_score_test_i20_r20.parquet"


def _default_close_path() -> Path:
    return DATA_ROOT / "close.parquet"


def _default_output_path() -> Path:
    return PROJECT_ROOT / "bt"


@dataclass(frozen=True)
class BacktestConfig:
    scores_path: Path | Sequence[Path] = field(default_factory=_default_scores_path)
    close_path: Path | str = field(default_factory=_default_close_path)
    output_dir: Path | str = field(default_factory=_default_output_path)
    constituent_universe: MarketUniverse | None = MarketUniverse.KOSPI200
    constituent_path: Path | str | None = None

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
    show_progress: bool = True

    def __post_init__(self) -> None:
        def to_project_path(raw: Path | str) -> Path:
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

        raw_scores = self.scores_path
        if isinstance(raw_scores, (str, Path)):
            score_paths = (to_project_path(raw_scores),)
        else:
            score_paths = tuple(to_project_path(p) for p in raw_scores)
        if not score_paths:
            raise ValueError("At least one score file must be provided.")
        object.__setattr__(self, "scores_path", score_paths)
        object.__setattr__(self, "close_path", to_project_path(self.close_path))
        object.__setattr__(self, "output_dir", to_project_path(self.output_dir))
        object.__setattr__(self, "constituent_path", self._resolve_constituent_path(to_project_path))
        self._validate_numeric_fields()

    @property
    def score_paths(self) -> Tuple[Path, ...]:
        """Backwards-compatible alias for scores_path."""
        return tuple(self.scores_path)

    def _resolve_constituent_path(self, converter: Callable[[Path | str], Path]) -> Path | None:
        if self.constituent_path is not None:
            return converter(self.constituent_path)
        if self.constituent_universe is None:
            return None
        universe_path = DATA_ROOT / self.constituent_universe.parquet_filename
        return converter(universe_path)

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

    def quantile_ids(self) -> Tuple[int, ...]:
        if not self.active_quantiles:
            return tuple(range(self.quantiles))
        unique_sorted = tuple(sorted(set(int(q) for q in self.active_quantiles)))
        for q in unique_sorted:
            if q < 0 or q >= self.quantiles:
                raise ValueError(f"Quantile id {q} is outside the range [0, {self.quantiles}).")
        return unique_sorted

    def with_overrides(self, **updates: Any) -> "BacktestConfig":
        normalized = dict(updates)
        if "score_paths" in normalized and "scores_path" not in normalized:
            normalized["scores_path"] = normalized.pop("score_paths")
        current = {f.name: getattr(self, f.name) for f in fields(self) if f.init}
        current.update(normalized)
        return BacktestConfig(**current)

    def ensure_io_paths(self, *, scores_in_memory: bool = False, prices_in_memory: bool = False) -> None:
        if not scores_in_memory:
            for path in self.score_paths:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Scores parquet not found: {path}")
        if not prices_in_memory and not self.close_path.exists():
            raise FileNotFoundError(f"Close price parquet not found: {self.close_path}")
        if self.constituent_path is not None and not self.constituent_path.exists():
            raise FileNotFoundError(f"Constituent parquet not found: {self.constituent_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
