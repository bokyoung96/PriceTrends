from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.const import MarketUniverse
from utils.root import DATA_ROOT, PROJECT_ROOT, SCORES_ROOT
from backtest.costs import ExecutionCostModel
from backtest.data_sources import BacktestDataLoader
from backtest.grouping import PortfolioGroupingStrategy, QuantileGroupingStrategy


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
    min_assets: int = 20
    active_quantiles: Sequence[int] | None = None
    benchmark_symbol: str | None = "IKS200"
    allow_partial_buckets: bool = False
    portfolio_grouping: PortfolioGroupingStrategy | None = None

    apply_trading_costs: bool = False
    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    tax_bps: float = 0.0
    entry_lag: int = 0
    show_progress: bool = True

    def __post_init__(self) -> None:
        prepared_scores = self._prepare_score_files(self.scores_path)
        object.__setattr__(self, "scores_path", prepared_scores)
        object.__setattr__(self, "close_path", self._to_project_path(self.close_path))
        object.__setattr__(self, "output_dir", self._to_project_path(self.output_dir))
        object.__setattr__(self, "constituent_path", self._select_constituent_path())
        self._validate_numeric_fields()

    def _to_project_path(self, raw: Path | str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _prepare_score_files(self, raw_paths: Path | Sequence[Path]) -> Tuple[Path, ...]:
        if isinstance(raw_paths, (str, Path)):
            candidates: Sequence[Path | str] = (raw_paths,)
        else:
            candidates = tuple(raw_paths)
        if not candidates:
            raise ValueError("At least one score file must be provided.")
        return tuple(self._to_project_path(path) for path in candidates)

    def _select_constituent_path(self) -> Path | None:
        if self.constituent_path is not None:
            return self._to_project_path(self.constituent_path)
        if self.constituent_universe is None:
            return None
        universe_path = DATA_ROOT / self.constituent_universe.parquet_filename
        return self._to_project_path(universe_path)

    def grouping_strategy(self) -> PortfolioGroupingStrategy:
        if self.portfolio_grouping is not None:
            return self.portfolio_grouping
        return QuantileGroupingStrategy(
            quantiles=self.quantiles,
            min_assets=self.min_assets,
            allow_partial=self.allow_partial_buckets,
            enabled_quantiles=self.active_quantiles,
        )

    def with_overrides(self, **updates: Any) -> "BacktestConfig":
        current = {field.name: getattr(self, field.name) for field in fields(self) if field.init}
        current.update(updates)
        return BacktestConfig(**current)

    def ensure_io_paths(self, *, scores_in_memory: bool = False, prices_in_memory: bool = False) -> None:
        if not scores_in_memory:
            for path in self.scores_path:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Scores parquet not found: {path}")
        if not prices_in_memory and not Path(self.close_path).exists():
            raise FileNotFoundError(f"Close price parquet not found: {self.close_path}")
        if self.constituent_path is not None and not Path(self.constituent_path).exists():
            raise FileNotFoundError(f"Constituent parquet not found: {self.constituent_path}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def cost_model(self) -> ExecutionCostModel:
        return ExecutionCostModel(
            enabled=self.apply_trading_costs,
            buy_bps=self.buy_cost_bps,
            sell_bps=self.sell_cost_bps,
            tax_bps=self.tax_bps,
        )

    def data_loader(
        self,
        *,
        scores: Any | None = None,
        prices: Any | None = None,
    ) -> BacktestDataLoader:
        return BacktestDataLoader(
            scores_source=scores if scores is not None else self.scores_path[0],
            close_source=prices if prices is not None else self.close_path,
            constituent_source=self.constituent_path,
            benchmark_symbol=self.benchmark_symbol,
        )

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
