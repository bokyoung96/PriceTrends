from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import ExecutionCostModel
from backtest.data_sources import BacktestDataLoader
from backtest.grouping import (PortfolioGroupingStrategy,
                               QuantileGroupingStrategy,
                               SectorNeutralGroupingStrategy)
from core.spec import MarketMetric, MarketUniverse
from utils.root import DATA_ROOT, PROJECT_ROOT, SCORES_ROOT


def score_path(
    input_days: int,
    return_days: int,
    *,
    mode: str = "TEST",
    fusion: bool = False,
    ensemble: bool = False,
) -> Path:
    suffix = "_fusion" if fusion else ""
    mode_str = mode.lower()
    if ensemble:
        return SCORES_ROOT / f"price_trends_score_{mode_str}_ensemble{suffix}.parquet"
    return SCORES_ROOT / f"price_trends_score_{mode_str}_i{input_days}_r{return_days}{suffix}.parquet"


def transformer_score_path(
    mode: str = "TEST",
    timeframe: str = "MEDIUM",
    name: str = "transformer",
) -> Path:
    mode_str = mode.lower()
    timeframe_str = timeframe.lower()

    if name == "transformer":
        full_name = f"transformer_{mode_str}_{timeframe_str}"
    else:
        full_name = name
        
    return SCORES_ROOT / f"price_trends_score_{full_name}.parquet"


def _default_scores_path() -> Path:
    return score_path(20, 20, mode="TEST", fusion=False)


def _default_close_path() -> Path:
    return DATA_ROOT / "close.parquet"


def _default_output_path() -> Path:
    return PROJECT_ROOT / "bt"


def _default_weight_data_path() -> Path:
    return DATA_ROOT / MarketMetric.MKTCAP.parquet_filename


def _default_open_path() -> Path:
    return DATA_ROOT / "open.parquet"


def _default_benchmark_path() -> Path:
    return DATA_ROOT / MarketMetric.BM.parquet_filename


def _default_sector_path() -> Path:
    return DATA_ROOT / MarketMetric.SECTOR.parquet_filename


class BenchmarkType(str, Enum):
    KOSPI200 = "IKS200"
    KOSPI200EQ = "IKS500"
    KOSPI200TR = "IKS270"
    KOSPI = "IKS001"
    KOSPITR = "IKS170"
    KOSPIBIG = "IKS002"
    KOSPIMID = "IKS003"
    KOSPISMALL = "IKS004"

    @classmethod
    def parse(cls, raw: "BenchmarkType | str | None") -> "BenchmarkType | None":
        if raw is None:
            return None
        if isinstance(raw, BenchmarkType):
            return raw
        normalized = str(raw).strip().upper()
        if normalized in cls.__members__:
            return cls[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(f"Unknown benchmark type: {raw}")


class PortfolioWeights(str, Enum):
    EQUAL = "eq"
    MARKET_CAP = "mc"

    @property
    def requires_market_caps(self) -> bool:
        return self is PortfolioWeights.MARKET_CAP

    @classmethod
    def parse(cls, raw: "PortfolioWeights | str") -> "PortfolioWeights":
        if isinstance(raw, PortfolioWeights):
            return raw
        normalized = str(raw).strip().lower()
        if normalized in {"eq", "equal", "ew"}:
            return cls.EQUAL
        if normalized in {"mc", "market_cap", "marketcap"}:
            return cls.MARKET_CAP
        raise ValueError(f"Unknown portfolio weighting mode: {raw}")


class EntryPriceMode(str, Enum):
    CLOSE = "close"
    NEXT_OPEN = "next_open"

    @classmethod
    def parse(cls, raw: "EntryPriceMode | str") -> "EntryPriceMode":
        if isinstance(raw, EntryPriceMode):
            return raw
        normalized = str(raw).strip().lower()
        if normalized in {"close", "c"}:
            return cls.CLOSE
        if normalized in {"next_open", "open", "next", "no"}:
            return cls.NEXT_OPEN
        raise ValueError(f"Unknown entry price mode: {raw}")


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
    active_quantiles: Sequence[int] | None = None
    benchmark_symbol: BenchmarkType | str | None = BenchmarkType.KOSPI200
    benchmark_path: Path | str = field(default_factory=_default_benchmark_path)
    allow_partial_buckets: bool = False
    sector_neutral: bool = False
    sector_path: Path | str = field(default_factory=_default_sector_path)
    portfolio_grouping: PortfolioGroupingStrategy | None = None

    portfolio_weighting: PortfolioWeights | str = PortfolioWeights.EQUAL
    weight_data_path: Path | str | None = field(default_factory=_default_weight_data_path)

    entry_price_mode: EntryPriceMode | str = EntryPriceMode.CLOSE
    open_path: Path | str = field(default_factory=_default_open_path)

    apply_trading_costs: bool = False
    buy_cost_bps: float = 0.0
    sell_cost_bps: float = 0.0
    tax_bps: float = 0.0
    entry_lag: int = 0
    tax_bps: float = 0.0
    entry_lag: int = 0
    show_progress: bool = True

    start_date: str | pd.Timestamp | None = None
    end_date: str | pd.Timestamp | None = None

    def __post_init__(self) -> None:
        prepared_scores = self._get_score_files(self.scores_path)
        object.__setattr__(self, "scores_path", prepared_scores)
        object.__setattr__(self, "close_path", self._to_project_path(self.close_path))
        object.__setattr__(self, "output_dir", self._to_project_path(self.output_dir))
        object.__setattr__(self, "constituent_path", self._get_constituent_path())
        weighting = PortfolioWeights.parse(self.portfolio_weighting)
        object.__setattr__(self, "portfolio_weighting", weighting)
        weight_path = self._get_weight_path(self.weight_data_path)
        object.__setattr__(self, "weight_data_path", weight_path)
        if weighting.requires_market_caps and weight_path is None:
            raise ValueError("Market-cap weighting requires 'weight_data_path' to be set.")

        entry_mode = EntryPriceMode.parse(self.entry_price_mode)
        object.__setattr__(self, "entry_price_mode", entry_mode)
        object.__setattr__(self, "open_path", self._to_project_path(self.open_path))
        
        bench_type = BenchmarkType.parse(self.benchmark_symbol)
        object.__setattr__(self, "benchmark_symbol", bench_type)
        object.__setattr__(self, "benchmark_path", self._to_project_path(self.benchmark_path))
        object.__setattr__(self, "sector_path", self._to_project_path(self.sector_path))

        self._validate_numeric_fields()

        if self.start_date is not None:
            object.__setattr__(self, "start_date", pd.Timestamp(self.start_date))
        if self.end_date is not None:
            object.__setattr__(self, "end_date", pd.Timestamp(self.end_date))

    def _to_project_path(self, raw: Path | str) -> Path:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _get_score_files(self, raw_paths: Path | Sequence[Path]) -> Tuple[Path, ...]:
        if isinstance(raw_paths, (str, Path)):
            candidates: Sequence[Path | str] = (raw_paths,)
        else:
            candidates = tuple(raw_paths)
        if not candidates:
            raise ValueError("At least one score file must be provided.")
        return tuple(self._to_project_path(path) for path in candidates)

    def _get_constituent_path(self) -> Path | None:
        universe_name = getattr(self.constituent_universe, "parquet_filename", None) if self.constituent_universe else None
        universe_path = universe_name and DATA_ROOT / universe_name
        selected = self.constituent_path or universe_path
        return selected and self._to_project_path(selected)

    def _get_weight_path(self, raw: Path | str | None) -> Path | None:
        return raw is not None and self._to_project_path(raw) or None

    def grouping_strategy(self) -> PortfolioGroupingStrategy:
        if self.portfolio_grouping is not None:
            return self.portfolio_grouping
            
        base_strategy = QuantileGroupingStrategy(
            quantiles=self.quantiles,
            min_assets=self.min_assets,
            allow_partial=self.allow_partial_buckets,
            enabled_quantiles=self.active_quantiles,
        )
        
        if not self.sector_neutral:
            return base_strategy

        sector_path = Path(self.sector_path)
        if not sector_path.exists():
            raise FileNotFoundError(f"Sector data not found at {sector_path} for sector neutral strategy.")
            
        return SectorNeutralGroupingStrategy(
            sector_panel=pd.read_parquet(sector_path),
            inner_strategy=base_strategy
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
        if self.portfolio_weighting.requires_market_caps:
            if self.weight_data_path is None or not Path(self.weight_data_path).exists():
                raise FileNotFoundError(
                    f"Market cap metric parquet not found: {self.weight_data_path or '<unspecified>'}"
                )
        if self.entry_price_mode == EntryPriceMode.NEXT_OPEN:
            if not prices_in_memory and not Path(self.open_path).exists():
                raise FileNotFoundError(f"Open price parquet not found: {self.open_path}")
        if self.benchmark_symbol is not None:
            if not Path(self.benchmark_path).exists():
                raise FileNotFoundError(f"Benchmark parquet not found: {self.benchmark_path}")

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
            benchmark_symbol=self.benchmark_symbol.value if self.benchmark_symbol else None,
            benchmark_source=self.benchmark_path,
            weight_source=self.weight_data_path if self.portfolio_weighting.requires_market_caps else None,
            open_source=self.open_path if self.entry_price_mode == EntryPriceMode.NEXT_OPEN else None,
            start_date=self.start_date,
            end_date=self.end_date,
        )

    def _validate_numeric_fields(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if self.quantiles < 1:
            raise ValueError("quantiles must be at least 1.")
        if self.min_assets < self.quantiles:
            raise ValueError("min_assets must be >= quantiles to form buckets.")
        for field_name in ("buy_cost_bps", "sell_cost_bps", "tax_bps"):
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.entry_lag < 0:
            raise ValueError("entry_lag must be non-negative.")
