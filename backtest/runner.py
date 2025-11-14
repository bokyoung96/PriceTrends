from __future__ import annotations

from typing import Any

from .config import BacktestConfig
from .data_sources import BacktestDatasetBuilder
from .engine import BacktestEngine
from .quantiles import QuantileAssigner
from .report import BacktestReport


def run_backtest(config: BacktestConfig | None = None, **overrides: Any) -> BacktestReport:
    """Convenience helper that wires the builder, engine, and quantile logic together."""

    active_config = config or BacktestConfig()
    if overrides:
        active_config = active_config.with_overrides(**overrides)

    active_config.ensure_io_paths()

    dataset = BacktestDatasetBuilder(
        scores_path=active_config.scores_path,
        close_path=active_config.close_path,
    ).build()

    assigner = QuantileAssigner(
        quantiles=active_config.quantiles,
        min_assets=active_config.min_assets,
        allow_partial=active_config.allow_partial_buckets,
    )

    engine = BacktestEngine(
        config=active_config,
        dataset=dataset,
        assigner=assigner,
    )
    return engine.run()
