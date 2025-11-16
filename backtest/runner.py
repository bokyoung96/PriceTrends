from __future__ import annotations

import sys
from pathlib import Path
import logging
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig
from backtest.data_sources import BacktestDataset, BacktestDatasetBuilder
from backtest.engine import BacktestEngine
from backtest.quantiles import BucketAllocator
from backtest.report import BacktestReport


logger = logging.getLogger(__name__)


class Backtester:
    """Object-oriented facade for running a PriceTrends backtest."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._base_config = config or BacktestConfig()
        self._dataset: BacktestDataset | None = None
        self._report: BacktestReport | None = None

    def run(
        self,
        *,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        **overrides: Any,
    ) -> BacktestReport:
        active_config = self._config_with(overrides)
        dataset = self._build_dataset(active_config, scores=scores, prices=prices)
        self._dataset = dataset
        self._report = self._run_engine(config=active_config, dataset=dataset)
        return self._report

    def latest_report(self) -> BacktestReport:
        if self._report is None:
            raise RuntimeError("Backtest has not been run yet.")
        return self._report

    def save_results(self, output_dir=None, filename: str | None = None):
        return self.latest_report().save(output_dir=output_dir, filename=filename)

    def summary(self) -> str:
        return self.latest_report().render_summary()

    @property
    def score_df(self) -> pd.DataFrame:
        dataset = self._require_dataset()
        return dataset.scores.copy()

    @property
    def price_df(self) -> pd.DataFrame:
        dataset = self._require_dataset()
        return dataset.prices.copy()

    @property
    def hit_rate_df(self) -> pd.DataFrame:
        summary = self.latest_report().summary_table()
        win = summary["win_rate"].rename("hit_rate")
        return win.to_frame()

    @property
    def cumulative_returns(self) -> pd.Series:
        summary = self.latest_report().summary_table()
        return summary["total_return"]

    @property
    def equity_df(self) -> pd.DataFrame:
        return self.latest_report().equity_frame()

    @property
    def period_return_df(self) -> pd.DataFrame:
        return self.latest_report().return_frame()

    @property
    def daily_return_df(self) -> pd.DataFrame:
        dataset = self._require_dataset()
        equity = self.equity_df
        expanded = equity.reindex(dataset.dates).ffill()
        return expanded.pct_change().dropna()

    def _config_with(self, overrides: dict[str, Any]) -> BacktestConfig:
        if overrides:
            return self._base_config.with_overrides(**overrides)
        return self._base_config

    def _build_dataset(
        self,
        config: BacktestConfig,
        *,
        scores: pd.DataFrame | None,
        prices: pd.DataFrame | None,
    ) -> BacktestDataset:
        config.ensure_io_paths(
            scores_in_memory=scores is not None,
            prices_in_memory=prices is not None,
        )
        builder = BacktestDatasetBuilder(
            scores_source=scores if scores is not None else config.scores_path,
            close_source=prices if prices is not None else config.close_path,
        )
        return builder.build()

    def _run_engine(self, config: BacktestConfig, dataset: BacktestDataset) -> BacktestReport:
        assigner = BucketAllocator(
            quantiles=config.quantiles,
            min_assets=config.min_assets,
            allow_partial=config.allow_partial_buckets,
        )
        engine = BacktestEngine(config=config, dataset=dataset, assigner=assigner)
        return engine.run()

    def _require_dataset(self) -> BacktestDataset:
        if self._dataset is None:
            raise RuntimeError("Backtest has not been executed yet.")
        return self._dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    bt = Backtester()
    result = bt.run()
    logger.info("\n%s", result.summary_table())
    output_path = result.save()
    logger.info("Saved report to %s", output_path)
