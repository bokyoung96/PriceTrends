from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .config import BacktestConfig
from .costs import ExecutionCostModel
from .data_sources import BacktestDataset
from .portfolio import QuantilePortfolio
from .quantiles import QuantileAssigner
from .report import BacktestReport, QuantileReport


class BacktestEngine:
    """Coordinates data, quantile assignment, and portfolio evolution."""

    def __init__(self, config: BacktestConfig, dataset: BacktestDataset, assigner: QuantileAssigner) -> None:
        self.config = config
        self.dataset = dataset
        self.assigner = assigner

    def run(self) -> BacktestReport:
        rebalance_dates = self._rebalance_schedule(self.dataset.dates)
        quantile_ids = self.config.quantile_ids()
        cost_model = ExecutionCostModel(
            enabled=self.config.apply_trading_costs,
            buy_bps=self.config.buy_cost_bps,
            sell_bps=self.config.sell_cost_bps,
            tax_bps=self.config.tax_bps,
        )
        portfolios = {
            qid: QuantilePortfolio(
                quantile_id=qid,
                starting_capital=self.config.initial_capital,
                cost_model=cost_model,
            )
            for qid in quantile_ids
        }

        for portfolio in portfolios.values():
            portfolio.mark_initial(rebalance_dates[0])

        for start, end in zip(rebalance_dates[:-1], rebalance_dates[1:]):
            scores_row = self.dataset.scores.loc[start]
            bucket_view = self.assigner.assign(scores_row)
            entry_prices = self.dataset.prices.loc[start]
            exit_prices = self.dataset.prices.loc[end]

            for qid, portfolio in portfolios.items():
                tickers = bucket_view.tickers_for(qid)
                note = bucket_view.reason if bucket_view.skipped else None
                portfolio.rebalance(
                    enter_date=start,
                    exit_date=end,
                    tickers=tickers,
                    entry_prices=entry_prices,
                    exit_prices=exit_prices,
                    note=note,
                )

        quantile_reports = {
            qid: self._build_quantile_report(portfolio) for qid, portfolio in portfolios.items()
        }
        return BacktestReport(config=self.config, quantiles=quantile_reports)

    def _rebalance_schedule(self, index: pd.DatetimeIndex) -> List[pd.Timestamp]:
        series = pd.Series(index=index, data=index)
        grouped = series.resample(self._grouper_frequency()).first().dropna()
        schedule: List[pd.Timestamp] = [pd.Timestamp(ts) for ts in grouped]

        if not schedule:
            raise ValueError("Rebalance schedule is empty. Check frequency or data coverage.")

        last_date = index[-1]
        if schedule[-1] != last_date:
            schedule.append(last_date)

        schedule = sorted(set(schedule))
        if len(schedule) < 2:
            raise ValueError("Need at least two rebalance timestamps.")
        return schedule

    def _build_quantile_report(self, portfolio: QuantilePortfolio) -> QuantileReport:
        equity = portfolio.equity_series()
        returns = portfolio.return_series()
        stats = self._compute_stats(equity, returns)
        return QuantileReport(
            quantile_id=portfolio.quantile_id,
            equity_curve=equity,
            period_returns=returns,
            trades=portfolio.trades,
            stats=stats,
        )

    def _compute_stats(self, equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
        if equity.empty:
            return {"total_return": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}

        total_return = 0.0
        cagr = 0.0
        if equity.iloc[0] != 0:
            total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

        periods_per_year = self._periods_per_year()
        periods = len(returns)
        years = periods / periods_per_year if periods_per_year > 0 else 0.0
        if years > 0 and equity.iloc[0] > 0:
            cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0

        vol = returns.std(ddof=0) * (periods_per_year**0.5) if not returns.empty else 0.0
        avg_return = returns.mean() * periods_per_year if not returns.empty else 0.0
        sharpe = 0.0 if vol == 0 else avg_return / vol
        max_dd = self._max_drawdown(equity)

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        return float(drawdown.min())

    def _periods_per_year(self) -> float:
        freq = self.config.rebalance_frequency.upper()
        mapping = {
            "D": 252,
            "B": 252,
            "W": 52,
            "M": 12,
            "MS": 12,
            "BM": 12,
            "BMS": 12,
            "Q": 4,
            "QS": 4,
        }
        return float(mapping.get(freq, 12))

    def _grouper_frequency(self) -> str:
        freq = self.config.rebalance_frequency.upper()
        replacements = {
            "M": "ME",
            "Q": "QE",
        }
        return replacements.get(freq, freq)
