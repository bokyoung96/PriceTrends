from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig
from backtest.costs import ExecutionCostModel
from backtest.data_sources import BacktestDataset
from backtest.portfolio import BucketPortfolio
from backtest.quantiles import BucketAllocator
from backtest.report import BucketReport, SimulationReport
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RebalanceWindow:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp


class RebalanceTimeline:
    def __init__(self, dates: pd.DatetimeIndex, frequency: str, entry_lag: int = 1) -> None:
        if dates.empty:
            raise ValueError("Dataset index is empty; cannot run backtest.")
        if entry_lag < 0:
            raise ValueError("entry_lag must be non-negative.")
        self._dates = pd.DatetimeIndex(dates).sort_values().unique()
        self.frequency = frequency
        self.entry_lag = entry_lag
        self._schedule = self._build_schedule()
        self._windows = self._build_windows()

    @property
    def schedule(self) -> list[pd.Timestamp]:
        return list(self._schedule)

    @property
    def anchor(self) -> pd.Timestamp:
        return self._schedule[0]

    def windows(self) -> list[RebalanceWindow]:
        return list(self._windows)

    def _build_schedule(self) -> list[pd.Timestamp]:
        series = pd.Series(index=self._dates, data=self._dates)
        grouped = series.resample(self.frequency).first().dropna()
        schedule = [pd.Timestamp(ts) for ts in grouped]

        last_date = self._dates[-1]
        if not schedule or schedule[-1] != last_date:
            schedule.append(last_date)

        schedule = sorted(set(schedule))
        if len(schedule) < 2:
            raise ValueError("Need at least two timestamps to build a rebalance schedule.")
        return schedule

    def _build_windows(self) -> list[RebalanceWindow]:
        windows: list[RebalanceWindow] = []
        for signal, exit_date in zip(self._schedule[:-1], self._schedule[1:]):
            entry = self._entry_at_lag(signal)
            if entry is None:
                continue
            if entry >= exit_date:
                continue
            windows.append(RebalanceWindow(signal_date=signal, entry_date=entry, exit_date=exit_date))
        if not windows:
            raise ValueError(
                "No tradable windows remained after applying the entry lag. "
                "Ensure each signal date has sufficient subsequent price observations."
            )
        return windows

    def _entry_at_lag(self, signal: pd.Timestamp) -> pd.Timestamp | None:
        try:
            loc = self._dates.get_loc(signal)
        except KeyError:
            loc = int(self._dates.searchsorted(signal, side="left"))
        if isinstance(loc, slice):
            loc = loc.start
        idx = loc + int(self.entry_lag)
        if idx >= len(self._dates):
            return None
        return pd.Timestamp(self._dates[idx])


class BacktestEngine:
    """Coordinates data, quantile assignment, and portfolio evolution."""

    def __init__(self, config: BacktestConfig, dataset: BacktestDataset, assigner: BucketAllocator) -> None:
        self.config = config
        self.dataset = dataset
        self.assigner = assigner

    def run(self) -> SimulationReport:
        timeline = self._build_timeline()
        windows = timeline.windows()
        quantile_ids = self.config.quantile_ids()
        cost_model = ExecutionCostModel(
            enabled=self.config.apply_trading_costs,
            buy_bps=self.config.buy_cost_bps,
            sell_bps=self.config.sell_cost_bps,
            tax_bps=self.config.tax_bps,
        )
        portfolios = {
            qid: BucketPortfolio(
                bucket_id=qid,
                starting_capital=self.config.initial_capital,
                cost_model=cost_model,
            )
            for qid in quantile_ids
        }

        for portfolio in portfolios.values():
            portfolio.mark_initial(timeline.anchor)

        for window in self._iter_windows(windows):
            scores_row, price_slice = self._window_view(window)
            bucket_view = self.assigner.assign(scores_row)

            for qid, portfolio in portfolios.items():
                tickers = bucket_view.tickers_for(qid)
                note = bucket_view.reason if bucket_view.skipped else None
                portfolio.rebalance(
                    enter_date=window.entry_date,
                    exit_date=window.exit_date,
                    tickers=tickers,
                    price_slice=price_slice,
                    note=note,
                )

        quantile_reports = {
            qid: self._build_quantile_report(portfolio) for qid, portfolio in portfolios.items()
        }
        bench_equity = self._benchmark_equity_series()
        return SimulationReport(config=self.config, quantiles=quantile_reports, bench_equity=bench_equity)

    def _build_quantile_report(self, portfolio: BucketPortfolio) -> BucketReport:
        equity = portfolio.equity_series()
        returns = equity.pct_change().dropna()
        stats = self._compute_stats(equity, returns)
        return BucketReport(
            bucket_id=portfolio.bucket_id,
            equity_curve=equity,
            period_returns=portfolio.return_series(),
            trades=portfolio.trades,
            stats=stats,
        )

    def _compute_stats(self, equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
        if equity.empty:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "final_equity": 0.0,
                "pnl": 0.0,
                "avg_period_return": 0.0,
                "win_rate": 0.0,
            }

        start_equity = float(equity.iloc[0])
        final_equity = float(equity.iloc[-1])

        total_return = 0.0
        cagr = 0.0
        if start_equity != 0:
            total_return = (final_equity / start_equity) - 1.0

        periods_per_year = self._periods_per_year(equity.index if isinstance(equity, pd.Series) else None)
        periods = len(returns)
        years = periods / periods_per_year if periods_per_year > 0 else 0.0
        if years > 0 and start_equity > 0:
            cagr = (final_equity / start_equity) ** (1 / years) - 1.0

        vol = returns.std(ddof=0) * (periods_per_year**0.5) if not returns.empty else 0.0
        avg_return = returns.mean() * periods_per_year if not returns.empty else 0.0
        sharpe = 0.0 if vol == 0 else avg_return / vol
        max_dd = self._max_drawdown(equity)
        avg_period_return = float(returns.mean()) if not returns.empty else 0.0
        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
        pnl = final_equity - start_equity

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_equity": float(final_equity),
            "pnl": float(pnl),
            "avg_period_return": float(avg_period_return),
            "win_rate": float(win_rate),
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        return float(drawdown.min())

    def _benchmark_equity_series(self) -> pd.Series | None:
        bench = getattr(self.dataset, "bench", None)
        if bench is None or bench.empty:
            return None
        aligned = bench.reindex(self.dataset.dates).ffill().dropna()
        if aligned.empty:
            return None
        first = aligned.iloc[0]
        if first == 0:
            return None
        scaled = aligned / first * self.config.initial_capital
        return scaled

    def _build_timeline(self) -> RebalanceTimeline:
        offset = self._rebalance_offset()
        return RebalanceTimeline(self.dataset.dates, offset, entry_lag=self.config.entry_lag)

    def _iter_windows(self, windows: Sequence[RebalanceWindow]) -> Iterable[RebalanceWindow]:
        if not windows:
            return iter(())
        if self.config.show_progress:
            return tqdm(windows, desc="Running backtest", unit="window", leave=False)
        return iter(windows)

    def _window_view(self, window: RebalanceWindow) -> Tuple[pd.Series, pd.DataFrame]:
        try:
            scores_row = self.dataset.scores.loc[window.signal_date]
        except KeyError as exc:  # noqa: BLE001
            raise KeyError(f"Missing scores for signal date {window.signal_date}.") from exc
        price_slice = self.dataset.prices.loc[window.entry_date : window.exit_date]
        if price_slice.empty:
            raise ValueError(
                f"Price slice empty for window {window.entry_date} -> {window.exit_date}. "
                "Ensure price data covers the full rebalance interval."
            )
        return scores_row, price_slice

    def _periods_per_year(self, index: pd.Index | None = None) -> float:
        default = self._default_periods_per_year()
        if index is None or len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
            return default
        diffs = pd.Series(index).diff().dropna()
        if diffs.empty:
            return default
        avg_days = diffs.dt.total_seconds().mean() / 86_400
        if avg_days <= 0:
            return default
        if avg_days <= 3:
            return 252.0
        return max(default, 365.25 / avg_days)

    def _default_periods_per_year(self) -> float:
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

    def _rebalance_offset(self):
        freq = self.config.rebalance_frequency.upper()
        offsets = {
            "M": pd.offsets.MonthEnd(),
            "ME": pd.offsets.MonthEnd(),
            "Q": pd.offsets.QuarterEnd(),
            "QE": pd.offsets.QuarterEnd(),
        }
        return offsets.get(freq, freq)
