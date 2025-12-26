from __future__ import annotations

import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig, PortfolioWeights, EntryPriceMode
from backtest.costs import ExecutionCostModel
from backtest.data_sources import BacktestDataset
from backtest.grouping import PortfolioGroupingStrategy
from backtest.portfolio import PortfolioTrack, PositionSide
from backtest.config import LongShortMode
from backtest.report import BacktestReport, PortfolioReport


@dataclass(frozen=True)
class RebalanceWindow:
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp


@dataclass(frozen=True)
class WindowPayload:
    window: RebalanceWindow
    scores: pd.Series
    prices: pd.DataFrame
    weights: pd.Series | None = None
    entry_prices: pd.Series | None = None


logger = logging.getLogger(__name__)


class RebalancePlanner:
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
    def schedule(self) -> Sequence[pd.Timestamp]:
        return list(self._schedule)

    @property
    def anchor(self) -> pd.Timestamp:
        return self._schedule[0]

    def windows(self) -> Sequence[RebalanceWindow]:
        return list(self._windows)

    def _build_schedule(self) -> list[pd.Timestamp]:
        freq = self.frequency
        if isinstance(freq, str) and freq.upper() == "M":
            freq = "ME"
        series = pd.Series(index=self._dates, data=self._dates)
        grouped = series.resample(freq).last().dropna()
        # NOTE: Use actual trading dates from the values (avoid calendar labels like month-end).
        schedule = [pd.Timestamp(ts) for ts in grouped.to_numpy()]
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
            if entry is None or entry >= exit_date:
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
            loc = int(self._dates.searchsorted(signal, side="right")) - 1
        if isinstance(loc, slice):
            loc = loc.start
        idx = loc + int(self.entry_lag)
        if idx >= len(self._dates):
            return None
        return pd.Timestamp(self._dates[idx])


class PerformanceCalculator:
    def __init__(self, frequency: str) -> None:
        self.frequency = frequency

    def summarize(self, equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
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

        equity = equity.sort_index()
        daily_returns = equity.pct_change().dropna()
        start_equity = float(equity.iloc[0])
        final_equity = float(equity.iloc[-1])
        total_return = 0.0 if start_equity == 0 else (final_equity / start_equity) - 1.0

        periods_per_year = self._periods_per_year(equity.index)
        periods = len(daily_returns)
        years = periods / periods_per_year if periods_per_year > 0 else 0.0
        cagr = 0.0 if years <= 0 or start_equity <= 0 else (final_equity / start_equity) ** (1 / years) - 1.0

        vol = daily_returns.std(ddof=0) * (periods_per_year**0.5) if not daily_returns.empty else 0.0
        avg_daily = daily_returns.mean() if not daily_returns.empty else 0.0
        avg_return = avg_daily * periods_per_year
        sharpe = 0.0 if vol == 0 else avg_return / vol
        period_return = float(returns.mean()) if not returns.empty else 0.0
        max_dd = self._max_drawdown(equity)
        avg_period_return = float(avg_daily)
        win_rate = float((daily_returns > 0).mean()) if not daily_returns.empty else 0.0
        pnl = final_equity - start_equity

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_equity": float(final_equity),
            "pnl": float(pnl),
            "avg_period_return": float(period_return),
            "win_rate": float(win_rate),
        }

    def _max_drawdown(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        return float(drawdown.min())

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
        freq = self.frequency.upper()
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


class BacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        dataset: BacktestDataset,
        grouping: PortfolioGroupingStrategy | None = None,
        cost_model: ExecutionCostModel | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.grouping = grouping or config.grouping_strategy()
        self.cost_model = cost_model or config.cost_model()
        self.weighting = config.portfolio_weighting
        self._calculator = PerformanceCalculator(config.rebalance_frequency)
        self._validate_weight_support()
        self._short_groups = self._short_group_ids()
        self._group_sides: dict[str, PositionSide] = {}

    def run(self) -> BacktestReport:
        return self._run(swap_leg_reports=True)

    def run_raw(self) -> BacktestReport:
        """Run backtest without swapping DN legs to no-DN curves."""
        return self._run(swap_leg_reports=False)

    def _run(self, *, swap_leg_reports: bool) -> BacktestReport:
        timeline = self._build_timeline()
        groups = self.grouping.groups()
        label_map = {group.identifier: group.label for group in groups}
        group_sides = {group.identifier: self._group_side(group.identifier) for group in groups}
        long_ids = [gid for gid, side in group_sides.items() if side is PositionSide.LONG]
        short_ids = [gid for gid, side in group_sides.items() if side is PositionSide.SHORT]
        long_cap, short_cap = self._initial_side_capitals(long_ids, short_ids)
        portfolios = {
            group.identifier: PortfolioTrack(
                group_id=group.identifier,
                starting_capital=self._starting_capital(
                    group.identifier,
                    group_sides[group.identifier],
                    long_ids,
                    short_ids,
                    long_cap,
                    short_cap,
                ),
                cost_model=self.cost_model,
                side=group_sides[group.identifier],
            )
            for group in groups
        }
        self._group_sides = {gid: portfolio.side for gid, portfolio in portfolios.items()}
        for portfolio in portfolios.values():
            portfolio.mark_initial(timeline.anchor)

        for payload in self._iter_windows(timeline.windows()):
            self._rebalance_dollar_neutral(portfolios)
            allocation = self.grouping.allocate(payload.scores)
            note = allocation.message if allocation.skipped else None
            for group_id, portfolio in portfolios.items():
                tickers = allocation.tickers_for(group_id)
                weights = self._portfolio_weights(group_id, payload.window.signal_date, tickers, payload.weights)
                portfolio.rebalance(
                    enter_date=payload.window.entry_date,
                    exit_date=payload.window.exit_date,
                    tickers=tickers,
                    price_slice=payload.prices,
                    note=note,
                    weights=weights,
                    entry_prices=payload.entry_prices,
                )

        reports = {gid: self._build_group_report(portfolio) for gid, portfolio in portfolios.items()}
        self._add_net_report(reports, label_map)
        if swap_leg_reports:
            # NOTE: In dollar-neutral mode, keep net as DN but show leg curves as standalone (no-DN) P&L.
            self._swap_leg_reports(reports)
        bench_equity = self._benchmark_equity_series()
        return BacktestReport(config=self.config, groups=reports, bench_equity=bench_equity, labels=label_map)

    def _initial_side_capitals(self, long_ids: list[str], short_ids: list[str]) -> tuple[float, float]:
        if not self._dollar_neutral_enabled() or not long_ids or not short_ids:
            return self.config.initial_capital, self.config.initial_capital
        half = float(self.config.initial_capital) / 2.0
        return half, half

    def _starting_capital(
        self,
        group_id: str,
        side: PositionSide,
        long_ids: list[str],
        short_ids: list[str],
        long_cap: float,
        short_cap: float,
    ) -> float:
        if not self._dollar_neutral_enabled():
            return float(self.config.initial_capital)
        if side is PositionSide.LONG and long_ids:
            return long_cap / len(long_ids)
        if side is PositionSide.SHORT and short_ids:
            return short_cap / len(short_ids)
        return float(self.config.initial_capital)

    def _rebalance_dollar_neutral(self, portfolios: dict[str, PortfolioTrack]) -> None:
        if not self._dollar_neutral_enabled():
            return
        long_ids = [gid for gid, side in self._group_sides.items() if side is PositionSide.LONG]
        short_ids = [gid for gid, side in self._group_sides.items() if side is PositionSide.SHORT]
        if not long_ids or not short_ids:
            return
        long_total = sum(portfolios[gid].capital for gid in long_ids)
        short_total = sum(portfolios[gid].capital for gid in short_ids)
        total = long_total + short_total
        if total <= 0:
            return
        target = total / 2.0
        self._scale_side_capital(portfolios, long_ids, target, long_total)
        self._scale_side_capital(portfolios, short_ids, target, short_total)

    def _scale_side_capital(
        self,
        portfolios: dict[str, PortfolioTrack],
        side_ids: list[str],
        target: float,
        side_total: float,
    ) -> None:
        if not side_ids:
            return
        if side_total <= 0:
            equal = target / len(side_ids)
            for gid in side_ids:
                portfolios[gid].capital = equal
            return
        for gid in side_ids:
            weight = portfolios[gid].capital / side_total
            portfolios[gid].capital = target * weight

    def _dollar_neutral_enabled(self) -> bool:
        return (
            self.config.long_short_mode is LongShortMode.NET
            and bool(self.config.dollar_neutral_net)
        )

    def _build_group_report(self, portfolio: PortfolioTrack) -> PortfolioReport:
        equity = portfolio.equity_series()
        returns = portfolio.return_series()
        stats = self._calculator.summarize(equity, returns)
        return PortfolioReport(
            group_id=portfolio.group_id,
            equity_curve=equity,
            period_returns=returns,
            trades=portfolio.trades,
            stats=stats,
        )

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

    def _build_timeline(self) -> RebalancePlanner:
        offset = self._rebalance_offset()
        return RebalancePlanner(self.dataset.dates, offset, entry_lag=self.config.entry_lag)

    def _iter_windows(self, windows: Sequence[RebalanceWindow]) -> Iterable[WindowPayload]:
        iterable = windows if not self.config.show_progress else tqdm(windows, desc="Running backtest", unit="window", leave=False)
        for window in iterable:
            yield self._window_payload(window)

    def _window_payload(self, window: RebalanceWindow) -> WindowPayload:
        effective_date = window.entry_date
        try:
            scores_row = self.dataset.scores.loc[effective_date]
        except KeyError:
            # Fallback: use the last available scores on or before the entry date
            scores_before = self.dataset.scores.loc[: effective_date]
            if scores_before.empty:
                raise KeyError(f"Missing scores for entry date {effective_date}.")
            scores_row = scores_before.iloc[-1]
        price_slice = self.dataset.prices.loc[window.entry_date : window.exit_date]
        if price_slice.empty:
            raise ValueError(
                f"Price slice empty for window {window.entry_date} -> {window.exit_date}. "
                "Ensure price data covers the full rebalance interval."
            )
        weights_row = None
        if getattr(self.dataset, "weights", None) is not None:
            try:
                weights_row = self.dataset.weights.loc[effective_date]
            except KeyError:
                weights_before = self.dataset.weights.loc[: effective_date]
                weights_row = weights_before.iloc[-1] if not weights_before.empty else None

        entry_prices = None
        if self.config.entry_price_mode == EntryPriceMode.NEXT_OPEN:
            if getattr(self.dataset, "open_prices", None) is None:
                raise ValueError("EntryPriceMode.NEXT_OPEN requires open_prices in dataset.")
            try:
                entry_prices = self.dataset.open_prices.loc[effective_date]
            except KeyError:
                raise KeyError(f"Missing open prices for entry date {effective_date}")

        return WindowPayload(
            window=window,
            scores=scores_row,
            prices=price_slice,
            weights=weights_row,
            entry_prices=entry_prices,
        )

    def _portfolio_weights(
        self,
        group_id: str,
        signal_date: pd.Timestamp,
        tickers: Sequence[str],
        weight_row: pd.Series | None,
    ) -> pd.Series | None:
        if not tickers:
            return None
        if self.weighting is not PortfolioWeights.MARKET_CAP:
            return None
        if weight_row is None:
            self._log_weight_fallback(group_id, signal_date, "missing weight row for signal date")
            return None
        working = pd.Series(weight_row, dtype=float).reindex(tickers)
        if working.isna().any():
            self._log_weight_fallback(group_id, signal_date, "incomplete market-cap data for selected tickers")
            return None
        working = working.clip(lower=0.0)
        total = working.sum()
        if total <= 0:
            self._log_weight_fallback(group_id, signal_date, "non-positive market-cap sum after clipping")
            return None
        return working / total

    def _log_weight_fallback(self, group_id: str, signal_date: pd.Timestamp, reason: str) -> None:
        logger.warning(
            "Falling back to equal weights for %s on %s: %s",
            group_id,
            pd.Timestamp(signal_date).date(),
            reason,
        )

    def _validate_weight_support(self) -> None:
        if self.weighting.requires_market_caps and getattr(self.dataset, "weights", None) is None:
            raise ValueError("Market-cap weighting requested but dataset does not supply weights.")

    def _short_group_ids(self) -> set[str]:
        if self.config.long_short_mode is LongShortMode.OFF:
            return set()
        if not self.config.short_quantiles:
            return {"q1"}
        return {f"q{int(idx) + 1}" for idx in self.config.short_quantiles}

    def _group_side(self, group_id: str) -> PositionSide:
        return PositionSide.SHORT if group_id.lower() in self._short_groups else PositionSide.LONG

    def _rebalance_offset(self):
        freq = self.config.rebalance_frequency.upper()
        offsets = {
            "M": pd.offsets.MonthEnd(),
            "ME": pd.offsets.MonthEnd(),
            "Q": pd.offsets.QuarterEnd(),
            "QE": pd.offsets.QuarterEnd(),
        }
        return offsets.get(freq, freq)

    def _add_net_report(self, reports: dict[str, PortfolioReport], labels: dict[str, str]) -> None:
        if self.config.long_short_mode is not LongShortMode.NET:
            return
        longs = [gid for gid, side in self._group_sides.items() if side is PositionSide.LONG and gid in reports]
        shorts = [gid for gid, side in self._group_sides.items() if side is PositionSide.SHORT and gid in reports]
        if not longs or not shorts:
            return
        long_id = longs[0]
        short_id = shorts[0]
        long_eq = reports[long_id].equity_curve
        short_eq = reports[short_id].equity_curve
        all_dates = pd.Index(sorted(set(long_eq.index) | set(short_eq.index)))
        long_eq = long_eq.reindex(all_dates).ffill()
        short_eq = short_eq.reindex(all_dates).ffill()
        net_equity = long_eq.add(short_eq, fill_value=0.0)
        net_equity = net_equity.sort_index()
        if net_equity.empty:
            return
        net_returns = net_equity.pct_change().fillna(0.0)
        stats = self._calculator.summarize(net_equity, net_returns)
        reports["net"] = PortfolioReport(
            group_id="net",
            equity_curve=net_equity,
            period_returns=net_returns,
            trades=tuple(),
            stats=stats,
        )
        labels["net"] = "net"

    def _swap_leg_reports(self, reports: dict[str, PortfolioReport]) -> None:
        if not self._dollar_neutral_enabled():
            return
        baseline_config = self.config.with_overrides(
            dollar_neutral_net=False,
            long_short_mode=LongShortMode.LEGS,
        )
        baseline_engine = BacktestEngine(
            config=baseline_config,
            dataset=self.dataset,
            grouping=self.grouping,
            cost_model=self.cost_model,
        )
        baseline_report = baseline_engine.run()
        for group_id, baseline in baseline_report.groups.items():
            if group_id == "net":
                continue
            if group_id not in reports:
                continue
            reports[group_id] = baseline
