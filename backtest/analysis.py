from __future__ import annotations

import sys
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.portfolio import PositionLedgerEntry, TradeRecord
from backtest.report import BacktestReport
from backtest.runner import Backtester
from backtest.main import run_single_example, run_single_fusion_example
from backtest.data_sources import BacktestDataset


@dataclass
class LatestPortfolioSnapshot:
    group_id: str
    enter_date: pd.Timestamp
    exit_date: pd.Timestamp
    positions: pd.DataFrame


class PortfolioAnalyzer:    
    def __init__(self, report: BacktestReport) -> None:
        self.report = report

    @property
    def group_ids(self) -> List[str]:
        return sorted(self.report.groups.keys())

    def latest_trade(self, group_id: str) -> Optional[TradeRecord]:
        portfolio_report = self.report.groups.get(group_id)
        if portfolio_report is None or not portfolio_report.trades:
            return None
        return portfolio_report.trades[-1]

    def latest_snapshot(self, group_id: str) -> Optional[LatestPortfolioSnapshot]:
        trade = self.latest_trade(group_id)
        if trade is None or not trade.positions:
            return None
        df = self._positions_frame(trade.positions, trade.capital_in)
        return LatestPortfolioSnapshot(
            group_id=group_id,
            enter_date=trade.enter_date,
            exit_date=trade.exit_date,
            positions=df,
        )

    def latest_snapshots(
        self,
        groups: Optional[Sequence[str]] = None,
    ) -> Dict[str, LatestPortfolioSnapshot]:
        target_groups = list(groups) if groups is not None else self.group_ids
        snapshots: Dict[str, LatestPortfolioSnapshot] = {}
        for gid in target_groups:
            snapshot = self.latest_snapshot(gid)
            if snapshot is not None:
                snapshots[gid] = snapshot
        return snapshots

    def latest_long_tickers(
        self,
        group_id: str,
        *,
        min_weight: float = 0.0,
    ) -> List[str]:
        snapshot = self.latest_snapshot(group_id)
        if snapshot is None:
            return []
        df = snapshot.positions
        if min_weight > 0:
            df = df[df["weight"] >= min_weight]
        return df["ticker"].tolist()

    def position_history(
        self,
        group_id: str,
        tickers: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        portfolio_report = self.report.groups.get(group_id)
        if portfolio_report is None or not portfolio_report.trades:
            return pd.DataFrame()
        rows: List[dict] = []
        tickers_set = set(str(t) for t in tickers) if tickers is not None else None
        for trade in portfolio_report.trades:
            for pos in trade.positions:
                if tickers_set is not None and pos.ticker not in tickers_set:
                    continue
                rows.append(
                    {
                        "group_id": group_id,
                        "enter_date": trade.enter_date,
                        "exit_date": trade.exit_date,
                        "ticker": pos.ticker,
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.exit_price,
                        "entry_value": pos.entry_value,
                        "exit_value": pos.exit_value,
                        "period_return": trade.period_return,
                        "note": trade.note,
                    }
                )
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.sort_values(["enter_date", "exit_date", "ticker"], inplace=True)
        return df

    def _positions_frame(
        self,
        positions: Sequence[PositionLedgerEntry],
        capital_in: float,
    ) -> pd.DataFrame:
        rows: List[dict] = []
        for pos in positions:
            weight = 0.0 if capital_in == 0 else float(pos.entry_value) / float(capital_in)
            rows.append(
                {
                    "ticker": pos.ticker,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "entry_value": pos.entry_value,
                    "weight": weight,
                }
            )
        df = pd.DataFrame(rows)
        df.sort_values("weight", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


class MarketAnalyzer:
    def __init__(self, dataset: BacktestDataset | None) -> None:
        self.dataset = dataset

    def get_price_history(self, tickers: Sequence[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        if self.dataset is None:
            print("Warning: Dataset not available in analyzer. Cannot retrieve price history.")
            return pd.DataFrame()
        
        prices = self.dataset.prices
        if prices is None or prices.empty:
            return pd.DataFrame()

        if tickers is not None:
            available_tickers = [t for t in tickers if t in prices.columns]
            if not available_tickers:
                return pd.DataFrame()
            df = prices[available_tickers].copy()
        else:
            df = prices.copy()

        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        return df

    def get_score_history(self, tickers: Sequence[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        if self.dataset is None:
            print("Warning: Dataset not available in analyzer. Cannot retrieve score history.")
            return pd.DataFrame()
        
        scores = self.dataset.scores
        if scores is None or scores.empty:
            return pd.DataFrame()

        if tickers is not None:
            available_tickers = [t for t in tickers if t in scores.columns]
            if not available_tickers:
                return pd.DataFrame()
            df = scores[available_tickers].copy()
        else:
            df = scores.copy()
    
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        return df


class PerformanceAnalyzer:
    def __init__(self, report: BacktestReport) -> None:
        self.report = report

    def get_monthly_returns(self, group_id: str) -> pd.DataFrame:
        portfolio_report = self.report.groups.get(group_id)
        if portfolio_report is None:
            return pd.DataFrame()
        return self._calculate_monthly_returns(portfolio_report.equity_curve)

    def get_benchmark_monthly_returns(self) -> pd.DataFrame:
        if self.report.bench_equity is None or self.report.bench_equity.empty:
            return pd.DataFrame()
        return self._calculate_monthly_returns(self.report.bench_equity)

    def get_summary_stats(self) -> pd.DataFrame:
        return self.report.summary_table()

    def _calculate_monthly_returns(self, equity: pd.Series) -> pd.DataFrame:
        if equity is None or equity.empty:
            return pd.DataFrame()
        
        series = equity.dropna()
        if series.empty:
            return pd.DataFrame()
            
        monthly_nav = series.resample("ME").last().dropna()
        monthly_returns = monthly_nav.pct_change().dropna()
        
        if monthly_returns.empty:
            return pd.DataFrame()

        df = pd.DataFrame({"value": monthly_returns.values}, index=monthly_returns.index)
        df["year"] = df.index.year
        df["month"] = df.index.month
        
        pivot = df.pivot(index="year", columns="month", values="value").sort_index()
        
        all_months = range(1, 13)
        pivot = pivot.reindex(columns=all_months)
        
        pivot.columns = [calendar.month_abbr[m] for m in pivot.columns]
        return pivot


class TradeVerifier:
    def __init__(self, report: BacktestReport, dataset: BacktestDataset | None) -> None:
        self.report = report
        self.dataset = dataset

    def verify_trade_execution(self, group_id: str, tolerance: float = 1e-6) -> pd.DataFrame:
        if self.dataset is None:
            print("Warning: Dataset not available in analyzer. Cannot verify trade execution.")
            return pd.DataFrame()
            
        portfolio_report = self.report.groups.get(group_id)
        if portfolio_report is None or not portfolio_report.trades:
            return pd.DataFrame()
            
        discrepancies = []
        
        for trade in portfolio_report.trades:
            enter_date = trade.enter_date
            try:
                if enter_date in self.dataset.prices.index:
                    market_prices = self.dataset.prices.loc[enter_date]
                    
                    for pos in trade.positions:
                        ticker = pos.ticker
                        if ticker in market_prices:
                            market_price = market_prices[ticker]
                            diff = abs(pos.entry_price - market_price)
                            if diff > tolerance:
                                discrepancies.append({
                                    "date": enter_date,
                                    "type": "ENTRY",
                                    "ticker": ticker,
                                    "trade_price": pos.entry_price,
                                    "market_price": market_price,
                                    "diff": diff
                                })
            except KeyError:
                pass

            exit_date = trade.exit_date
            try:
                if exit_date in self.dataset.prices.index:
                    market_prices = self.dataset.prices.loc[exit_date]
                    
                    for pos in trade.positions:
                        ticker = pos.ticker
                        if ticker in market_prices:
                            market_price = market_prices[ticker]
                            diff = abs(pos.exit_price - market_price)
                            if diff > tolerance:
                                discrepancies.append({
                                    "date": exit_date,
                                    "type": "EXIT",
                                    "ticker": ticker,
                                    "trade_price": pos.exit_price,
                                    "market_price": market_price,
                                    "diff": diff
                                })
            except KeyError:
                pass
                
        return pd.DataFrame(discrepancies)


class BacktestAnalyzer:
    def __init__(self, report: BacktestReport, dataset: BacktestDataset | None = None) -> None:
        self.report = report
        self.dataset = dataset
        
        self.portfolio = PortfolioAnalyzer(report)
        self.market = MarketAnalyzer(dataset)
        self.performance = PerformanceAnalyzer(report)
        self.verification = TradeVerifier(report, dataset)

    @classmethod
    def from_tester(cls, tester: Backtester) -> "BacktestAnalyzer":
        dataset = None
        if tester._jobs and tester._jobs[0].dataset:
            dataset = tester._jobs[0].dataset
        return cls(tester.latest_report(), dataset)

    @property
    def group_ids(self) -> List[str]:
        return self.portfolio.group_ids

    def latest_snapshot(self, group_id: str) -> Optional[LatestPortfolioSnapshot]:
        return self.portfolio.latest_snapshot(group_id)

    def latest_snapshots(self, groups: Optional[Sequence[str]] = None) -> Dict[str, LatestPortfolioSnapshot]:
        return self.portfolio.latest_snapshots(groups)
        
    def latest_long_tickers(self, group_id: str, *, min_weight: float = 0.0) -> List[str]:
        return self.portfolio.latest_long_tickers(group_id, min_weight=min_weight)

    def position_history(self, group_id: str, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        return self.portfolio.position_history(group_id, tickers)

    def get_price_history(self, tickers: Sequence[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        return self.market.get_price_history(tickers, start_date, end_date)

    def get_score_history(self, tickers: Sequence[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        return self.market.get_score_history(tickers, start_date, end_date)

    def get_monthly_returns(self, group_id: str) -> pd.DataFrame:
        return self.performance.get_monthly_returns(group_id)

    def get_benchmark_monthly_returns(self) -> pd.DataFrame:
        return self.performance.get_benchmark_monthly_returns()

    def get_summary_stats(self) -> pd.DataFrame:
        return self.performance.get_summary_stats()

    def verify_trade_execution(self, group_id: str, tolerance: float = 1e-6) -> pd.DataFrame:
        return self.verification.verify_trade_execution(group_id, tolerance)


if __name__ == "__main__":
    tester = run_single_example(
        input_days=20,
        return_days=20,
        rebalance_frequency="M",
        apply_trading_costs=False,
        entry_lag=0
    )
    analyzer = BacktestAnalyzer.from_tester(tester)
    
    # NOTE: Quantile 5
    # NOTE: 1. Portfolio Analysis
    # NOTE: 2. Performance Analysis
    # NOTE: 3. Verification
    group = analyzer.group_ids[-1]
    
    snapshot = analyzer.portfolio.latest_snapshot(group)
    if snapshot is None:
        print("No positions available.")
    else:
        print(f"Latest portfolio snapshot for group '{group}'")
        print(f"Enter: {snapshot.enter_date}, Exit: {snapshot.exit_date}")
        print(snapshot.positions.to_string(index=False))
    
    print("\nSummary Statistics:")
    print(analyzer.performance.get_summary_stats())
    
    print("\nMonthly Returns (Top Quantile):")
    print(analyzer.performance.get_monthly_returns(group))
    
    print("\nVerifying trade execution...")
    discrepancies = analyzer.verification.verify_trade_execution(group)
    if discrepancies.empty:
        print("No execution discrepancies found.")
    else:
        print(f"Found {len(discrepancies)} discrepancies:")
        print(discrepancies.head())