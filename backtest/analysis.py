from __future__ import annotations

import sys
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


@dataclass
class LatestPortfolioSnapshot:
    group_id: str
    enter_date: pd.Timestamp
    exit_date: pd.Timestamp
    positions: pd.DataFrame


class BacktestAnalyzer:
    def __init__(self, report: BacktestReport) -> None:
        self.report = report

    @classmethod
    def from_tester(cls, tester: Backtester) -> "BacktestAnalyzer":
        return cls(tester.latest_report())

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
    group = analyzer.group_ids[-1]
    snapshot = analyzer.latest_snapshot(group)
    if snapshot is None:
        print("No positions available.")
    else:
        print(f"Latest portfolio snapshot for group '{group}'")
        print(f"Enter: {snapshot.enter_date}, Exit: {snapshot.exit_date}")
        print(snapshot.positions.to_string(index=False))

