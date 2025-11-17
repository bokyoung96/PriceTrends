from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import ExecutionCostModel

@dataclass(frozen=True)
class TradeRecord:
    bucket_id: int
    enter_date: pd.Timestamp
    exit_date: pd.Timestamp
    tickers: Tuple[str, ...]
    capital_in: float
    capital_out: float
    period_return: float
    note: str | None = None


class BucketPortfolio:
    """Tracks capital allocated to a single bucket through time."""

    def __init__(
        self,
        bucket_id: int,
        starting_capital: float,
        cost_model: ExecutionCostModel | None = None,
        *,
        min_price_relative: float = 0.05,
        max_price_relative: float = 20.0,
    ) -> None:
        self.bucket_id = bucket_id
        self.capital = starting_capital
        self._equity: Dict[pd.Timestamp, float] = {}
        self._period_returns: Dict[pd.Timestamp, float] = {}
        self.trades: List[TradeRecord] = []
        self._initialized = False
        self.cost_model = cost_model or ExecutionCostModel.disabled()
        self.min_price_relative = float(min_price_relative)
        self.max_price_relative = float(max_price_relative)

    def mark_initial(self, as_of: pd.Timestamp) -> None:
        if not self._initialized:
            self._equity[as_of] = self.capital
            self._initialized = True

    def rebalance(
        self,
        enter_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        tickers: Sequence[str],
        price_slice: pd.DataFrame,
        note: str | None = None,
    ) -> None:
        """Allocate capital for one holding period and record realised performance."""
        capital_in = self.capital
        requested = tuple(tickers)
        if price_slice.empty:
            raise ValueError("Price slice for the rebalance window is empty.")
        if capital_in == 0 or not requested:
            self._carry_forward(exit_date, capital_in, requested, enter_date, note, price_slice)
            return

        entry_prices = price_slice.iloc[0]
        exit_prices = price_slice.iloc[-1]
        entries, exits, halted = self._select_valid_prices(entry_prices, exit_prices, requested)

        if entries.empty:
            hint = note or "No overlapping price data for requested tickers."
            self._carry_forward(exit_date, capital_in, requested, enter_date, hint, price_slice)
            return

        investable = self.cost_model.net_entry_capital(capital_in)
        if investable <= 0:
            hint = note or "Capital depleted after entry costs."
            self._carry_forward(exit_date, capital_in, tuple(entries.index), enter_date, hint, price_slice)
            return

        price_relatives = (exits / entries).astype(float)
        mean_relative = price_relatives.mean()
        if pd.isna(mean_relative):
            hint = note or "Unable to compute price relatives."
            self._carry_forward(exit_date, capital_in, requested, enter_date, hint, price_slice)
            return

        gross_exit = investable * float(mean_relative)
        tradable_fraction = 0.0 if len(entries) == 0 else (len(entries) - len(halted)) / len(entries)
        capital_out = self.cost_model.net_exit_capital(gross_exit, tradable_fraction=tradable_fraction)

        valid_prices = price_slice.reindex(columns=entries.index)
        daily_equity = self._build_daily_equity_series(
            price_slice=valid_prices,
            entries=entries,
            investable=investable,
            exit_capital=capital_out,
        )

        trade_note = self._merge_notes(note, self._halted_message(halted))
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            tickers=tuple(entries.index),
            capital_in=capital_in,
            capital_out=capital_out,
            note=trade_note,
            daily_equity=daily_equity,
        )

    def equity_series(self) -> pd.Series:
        """Return the equity curve indexed by rebalance exit dates."""
        return pd.Series(self._equity).sort_index()

    def return_series(self) -> pd.Series:
        """Return per-period returns aligned to equity timestamps."""
        return pd.Series(self._period_returns).sort_index()

    def _carry_forward(
        self,
        exit_date: pd.Timestamp,
        capital_in: float,
        tickers: Sequence[str],
        enter_date: pd.Timestamp,
        note: str | None,
        price_slice: pd.DataFrame,
    ) -> None:
        message = note or ("No eligible tickers for this bucket." if not tickers else "Capital set to zero.")
        daily = self._constant_equity_series(price_slice.index, capital_in)
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            tickers=tuple(tickers),
            capital_in=capital_in,
            capital_out=capital_in,
            note=message,
            daily_equity=daily,
        )

    def _select_valid_prices(
        self,
        entry_prices: pd.Series,
        exit_prices: pd.Series,
        tickers: Sequence[str],
    ) -> Tuple[pd.Series, pd.Series, Tuple[str, ...]]:
        entries = entry_prices.reindex(tickers)
        valid_entries = entries.notna() & (entries > 0)
        entries = entries[valid_entries]
        exits = exit_prices.reindex(entries.index)
        missing_exit = exits.isna() | (exits <= 0)
        halted = list(exits.index[missing_exit])

        if missing_exit.any():
            exits = exits.copy()
            exits[missing_exit] = entries[missing_exit]

        relatives = exits / entries
        unstable = (relatives > self.max_price_relative) | (relatives < self.min_price_relative)
        if unstable.any():
            halted.extend(relatives.index[unstable])
            entries = entries[~unstable]
            exits = exits[~unstable]

        return entries, exits, tuple(halted)

    def _halted_message(self, halted: Tuple[str, ...]) -> str | None:
        if not halted:
            return None
        tickers = ", ".join(halted)
        return f"Held positions for halted tickers ({tickers})."

    def _merge_notes(self, *notes: str | None) -> str | None:
        filtered = [n for n in notes if n]
        if not filtered:
            return None
        return " | ".join(filtered)
    def _record_trade(
        self,
        enter_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        tickers: Tuple[str, ...],
        capital_in: float,
        capital_out: float,
        note: str | None,
        daily_equity: pd.Series | None = None,
    ) -> None:
        self.capital = capital_out
        if daily_equity is not None and not daily_equity.empty:
            for timestamp, value in daily_equity.sort_index().items():
                ts = pd.Timestamp(timestamp)
                if ts not in self._equity:
                    self._equity[ts] = float(value)
        self._equity[exit_date] = capital_out
        period_return = 0.0 if capital_in == 0 else (capital_out / capital_in) - 1.0
        self._period_returns[exit_date] = period_return
        self.trades.append(
            TradeRecord(
                bucket_id=self.bucket_id,
                enter_date=enter_date,
                exit_date=exit_date,
                tickers=tickers,
                capital_in=capital_in,
                capital_out=capital_out,
                period_return=period_return,
                note=note,
            )
        )

    def _constant_equity_series(self, dates: pd.Index, value: float) -> pd.Series | None:
        if len(dates) == 0:
            return None
        data = pd.Series(value, index=pd.Index(dates))
        return data

    def _build_daily_equity_series(
        self,
        price_slice: pd.DataFrame,
        entries: pd.Series,
        investable: float,
        exit_capital: float,
    ) -> pd.Series | None:
        if investable <= 0 or entries.empty:
            return None
        working = price_slice.reindex(columns=entries.index)
        if working.empty:
            return None
        working = working.ffill().dropna(how="all")
        if working.empty:
            return None
        relatives = working.div(entries, axis=1)
        mean_relative = relatives.mean(axis=1)
        equity = mean_relative * investable
        if not equity.empty:
            equity.iloc[0] = investable
            equity.iloc[-1] = exit_capital
        return equity
