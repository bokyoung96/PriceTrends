from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .costs import ExecutionCostModel

@dataclass(frozen=True)
class TradeRecord:
    quantile_id: int
    enter_date: pd.Timestamp
    exit_date: pd.Timestamp
    tickers: Tuple[str, ...]
    capital_in: float
    capital_out: float
    period_return: float
    note: str | None = None


class QuantilePortfolio:
    """Tracks capital allocated to a single quantile bucket through time."""

    def __init__(
        self,
        quantile_id: int,
        starting_capital: float,
        cost_model: ExecutionCostModel | None = None,
    ) -> None:
        self.quantile_id = quantile_id
        self.capital = starting_capital
        self._equity: Dict[pd.Timestamp, float] = {}
        self._period_returns: Dict[pd.Timestamp, float] = {}
        self.trades: List[TradeRecord] = []
        self._initialized = False
        self.cost_model = cost_model or ExecutionCostModel.disabled()

    def mark_initial(self, as_of: pd.Timestamp) -> None:
        if not self._initialized:
            self._equity[as_of] = self.capital
            self._initialized = True

    def rebalance(
        self,
        enter_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        tickers: Sequence[str],
        entry_prices: pd.Series,
        exit_prices: pd.Series,
        note: str | None = None,
    ) -> None:
        """Allocate capital for one holding period and record realised performance."""
        capital_in = self.capital
        requested = tuple(tickers)
        if capital_in == 0 or not requested:
            self._carry_forward(exit_date, capital_in, requested, enter_date, note)
            return

        entries, exits, halted = self._select_valid_prices(entry_prices, exit_prices, requested)

        if entries.empty:
            hint = note or "No overlapping price data for requested tickers."
            self._carry_forward(exit_date, capital_in, requested, enter_date, hint)
            return

        investable = self.cost_model.net_entry_capital(capital_in)
        if investable <= 0:
            hint = note or "Capital depleted after entry costs."
            self._carry_forward(exit_date, capital_in, tuple(entries.index), enter_date, hint)
            return

        price_relatives = (exits / entries).astype(float)
        mean_relative = price_relatives.mean()
        if pd.isna(mean_relative):
            hint = note or "Unable to compute price relatives."
            self._carry_forward(exit_date, capital_in, requested, enter_date, hint)
            return

        gross_exit = investable * float(mean_relative)
        tradable_fraction = 0.0 if len(entries) == 0 else (len(entries) - len(halted)) / len(entries)
        capital_out = self.cost_model.net_exit_capital(gross_exit, tradable_fraction=tradable_fraction)

        trade_note = self._merge_notes(note, self._halted_message(halted))
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            tickers=tuple(entries.index),
            capital_in=capital_in,
            capital_out=capital_out,
            note=trade_note,
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
    ) -> None:
        message = note or ("No eligible tickers for this bucket." if not tickers else "Capital set to zero.")
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            tickers=tuple(tickers),
            capital_in=capital_in,
            capital_out=capital_in,
            note=message,
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
        halted = tuple(exits.index[missing_exit])

        if missing_exit.any():
            exits = exits.copy()
            exits[missing_exit] = entries[missing_exit]

        return entries, exits, halted

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
    ) -> None:
        self.capital = capital_out
        self._equity[exit_date] = capital_out
        period_return = 0.0 if capital_in == 0 else (capital_out / capital_in) - 1.0
        self._period_returns[exit_date] = period_return
        self.trades.append(
            TradeRecord(
                quantile_id=self.quantile_id,
                enter_date=enter_date,
                exit_date=exit_date,
                tickers=tickers,
                capital_in=capital_in,
                capital_out=capital_out,
                period_return=period_return,
                note=note,
            )
        )
