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
class PositionLedgerEntry:
    ticker: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_value: float
    exit_value: float


@dataclass(frozen=True)
class TradeRecord:
    group_id: str
    enter_date: pd.Timestamp
    exit_date: pd.Timestamp
    positions: Tuple[PositionLedgerEntry, ...]
    capital_in: float
    capital_out: float
    period_return: float
    note: str | None = None


class PortfolioTrack:
    """Tracks the evolution of capital allocated to a single portfolio group."""

    def __init__(
        self,
        group_id: str,
        starting_capital: float,
        cost_model: ExecutionCostModel | None = None,
        *,
        min_price_relative: float = 0.05,
        max_price_relative: float = 20.0,
    ) -> None:
        self.group_id = str(group_id)
        self.capital = starting_capital
        self.cost_model = cost_model or ExecutionCostModel.disabled()
        self.min_price_relative = float(min_price_relative)
        self.max_price_relative = float(max_price_relative)
        self._equity: Dict[pd.Timestamp, float] = {}
        self._period_returns: Dict[pd.Timestamp, float] = {}
        self.trades: List[TradeRecord] = []
        self._initialized = False

    def mark_initial(self, as_of: pd.Timestamp) -> None:
        if not self._initialized:
            self._equity[pd.Timestamp(as_of)] = float(self.capital)
            self._initialized = True

    def rebalance(
        self,
        *,
        enter_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        tickers: Sequence[str],
        price_slice: pd.DataFrame,
        note: str | None = None,
    ) -> None:
        capital_in = float(self.capital)
        requested = tuple(tickers)
        if price_slice.empty:
            raise ValueError("Price slice for the rebalance window is empty.")
        if capital_in <= 0 or not requested:
            self._carry_forward(
                exit_date=exit_date,
                capital_in=capital_in,
                positions=tuple(),
                enter_date=enter_date,
                note=note or "No eligible tickers for this portfolio group.",
                price_slice=price_slice,
            )
            return

        entry_prices = price_slice.iloc[0]
        exit_prices = price_slice.iloc[-1]
        entries, exits, halted = self._select_valid_prices(entry_prices, exit_prices, requested)
        if entries.empty:
            hint = note or "No overlapping price data for requested tickers."
            self._carry_forward(
                exit_date=exit_date,
                capital_in=capital_in,
                positions=tuple(),
                enter_date=enter_date,
                note=hint,
                price_slice=price_slice,
            )
            return

        investable = self.cost_model.net_entry_capital(capital_in)
        if investable <= 0:
            hint = note or "Capital depleted after entry costs."
            self._carry_forward(
                exit_date=exit_date,
                capital_in=capital_in,
                positions=tuple(),
                enter_date=enter_date,
                note=hint,
                price_slice=price_slice,
            )
            return

        quantities, ledger_entries = self._build_position_ledgers(entries, exits, investable)
        if quantities.empty:
            hint = note or "Unable to determine position sizes."
            self._carry_forward(
                exit_date=exit_date,
                capital_in=capital_in,
                positions=tuple(),
                enter_date=enter_date,
                note=hint,
                price_slice=price_slice,
            )
            return

        gross_exit = float((exits.reindex(index=quantities.index) * quantities).sum())
        tradable_fraction = 0.0 if len(entries) == 0 else (len(entries) - len(halted)) / len(entries)
        capital_out = self.cost_model.net_exit_capital(gross_exit, tradable_fraction=tradable_fraction)

        valid_prices = price_slice.reindex(columns=quantities.index)
        daily_equity = self._build_daily_equity_series(
            price_slice=valid_prices,
            quantities=quantities,
            investable=investable,
            exit_capital=capital_out,
        )

        trade_note = self._merge_notes(note, self._halted_message(halted))
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            positions=ledger_entries,
            capital_in=capital_in,
            capital_out=capital_out,
            note=trade_note,
            daily_equity=daily_equity,
        )

    def equity_series(self) -> pd.Series:
        return pd.Series(self._equity).sort_index()

    def return_series(self) -> pd.Series:
        return pd.Series(self._period_returns).sort_index()

    def _carry_forward(
        self,
        *,
        exit_date: pd.Timestamp,
        capital_in: float,
        positions: Tuple[PositionLedgerEntry, ...],
        enter_date: pd.Timestamp,
        note: str | None,
        price_slice: pd.DataFrame,
    ) -> None:
        message = note or ("No eligible tickers for this portfolio group." if not positions else "Capital set to zero.")
        daily = self._constant_equity_series(price_slice.index, capital_in)
        self._record_trade(
            enter_date=enter_date,
            exit_date=exit_date,
            positions=positions,
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
        *,
        enter_date: pd.Timestamp,
        exit_date: pd.Timestamp,
        positions: Tuple[PositionLedgerEntry, ...],
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
        self._equity[pd.Timestamp(exit_date)] = capital_out
        period_return = 0.0 if capital_in == 0 else (capital_out / capital_in) - 1.0
        self._period_returns[pd.Timestamp(exit_date)] = period_return
        self.trades.append(
            TradeRecord(
                group_id=self.group_id,
                enter_date=enter_date,
                exit_date=exit_date,
                positions=positions,
                capital_in=capital_in,
                capital_out=capital_out,
                period_return=period_return,
                note=note,
            )
        )

    def _constant_equity_series(self, dates: pd.Index, value: float) -> pd.Series | None:
        if len(dates) == 0:
            return None
        return pd.Series(value, index=pd.Index(dates))

    def _build_daily_equity_series(
        self,
        *,
        price_slice: pd.DataFrame,
        quantities: pd.Series,
        investable: float,
        exit_capital: float,
    ) -> pd.Series | None:
        if investable <= 0 or quantities.empty:
            return None
        working = price_slice.reindex(columns=quantities.index)
        if working.empty:
            return None
        working = working.ffill().dropna(how="all")
        if working.empty:
            return None
        values = working.mul(quantities, axis=1)
        equity = values.sum(axis=1)
        if not equity.empty:
            equity.iloc[0] = investable
            equity.iloc[-1] = exit_capital
        return equity

    def _build_position_ledgers(
        self,
        entries: pd.Series,
        exits: pd.Series,
        investable: float,
    ) -> tuple[pd.Series, Tuple[PositionLedgerEntry, ...]]:
        if entries.empty or investable <= 0:
            return pd.Series(dtype=float), tuple()
        per_position = investable / len(entries)
        ledger_entries: list[PositionLedgerEntry] = []
        quantities: Dict[str, float] = {}
        for ticker, entry_price in entries.items():
            if entry_price <= 0:
                continue
            qty = per_position / float(entry_price)
            exit_price = float(exits.get(ticker, entry_price))
            ticker_id = str(ticker)
            quantities[ticker_id] = float(qty)
            ledger_entries.append(
                PositionLedgerEntry(
                    ticker=ticker_id,
                    quantity=float(qty),
                    entry_price=float(entry_price),
                    exit_price=exit_price,
                    entry_value=float(entry_price * qty),
                    exit_value=float(exit_price * qty),
                )
            )
        return pd.Series(quantities, dtype=float), tuple(ledger_entries)
