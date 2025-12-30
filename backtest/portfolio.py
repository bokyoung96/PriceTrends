from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from enum import Enum

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.costs import ExecutionCostModel


class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def is_short(self) -> bool:
        return self is PositionSide.SHORT

    @property
    def is_long(self) -> bool:
        return self is PositionSide.LONG


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
    def __init__(
        self,
        group_id: str,
        starting_capital: float,
        cost_model: ExecutionCostModel | None = None,
        *,
        min_price_relative: float = 0.05,
        max_price_relative: float = 20.0,
        side: PositionSide = PositionSide.LONG,
    ) -> None:
        self.group_id = str(group_id)
        self.capital = starting_capital
        self.cost_model = cost_model or ExecutionCostModel.disabled()
        self.min_price_relative = float(min_price_relative)
        self.max_price_relative = float(max_price_relative)
        self.side = side
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
        weights: pd.Series | None = None,
        entry_prices: pd.Series | None = None,
    ) -> None:
        capital_in = max(0.0, float(self.capital))
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
 
        entry_prices = entry_prices if entry_prices is not None else price_slice.iloc[0]
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

        normalized_weights = self._normalize_weights(entries.index, weights)
        quantities, ledger_entries = self._build_position_ledgers(entries, exits, investable, normalized_weights)
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

        if self.side.is_short:
            pnl = float(((entries.reindex(index=quantities.index) - exits.reindex(index=quantities.index)) * quantities).sum())
            gross_exit = capital_in + pnl
        else:
            gross_exit = float((exits.reindex(index=quantities.index) * quantities).sum())
        tradable_fraction = 0.0 if len(entries) == 0 else (len(entries) - len(halted)) / len(entries)
        capital_out = self.cost_model.net_exit_capital(gross_exit, tradable_fraction=tradable_fraction)

        valid_prices = price_slice.reindex(columns=quantities.index)
        daily_equity = self._build_daily_equity_series(
            price_slice=valid_prices,
            quantities=quantities,
            investable=investable,
            exit_capital=capital_out,
            entry_prices=entry_prices,
        )
        daily_equity, hit_floor = self._apply_equity_floor(daily_equity)
        if hit_floor:
            capital_out = 0.0

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
                if ts == pd.Timestamp(enter_date) or ts not in self._equity:
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
        entry_prices: pd.Series | None = None,
    ) -> pd.Series | None:
        if investable <= 0 or quantities.empty:
            return None
        working = price_slice.reindex(columns=quantities.index)
        if working.empty:
            return None
        working = working.ffill().dropna(how="all")
        if working.empty:
            return None
        if self.side.is_short:
            if entry_prices is None:
                return None
            entries = pd.Series(entry_prices).reindex(quantities.index).ffill().bfill()
            pnl_series = (entries - working).mul(quantities, axis=1)
            equity = investable + pnl_series.sum(axis=1)
        else:
            values = working.mul(quantities, axis=1)
            equity = values.sum(axis=1)
        if equity.empty:
            return None
        equity.iloc[0] = investable
        equity.iloc[-1] = exit_capital
        return equity

    def _apply_equity_floor(
        self,
        equity: pd.Series | None,
        *,
        floor: float = 0.0,
    ) -> tuple[pd.Series | None, bool]:
        if equity is None or equity.empty:
            return equity, False
        hits = equity <= floor
        if not hits.any():
            return equity, False
        first_hit = hits.idxmax()
        floored = equity.copy()
        floored.loc[first_hit:] = floor
        return floored, True

    def _normalize_weights(self, tickers: Sequence[str], weights: pd.Series | None) -> pd.Series | None:
        if weights is None:
            return None
        working = pd.Series(weights, dtype=float).reindex(tickers)
        if working.isna().any():
            return None
        working = working.clip(lower=0.0)
        total = working.sum()
        if total <= 0:
            return None
        return working / total

    def _build_position_ledgers(
        self,
        entries: pd.Series,
        exits: pd.Series,
        investable: float,
        weights: pd.Series | None = None,
    ) -> tuple[pd.Series, Tuple[PositionLedgerEntry, ...]]:
        if entries.empty or investable <= 0:
            return pd.Series(dtype=float), tuple()
        if weights is None:
            allocation = pd.Series(1.0 / len(entries), index=entries.index, dtype=float)
        else:
            allocation = weights.reindex(entries.index).fillna(0.0)
        ledger_entries: list[PositionLedgerEntry] = []
        quantities: Dict[str, float] = {}
        for ticker, entry_price in entries.items():
            if entry_price <= 0:
                continue
            share = float(allocation.get(ticker, 0.0))
            if share <= 0:
                continue
            capital = investable * share
            if capital <= 0:
                continue
            qty = capital / float(entry_price)
            exit_price = float(exits.get(ticker, entry_price))
            ticker_id = str(ticker)
            quantities[ticker_id] = float(qty)
            if self.side.is_short:
                exit_value = float(capital + (float(entry_price) - exit_price) * qty)
                entry_value = float(capital)
            else:
                exit_value = float(exit_price * qty)
                entry_value = float(entry_price * qty)
            ledger_entries.append(
                PositionLedgerEntry(
                    ticker=ticker_id,
                    quantity=float(qty if self.side.is_long else -qty),
                    entry_price=float(entry_price),
                    exit_price=exit_price,
                    entry_value=entry_value,
                    exit_value=exit_value,
                )
            )
        return pd.Series(quantities, dtype=float), tuple(ledger_entries)
