from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class BacktestDataset:
    """Holds the aligned score and price matrices used during simulation."""

    scores: pd.DataFrame
    prices: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.scores.index, pd.DatetimeIndex):
            raise TypeError("scores index must be a DatetimeIndex.")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex.")
        if not self.scores.index.equals(self.prices.index):
            raise ValueError("scores and prices must share the exact same index.")
        if not self.scores.columns.equals(self.prices.columns):
            raise ValueError("scores and prices must expose the same ticker columns.")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.scores.index

    @property
    def tickers(self) -> pd.Index:
        return self.scores.columns


class BacktestDatasetBuilder:
    """Loads parquet files and applies minimal cleansing/alignment rules."""

    def __init__(self, scores_path: Path, close_path: Path) -> None:
        self.scores_path = Path(scores_path)
        self.close_path = Path(close_path)

    def build(self) -> BacktestDataset:
        scores = self._prepare_table(self._load_parquet(self.scores_path), table_name="scores")
        prices = self._prepare_table(self._load_parquet(self.close_path), table_name="close")
        aligned_scores, aligned_prices = self._align(scores, prices)
        return BacktestDataset(scores=aligned_scores, prices=aligned_prices)

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Expected parquet missing: {path}")
        return pd.read_parquet(path)

    def _prepare_table(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        df = df.copy()
        if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("Date", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to coerce {table_name} index to datetime.") from exc
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df.columns = df.columns.map(str)
        df = df.dropna(how="all")
        return df

    def _align(self, scores: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        common_cols = [c for c in scores.columns if c in prices.columns]
        if not common_cols:
            raise ValueError("No overlapping tickers between scores and prices.")

        scores = scores[common_cols]
        prices = prices[common_cols]

        common_index = scores.index.intersection(prices.index)
        if len(common_index) < 2:
            raise ValueError("Need at least two overlapping dates to backtest.")

        scores = scores.loc[common_index]
        prices = prices.loc[common_index]
        return scores, prices
