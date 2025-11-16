from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

FrameSource = Union[pd.DataFrame, str, Path]


@dataclass(frozen=True)
class BacktestDataset:
    """Holds the aligned signal, tradable price, and benchmark matrices."""

    scores: pd.DataFrame
    prices: pd.DataFrame
    bench: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        if not isinstance(self.scores.index, pd.DatetimeIndex):
            raise TypeError("scores index must be a DatetimeIndex.")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex.")
        if not self.scores.index.equals(self.prices.index):
            raise ValueError("scores and prices must share the exact same index.")
        if not self.scores.columns.equals(self.prices.columns):
            raise ValueError("scores and prices must expose the same ticker columns.")
        if self.bench is not None:
            if not isinstance(self.bench.index, pd.DatetimeIndex):
                raise TypeError("bench index must be a DatetimeIndex.")
            if not self.bench.index.equals(self.scores.index):
                raise ValueError("bench series must align with the score index.")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.scores.index

    @property
    def tickers(self) -> pd.Index:
        return self.scores.columns


class BacktestDatasetBuilder:
    """Loads parquet files and applies minimal cleansing/alignment rules."""

    def __init__(self, scores_source: FrameSource, close_source: FrameSource) -> None:
        self.scores_source = scores_source
        self.close_source = close_source

    def build(self) -> BacktestDataset:
        scores = self._prepare_table(self._resolve_source(self.scores_source, "scores"), table_name="scores")
        prices = self._prepare_table(self._resolve_source(self.close_source, "close"), table_name="close")
        aligned_scores, aligned_prices, bench = self._align(scores, prices)
        return BacktestDataset(scores=aligned_scores, prices=aligned_prices, bench=bench)

    def _resolve_source(self, source: FrameSource, table_name: str) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            if source.empty:
                raise ValueError(f"{table_name} dataframe is empty.")
            return source.copy()

        path = Path(source)
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

    def _align(self, scores: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        common_cols = [c for c in scores.columns if c in prices.columns]
        if not common_cols:
            raise ValueError("No overlapping tickers between scores and prices.")

        bench = prices.get("IKS200") if "IKS200" in prices.columns else None
        aligned_cols = [c for c in common_cols if c != "IKS200"]
        if not aligned_cols:
            raise ValueError("All overlapping columns were reserved for benchmark.")
        scores = scores[aligned_cols]
        prices = prices[aligned_cols]

        price_index = prices.index
        scores = scores.reindex(price_index).ffill()
        valid_index = scores.dropna(how="all").index
        if len(valid_index) < 2:
            raise ValueError("Need at least two overlapping dates to backtest after forward filling.")

        scores = scores.loc[valid_index]
        prices = prices.loc[valid_index]
        bench = bench.reindex(valid_index) if bench is not None else None
        return scores, prices, bench
