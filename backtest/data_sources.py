from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

FrameSource = Union[pd.DataFrame, str, Path]


@dataclass(frozen=True)
class BacktestDataset:
    scores: pd.DataFrame
    prices: pd.DataFrame
    bench: Optional[pd.Series] = None
    weights: Optional[pd.DataFrame] = None
    open_prices: Optional[pd.DataFrame] = None

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
        if self.weights is not None:
            if not isinstance(self.weights.index, pd.DatetimeIndex):
                raise TypeError("weights index must be a DatetimeIndex.")
            if not self.weights.index.equals(self.scores.index):
                raise ValueError("weights must align with the score index.")
            if not self.weights.columns.equals(self.scores.columns):
                raise ValueError("weights must expose the same ticker columns as scores/prices.")
        if self.open_prices is not None:
            if not isinstance(self.open_prices.index, pd.DatetimeIndex):
                raise TypeError("open_prices index must be a DatetimeIndex.")
            if not self.open_prices.index.equals(self.scores.index):
                raise ValueError("open_prices must align with the score index.")
            if not self.open_prices.columns.equals(self.scores.columns):
                raise ValueError("open_prices must expose the same ticker columns as scores/prices.")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.scores.index

    @property
    def tickers(self) -> pd.Index:
        return self.scores.columns


class FrameLoader:
    def __init__(self, table_name: str) -> None:
        self.table_name = table_name

    def load(self, source: FrameSource) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            if source.empty:
                raise ValueError(f"{self.table_name} dataframe is empty.")
            frame = source.copy()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Expected parquet missing: {path}")
            frame = pd.read_parquet(path)
        return self._prepare(frame)

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df.set_index("Date", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to coerce {self.table_name} index to datetime.") from exc
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated(keep="first")]
        df.columns = df.columns.map(str)
        df = df.dropna(how="all")
        return df


class BacktestDataLoader:
    def __init__(
        self,
        scores_source: FrameSource,
        close_source: FrameSource,
        *,
        constituent_source: FrameSource | None = None,
        benchmark_symbol: str | None = None,
        benchmark_source: FrameSource | None = None,
        weight_source: FrameSource | None = None,
        open_source: FrameSource | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> None:
        self.scores_source = scores_source
        self.close_source = close_source
        self.constituent_source = constituent_source
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_source = benchmark_source
        self.weight_source = weight_source
        self.open_source = open_source
        self.start_date = start_date
        self.end_date = end_date
        self._score_loader = FrameLoader("scores")
        self._price_loader = FrameLoader("close")
        self._score_loader = FrameLoader("scores")
        self._price_loader = FrameLoader("close")
        self._constituent_loader = FrameLoader("constituent")
        self._benchmark_loader = FrameLoader("benchmark") if benchmark_source is not None else None
        self._weight_loader = FrameLoader("weights") if weight_source is not None else None
        self._open_loader = FrameLoader("open") if open_source is not None else None

    def build(self) -> BacktestDataset:
        scores = self._score_loader.load(self.scores_source)
        prices = self._price_loader.load(self.close_source)
        weights = self._weight_loader.load(self.weight_source) if self._weight_loader is not None else None
        open_prices = self._open_loader.load(self.open_source) if self._open_loader is not None else None
        weights = self._weight_loader.load(self.weight_source) if self._weight_loader is not None else None
        open_prices = self._open_loader.load(self.open_source) if self._open_loader is not None else None
        bench = None
        if self.benchmark_symbol and self._benchmark_loader is not None:
            bench_df = self._benchmark_loader.load(self.benchmark_source)
            if self.benchmark_symbol in bench_df.columns:
                bench = bench_df[self.benchmark_symbol]
            else:
                print(f"Warning: Benchmark symbol '{self.benchmark_symbol}' not found in benchmark source.")
        
        aligned_scores, aligned_prices, bench_series = self._align(
            scores,
            prices,
            bench,
            bench_symbol=self.benchmark_symbol,
        )
        aligned_weights = self._align_optional(weights, aligned_scores.index, aligned_scores.columns)
        aligned_open = self._align_optional(open_prices, aligned_scores.index, aligned_scores.columns)
        if self.constituent_source is not None:
            mask = self._constituent_loader.load(self.constituent_source)
            aligned_scores, aligned_prices, aligned_weights, aligned_open = self._apply_constituent_mask(
                aligned_scores, aligned_prices, mask, aligned_weights, aligned_open
            )
        if self.start_date is not None or self.end_date is not None:
            aligned_scores = self._filter_dates(aligned_scores)
            aligned_prices = self._filter_dates(aligned_prices)
            if bench_series is not None:
                bench_series = self._filter_dates(bench_series)
            if aligned_weights is not None:
                aligned_weights = self._filter_dates(aligned_weights)
            if aligned_open is not None:
                aligned_open = self._filter_dates(aligned_open)

        return BacktestDataset(
            scores=aligned_scores,
            prices=aligned_prices,
            bench=bench_series,
            weights=aligned_weights,
            open_prices=aligned_open,
        )

    def _filter_dates(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        if self.start_date:
            data = data[data.index >= self.start_date]
        if self.end_date:
            data = data[data.index <= self.end_date]
        return data

    def _align(
        self,
        scores: pd.DataFrame,
        prices: pd.DataFrame,
        bench: Optional[pd.Series],
        *,
        bench_symbol: str | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series]]:
        common_cols = [c for c in scores.columns if c in prices.columns]
        if not common_cols:
            raise ValueError("No overlapping tickers between scores and prices.")

        # Exclude benchmark symbol from scores and prices if present
        aligned_cols = [c for c in common_cols if not bench_symbol or c != bench_symbol]
        if not aligned_cols:
            raise ValueError("All overlapping tickers were reserved for the benchmark column.")

        scores = scores[aligned_cols]
        prices = prices[aligned_cols]

        price_index = prices.index
        scores = scores.reindex(price_index).ffill()
        valid_index = scores.dropna(how="all").index
        if len(valid_index) < 2:
            raise ValueError("Need at least two overlapping dates to backtest after forward filling.")

        scores = scores.loc[valid_index]
        prices = prices.loc[valid_index]
        bench_series = bench.reindex(valid_index).ffill().dropna() if bench is not None else None
        return scores, prices, bench_series

    def _align_optional(
        self,
        frame: Optional[pd.DataFrame],
        target_index: pd.Index,
        target_columns: pd.Index,
    ) -> Optional[pd.DataFrame]:
        if frame is None:
            return None
        aligned = frame.reindex(target_index).ffill()
        aligned = aligned.reindex(columns=target_columns)
        aligned = aligned.loc[target_index]
        return aligned

    def _apply_constituent_mask(
        self,
        scores: pd.DataFrame,
        prices: pd.DataFrame,
        mask: pd.DataFrame,
        weights: Optional[pd.DataFrame] = None,
        open_prices: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        mask_frame = mask.reindex(scores.index).ffill().fillna(0.0)
        mask_frame = mask_frame.reindex(columns=scores.columns).fillna(0.0)
        masked_scores = scores.where(mask_frame > 0)
        valid_cols = mask_frame.any(axis=0)
        if not valid_cols.any():
            raise ValueError("Constituent filter removed all tickers from the universe.")
        masked_scores = masked_scores.loc[:, valid_cols]
        masked_prices = prices.loc[:, valid_cols]
        masked_weights = None
        if weights is not None:
            masked_weights = weights.where(mask_frame > 0)
            masked_weights = masked_weights.loc[:, valid_cols]
        masked_open = None
        if open_prices is not None:
            masked_open = open_prices.where(mask_frame > 0)
            masked_open = masked_open.loc[:, valid_cols]
        return masked_scores, masked_prices, masked_weights, masked_open
