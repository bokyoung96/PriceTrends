from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.root import DATA_ROOT

DatasetPreprocessor = Callable[[pd.DataFrame], pd.DataFrame]


class MarketUniverse(Enum):
    KOSPI200 = "KOSPI200"
    KOSPI = "KOSPI"
    KOSDAQ150 = "KOSDAQ150"

    @property
    def excel_filename(self) -> str:
        return f"CONST_{self.value}.xlsx"

    @property
    def parquet_filename(self) -> str:
        return f"CONST_{self.value}.parquet"


class MarketMetric(Enum):
    MKTCAP = "MKTCAP"
    FOREIGN = "FOREIGN"
    TRANS_BAN = "TRANS_BAN"
    BM = "BM"

    @property
    def source_filename(self) -> str:
        return f"{self.value}.xlsx"

    @property
    def parquet_filename(self) -> str:
        return f"{self.value}.parquet"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: Path
    output: Path
    header_row: int | None = 0
    index_column: int | str | None = 0
    preprocessors: Sequence[DatasetPreprocessor] = ()

    def with_paths(self, *, source: Path | None = None, output: Path | None = None) -> "DatasetSpec":
        return replace(self, source=source or self.source, output=output or self.output)

    def convert(self, *, persist: bool = True) -> pd.DataFrame:
        if not self.source.exists():
            raise FileNotFoundError(f"Source file not found: {self.source}")
        df = pd.read_excel(
            self.source,
            header=self.header_row,
            index_col=self.index_column,
        )
        for preprocessor in self.preprocessors:
            df = preprocessor(df)
        if persist:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.output, engine="pyarrow")
        return df


@dataclass(frozen=True)
class DatasetFactory:
    data_root: Path = DATA_ROOT

    def constituents(
        self,
        *,
        market: MarketUniverse,
        header_row: int = 7,
        skip_rows: int = 6,
    ) -> DatasetSpec:
        source = self.data_root / market.excel_filename
        output = self.data_root / market.parquet_filename
        preprocessors: list[DatasetPreprocessor] = [
            slice_rows(skip_rows),
            drop_invalid_marker(),
            ensure_datetime_index(),
            drop_all_na_columns(),
        ]
        return DatasetSpec(
            name=f"constituents:{market.value}",
            source=source,
            output=output,
            header_row=header_row,
            index_column=0,
            preprocessors=preprocessors,
        )

    def metric(
        self,
        *,
        metric: MarketMetric,
        source_name: str | None = None,
        header_row: int = 7,
        skip_rows: int = 6,
        index_column: int | str | None = 0,
    ) -> DatasetSpec:
        source = self.data_root / (source_name or metric.source_filename)
        output = self.data_root / metric.parquet_filename
        preprocessors: list[DatasetPreprocessor] = [
            slice_rows(skip_rows),
            drop_invalid_marker(),
            ensure_datetime_index(),
            drop_all_na_columns(),
        ]
        return DatasetSpec(
            name=f"metric:{metric.value}",
            source=source,
            output=output,
            header_row=header_row,
            index_column=index_column,
            preprocessors=preprocessors,
        )

    def metrics(
        self,
        *,
        metric: MarketMetric = MarketMetric.FOREIGN,
        windows: Sequence[int] = (5, 20, 60),
        source_name: str | None = None,
        header_row: int = 7,
        skip_rows: int = 6,
        index_column: int | str | None = 0,
    ) -> dict[int, DatasetSpec]:
        if not windows:
            raise ValueError("At least one rolling window must be provided")

        unique_windows: list[int] = []
        for window in windows:
            if window <= 0:
                raise ValueError(f"Rolling window must be positive, received {window}")
            if window not in unique_windows:
                unique_windows.append(window)

        base_preprocessors: Sequence[DatasetPreprocessor] = [
            slice_rows(skip_rows),
            drop_invalid_marker(),
            ensure_datetime_index(),
            drop_all_na_columns(),
        ]

        specs: dict[int, DatasetSpec] = {}
        source = self.data_root / (source_name or metric.source_filename)
        for window in unique_windows:
            preprocessors = list(base_preprocessors) + [
                ensure_numeric_values(),
                rolling_sum(window),
                drop_all_na_columns(),
            ]
            specs[window] = DatasetSpec(
                name=f"metric:{metric.value}:{window}",
                source=source,
                output=self.data_root / f"FOREIGN_{window}.parquet",
                header_row=header_row,
                index_column=index_column,
                preprocessors=preprocessors,
            )
        return specs

    def bm(
        self,
        *,
        source_name: str | None = None,
        header_row: int = 7,
        skip_rows: int = 6,
        index_column: int | str | None = 0,
    ) -> DatasetSpec:
        return self.metric(
            metric=MarketMetric.BM,
            source_name=source_name,
            header_row=header_row,
            skip_rows=skip_rows,
            index_column=index_column,
        )


def slice_rows(start: int) -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[start:, :]

    return _inner


def drop_invalid_marker(marker: str = "#INVALID OPTION") -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        invalid_columns = [
            column
            for column in df.columns
            if df[column].astype(str).str.contains(marker, na=False).any()
        ]
        if invalid_columns:
            df.loc[:, invalid_columns] = 0
        return df

    return _inner


def ensure_datetime_index() -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        return df

    return _inner


def drop_all_na_columns() -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how="all")

    return _inner


def lower_column_names() -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.lower()
        return df

    return _inner


def ensure_numeric_values() -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        return df.apply(pd.to_numeric, errors="coerce")

    return _inner


def rolling_sum(window: int) -> DatasetPreprocessor:
    def _inner(df: pd.DataFrame) -> pd.DataFrame:
        rolled = df.sort_index().rolling(window=window, min_periods=window).sum()
        return rolled.iloc[window - 1 :]

    return _inner


if __name__ == "__main__":
    factory = DatasetFactory()
    res: dict[str, pd.DataFrame] = {}
    examples = {
        # "constituent": factory.constituents(market=MarketUniverse.KOSPI200),
        # "metric:mktcap": factory.metric(metric=MarketMetric.MKTCAP),
        # "metric:trans_ban": factory.metric(metric=MarketMetric.TRANS_BAN),
        "metric:bm": factory.bm(),
        # "metric:foreign": factory.metrics(),
    }
    for label, spec in examples.items():
        try:
            if isinstance(spec, dict):
                for window, window_spec in spec.items():
                    res[f"{label}:{window}"] = window_spec.convert()
            else:
                res[label] = spec.convert()
        except FileNotFoundError:
            continue
