from doctest import DocFileTest
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.root import DATA_ROOT


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


@dataclass
class ConstituentConverter:
    market: MarketUniverse = MarketUniverse.KOSPI200
    data_root: Path = DATA_ROOT
    header_row: int = 7

    def convert(self) -> Path:
        df = pd.read_excel(
            self.excel_path,
            header=self.header_row,
            index_col=0,
        ).iloc[6:, :]
        df = self._pp_invalid_cols(df)
        df = self._pp_idx(df)

        output_path = self.data_root / self.market.parquet_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, engine="pyarrow")
        return df

    @property
    def excel_path(self) -> Path:
        return self.data_root / self.market.excel_filename

    def _pp_invalid_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        invalid_marker = "#INVALID OPTION"
        invalid_columns = [
            column
            for column in df.columns
            if df[column].astype(str).str.contains(invalid_marker, na=False).any()
        ]
        if invalid_columns:
            df.loc[:, invalid_columns] = 0
        return df

    def _pp_idx(self, df: pd.DataFrame) -> pd.DataFrame:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df.dropna(axis=1, how="all")
        return df


if __name__ == "__main__":
    converter = ConstituentConverter(market=MarketUniverse.KOSPI200)
    df = converter.convert()
