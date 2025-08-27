import os
import pandas as pd
import polars as pl
import json
from dataclasses import dataclass
from typing import Dict, List, Set


class DataConverter:
    def __init__(self, excel_path: str, output_dir: str) -> None:
        self.excel_path = excel_path
        self.output_dir = output_dir
        self.excluded_tickers: Set[str] = set()
        self._output_dir()

    def _output_dir(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_invalid_tickers(self, df: pd.DataFrame, tickers: List[str], data_types: List[str]) -> Set[str]:
        invalid_tickers = set()
        ohlc_types = ["수정시가", "수정고가", "수정저가", "수정주가"]
        
        for target_korean in ohlc_types:
            col_idx = 0
            for i, data_type in enumerate(data_types):
                if pd.notna(data_type) and data_type.strip() == target_korean:
                    if col_idx < len(tickers):
                        ticker = tickers[col_idx]
                        if pd.notna(ticker):
                            ticker_data = df.iloc[:, i]
                            if ticker_data.astype(str).str.contains("#INVALID OPTION", na=False).any():
                                invalid_tickers.add(ticker)
                col_idx += 1
        
        return invalid_tickers

    def _get_data_by_type(self, df: pd.DataFrame, tickers: List[str], data_types: List[str], target_type: str) -> pd.DataFrame:
        type_mapping = {
            "수정시가": "open",
            "수정고가": "high", 
            "수정저가": "low",
            "수정주가": "close",
            "거래량": "volume"
        }
        
        target_korean = None
        for korean, english in type_mapping.items():
            if english == target_type:
                target_korean = korean
                break
                
        if target_korean is None:
            return pd.DataFrame()
            
        result_data = {}
        col_idx = 0
        
        for i, data_type in enumerate(data_types):
            if pd.notna(data_type) and data_type.strip() == target_korean:
                if col_idx < len(tickers):
                    ticker = tickers[col_idx]
                    if pd.notna(ticker) and ticker not in self.excluded_tickers:
                        result_data[ticker] = df.iloc[:, i]
            col_idx += 1
            
        return pd.DataFrame(result_data)

    def data_convert(self) -> None:
        df_full = pd.read_excel(self.excel_path, header=None)
        
        tickers = df_full.iloc[7, 1:].tolist()
        data_types = df_full.iloc[13, 1:].tolist()
        
        dates = df_full.iloc[14:, 0]
        dates = pd.to_datetime(dates, errors="coerce").dropna()
        
        data_section = df_full.iloc[14:14+len(dates), 1:]
        data_section.index = dates
        data_section.index.name = None
        
        self.excluded_tickers = self._get_invalid_tickers(data_section, tickers, data_types)
        
        if self.excluded_tickers:
            excluded_path = os.path.join(self.output_dir, "excluded_tickers.json")
            with open(excluded_path, 'w', encoding='utf-8') as f:
                json.dump(list(self.excluded_tickers), f, ensure_ascii=False, indent=2)
            print(f"Excluded tickers due to #INVALID OPTION: {sorted(self.excluded_tickers)}")
        
        for data_type in ["open", "high", "low", "close", "volume"]:
            type_df = self._get_data_by_type(data_section, tickers, data_types, data_type)
            if not type_df.empty:
                type_df.index.name = None
                out_path = os.path.join(self.output_dir, f"{data_type}.parquet")
                type_df.to_parquet(out_path, engine="pyarrow")


class DataLoader:
    @dataclass(frozen=True)
    class DataEntry:
        name: str
        path: str

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self._registry: Dict[str, DataLoader.DataEntry] = self._register()
        self._cache: Dict[str, pl.DataFrame] = {}

    def _register(self) -> Dict[str, 'DataLoader.DataEntry']:
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.parquet')]
        return {
            os.path.splitext(f)[0]: DataLoader.DataEntry(
                name=os.path.splitext(f)[0],
                path=os.path.join(self.data_dir, f)
            ) for f in files
        }

    def available(self) -> List[str]:
        return list(self._registry.keys())

    def load(self, name: str) -> pl.DataFrame:
        if name not in self._registry:
            raise ValueError(f"Dataset '{name}' not found. Available: {self.available()}")
        
        if name in self._cache:
            return self._cache[name]
        
        data = pl.read_parquet(self._registry[name].path)
        self._cache[name] = data
        return data

    def get_excluded_tickers(self) -> List[str]:
        excluded_path = os.path.join(self.data_dir, "excluded_tickers.json")
        if os.path.exists(excluded_path):
            with open(excluded_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def __getattr__(self, name: str) -> pl.DataFrame:
        if name in self._registry:
            return self.load(name)
        raise AttributeError(f"No such dataset: {name}")
    
    def to_pandas(self, data: pl.DataFrame) -> pd.DataFrame:
        df = data.to_pandas()
        if 'Date' in df.columns:
            df.set_index('Date', inplace=True)
            df.index = pd.to_datetime(df.index)
        elif '__index_level_0__' in df.columns:
            df.set_index('__index_level_0__', inplace=True)
            df.index = pd.to_datetime(df.index)
            df.index.name = 'Date'
        return df
    
    def get_latest_date(self) -> str:
        close_data = self.load("close")
        latest = close_data.select(pl.col("Date").max()).item()
        return latest.strftime('%Y%m%d')


if __name__ == "__main__":
    # NOTE: DATA CONVERTER
    converter = DataConverter(
        excel_path=os.path.join(os.path.dirname(__file__), "..", "DATA", "DATA.xlsx"),
        output_dir=os.path.join(os.path.dirname(__file__), "..", "DATA")
    )
    converter.data_convert()

    # NOTE: DATA LOADER
    loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "DATA"))
    print("Available datasets:", loader.available())
    excluded = loader.get_excluded_tickers()
    if excluded:
        print("Excluded tickers:", excluded)