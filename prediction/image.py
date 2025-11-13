import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from PIL import Image, ImageDraw
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.loader import DataLoader
from utils.root import DATA_ROOT, IMAGES_ROOT


@dataclass(frozen=True)
class ChartConfig:
    intervals: int = 5
    image_height: int = 32
    include_ma: bool = True
    ma_windows: Optional[Tuple[int, ...]] = None
    include_volume: bool = True
    background_color: int = 0
    chart_color: int = 255
    volume_chart_gap: int = 1
    img_save_dir: str = str(IMAGES_ROOT)


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame
    low: pd.DataFrame
    high: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        self._validate_data_consistency()

    def _validate_data_consistency(self) -> None:
        dataframes = [self.open, self.low, self.high, self.close, self.volume]
        
        if not all(df.index.equals(dataframes[0].index) for df in dataframes):
            raise ValueError('Dataframes do not have matching indices')
        
        if not all(df.columns.equals(dataframes[0].columns) for df in dataframes):
            raise ValueError('Dataframes do not have matching columns')


class ChartGenerator:
    def __init__(self, 
                 market_data: MarketData, 
                 config: ChartConfig = ChartConfig()) -> None:
        self.market_data = market_data
        self.config = config
        self._ma_data_cache: Dict[int, pd.DataFrame] = {}
        self._ma_windows: Tuple[int, ...] = self._get_ma_windows()

    @property
    def ma_data(self) -> pd.DataFrame:
        primary_window = self._ma_windows[0] if self._ma_windows else self.config.intervals
        return self._get_ma_data(primary_window)

    @property
    def ma_windows(self) -> Tuple[int, ...]:
        return self._ma_windows

    def _get_ma_windows(self) -> Tuple[int, ...]:
        if not self.config.include_ma:
            return ()

        if self.config.ma_windows:
            ordered_windows = []
            seen = set()
            for window in self.config.ma_windows:
                if window is None:
                    continue
                if not isinstance(window, int):
                    raise ValueError("MA window sizes must be integers")
                if window <= 0:
                    raise ValueError("MA window sizes must be positive integers")
                if window in seen:
                    continue
                seen.add(window)
                ordered_windows.append(window)
            if not ordered_windows:
                return (self.config.intervals,)
            return tuple(ordered_windows)

        return (self.config.intervals,)

    def _get_ma_data(self, window: int) -> pd.DataFrame:
        if window not in self._ma_data_cache:
            self._ma_data_cache[window] = self.market_data.close.rolling(window=window).mean()
        return self._ma_data_cache[window]

    def get_valid_start_dates(self, ticker: str) -> pd.Index:
        if len(self.market_data.close.index) == 0:
            return self.market_data.close.index

        min_start = max(self.config.intervals - 1, 0)
        candidate_index = self.market_data.close.index[min_start:]

        if not self._ma_windows:
            return candidate_index

        valid_index = candidate_index
        for window in self._ma_windows:
            ma_series = self._get_ma_data(window)[ticker].dropna()
            valid_index = valid_index.intersection(ma_series.index)
        return valid_index

    def _extract_arrays(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> np.ndarray:
        date_slice = slice(start_date, end_date)
        
        arrays = [
            self.market_data.open.loc[date_slice, ticker].values,
            self.market_data.low.loc[date_slice, ticker].values,
            self.market_data.high.loc[date_slice, ticker].values,
            self.market_data.close.loc[date_slice, ticker].values,
        ]
        
        if self._ma_windows:
            for window in self._ma_windows:
                arrays.append(self._get_ma_data(window).loc[date_slice, ticker].values)
        
        if self.config.include_volume:
            arrays.append(self.market_data.volume.loc[date_slice, ticker].values)  
        return np.vstack(arrays)

    def _calculate_heights(self) -> Tuple[int, int, int]:
        total_height = self.config.image_height
        
        if self.config.include_volume:
            volume_height = int(total_height / 5)
            price_height = total_height - volume_height - self.config.volume_chart_gap
        else:
            volume_height = 0
            price_height = total_height
        return total_height, price_height, volume_height

    def _normalize_price_data(self, price_arrays: np.ndarray, price_height: int, volume_height: int) -> np.ndarray:
        if price_arrays.shape[0] < 4:
            start_close = price_arrays[0, 0]
        else:
            start_close = price_arrays[3, 0]

        if start_close == 0 or np.isnan(start_close):
            raise RuntimeError("First day close price is zero or NaN")
        
        normalized_prices = price_arrays / start_close
        
        min_price = np.nanmin(normalized_prices)
        max_price = np.nanmax(normalized_prices)
        
        if np.isnan(min_price) or np.isnan(max_price):
            raise RuntimeError("Price data contains invalid (NaN) values")
        
        if min_price == max_price:
            raise RuntimeError("All prices are the same")
        
        offset = volume_height + self.config.volume_chart_gap if self.config.include_volume else 0
        scaled_prices = (normalized_prices - min_price) / (max_price - min_price) * (price_height - 1) + offset
        
        return scaled_prices

    def _normalize_volume_data(self, volume_array: np.ndarray, volume_height: int) -> np.ndarray:
        max_volume = np.nanmax(volume_array)
        if max_volume == 0 or np.isnan(max_volume) or max_volume <= 0:
            return np.zeros_like(volume_array)
        
        normalized_volume = (volume_array / max_volume) * (volume_height - 1)
        return np.nan_to_num(normalized_volume, nan=0.0)

    def _create_base_image(self, total_height: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("L", (self.config.intervals * 3, total_height), self.config.background_color)
        return image, ImageDraw.Draw(image)

    def _draw_ohlc_chart(self, draw: ImageDraw.ImageDraw, price_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length):
            draw.point((i * 3, price_data[0][i]), fill=self.config.chart_color)
            draw.point((i * 3 + 2, price_data[3][i]), fill=self.config.chart_color)
            draw.line((i * 3 + 1, price_data[1][i], i * 3 + 1, price_data[2][i]), fill=self.config.chart_color)

    def _draw_ma_line(self, draw: ImageDraw.ImageDraw, ma_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length - 1):
            draw.line((i * 3, ma_data[i], (i + 1) * 3, ma_data[i + 1]), fill=self.config.chart_color)

    def _draw_volume_bars(self, draw: ImageDraw.ImageDraw, volume_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length):
            draw.line((i * 3 + 1, 0, i * 3 + 1, volume_data[i]), fill=self.config.chart_color)

    def _calculate_label(self, ticker: str, estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> int:
        start_price = self.market_data.close.loc[estimation_start, ticker]
        end_price = self.market_data.close.loc[estimation_end, ticker]
        return 1 if (end_price / start_price) > 1 else 0

    def generate_chart_image(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                           estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> Tuple[Image.Image, int]:
        try:
            arrays = self._extract_arrays(ticker, start_date, end_date)
            
            ma_array_count = len(self._ma_windows)
            price_array_count = 4 + ma_array_count
            price_arrays_to_check = arrays[:price_array_count]
            if np.isnan(price_arrays_to_check).any():
                raise ValueError("Price or MA data contains NaN, skipping chart.")
            
            data_length = len(self.market_data.close.loc[start_date:end_date].index)
            
            total_height, price_height, volume_height = self._calculate_heights()
            
            price_arrays_to_normalize = arrays[:price_array_count]
            normalized_data = self._normalize_price_data(price_arrays_to_normalize, price_height, volume_height).astype(int)
            normalized_prices = normalized_data[:4]
            
            image, draw = self._create_base_image(total_height)
            
            self._draw_ohlc_chart(draw, normalized_prices, data_length)
            
            if ma_array_count:
                normalized_ma_arrays = normalized_data[4:]
                for ma_series in normalized_ma_arrays:
                    self._draw_ma_line(draw, ma_series, data_length)
            
            if self.config.include_volume:
                volume_data = self._normalize_volume_data(arrays[-1], volume_height).astype(int)
                self._draw_volume_bars(draw, volume_data, data_length)
            
            final_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = self._calculate_label(ticker, estimation_start, estimation_end)
            return final_image, label
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate chart for {ticker}: {e}")


class ChartDebugger:
    def __init__(self, chart_generator: ChartGenerator) -> None:
        self.generator = chart_generator

    def save_chart_image(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                        estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> None:
        try:
            image, label = self.generator.generate_chart_image(ticker, start_date, end_date, estimation_start, estimation_end)
            
            filename = f'{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_{label}.png'
            save_dir = os.path.join(self.generator.config.img_save_dir, str(self.generator.config.intervals))
            save_path = os.path.join(save_dir, filename)
            
            os.makedirs(save_dir, exist_ok=True)
            image.save(save_path)
            
        except Exception as e:
            print(f'Failed to save image for {ticker}: {e}')
            raise


class ChartBatchProcessor:
    def __init__(self, chart_generator: ChartGenerator) -> None:
        self.generator = chart_generator

    def generate_batch_dataset(self, tickers: Optional[List[str]] = None) -> None:
        all_tickers = self.generator.market_data.close.columns
        selected_tickers = tickers if tickers is not None else all_tickers
        
        print(f"Processing {len(selected_tickers)} tickers with {self.generator.config.intervals}-day intervals.")
        if self.generator.ma_windows:
            print(f"Using MA windows: {self.generator.ma_windows}")
        
        metadata_list = []
        image_counter = 0
        
        total_combinations = 0
        ticker_windows: Dict[str, Tuple[pd.Index, int]] = {}
        for ticker in selected_tickers:
            ticker_dates = self.generator.get_valid_start_dates(ticker)
            max_idx = len(ticker_dates) - (2 * self.generator.config.intervals) - 1
            total_combinations += max(0, max_idx)
            ticker_windows[ticker] = (ticker_dates, max_idx)
        
        print(f"Total chart combinations to process: {total_combinations}")

        if total_combinations <= 0:
            print("No chart combinations were found for the given configuration.")
            return
        
        save_dir = os.path.join(self.generator.config.img_save_dir, str(self.generator.config.intervals))
        os.makedirs(save_dir, exist_ok=True)
        images_filename = os.path.join(save_dir, f'images_{self.generator.config.intervals}d.npy')
        temp_images_filename = images_filename + '.tmp'
        
        if os.path.exists(temp_images_filename):
            os.remove(temp_images_filename)
        
        image_height = self.generator.config.image_height
        image_width = self.generator.config.intervals * 3
        
        images_mmap: Optional[np.memmap] = None
        try:
            images_mmap = open_memmap(
                temp_images_filename,
                mode='w+',
                dtype=np.uint8,
                shape=(total_combinations, image_height, image_width)
            )
            
            pbar = tqdm(total=total_combinations, desc="Generating charts")
            try:
                for ticker in selected_tickers:
                    ticker_dates, max_idx = ticker_windows[ticker]
                    
                    for i, start_date in enumerate(ticker_dates):
                        if i < max_idx:
                            end_date = ticker_dates[i + self.generator.config.intervals - 1]
                            estimation_start = ticker_dates[i + self.generator.config.intervals]
                            estimation_end = ticker_dates[i + self.generator.config.intervals + self.generator.config.intervals]
                            
                            try:
                                image, label = self.generator.generate_chart_image(ticker, start_date, end_date, estimation_start, estimation_end)
                                
                                images_mmap[image_counter] = np.asarray(image, dtype=np.uint8)
                                
                                metadata_list.append({
                                    'ticker': ticker,
                                    'start_date': start_date.strftime('%Y%m%d'),
                                    'end_date': end_date.strftime('%Y%m%d'),
                                    'estimation_start': estimation_start.strftime('%Y%m%d'),
                                    'estimation_end': estimation_end.strftime('%Y%m%d'),
                                    'label': label,
                                    'image_idx': image_counter
                                })
                                image_counter += 1
                                
                            except Exception as e:
                                print(f"SKIP: {ticker} {start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')} | {str(e)}")
                            finally:
                                pbar.update(1)
            finally:
                pbar.close()
            
            if image_counter == 0:
                print("No charts were successfully generated.")
                return
            
            images_mmap.flush()
            with open(images_filename, 'wb') as out_f:
                np.save(out_f, images_mmap[:image_counter], allow_pickle=False)
        
        finally:
            if images_mmap is not None:
                del images_mmap
            if os.path.exists(temp_images_filename):
                os.remove(temp_images_filename)
        
        metadata_df = pd.DataFrame(metadata_list)
        metadata_filename = os.path.join(save_dir, f'charts_{self.generator.config.intervals}d_metadata.feather')
        metadata_df.to_feather(metadata_filename)
        
        print(f"Saved {image_counter} charts to {save_dir}")
        print(f"Images: {images_filename}")
        print(f"Metadata: {metadata_filename}")
        print(f"Image shape per chart: {self.generator.config.intervals * 3} x {self.generator.config.image_height}")

    def load_batch_dataset(self, intervals: int, tickers: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        save_dir = os.path.join(self.generator.config.img_save_dir, str(intervals))
        
        images_filename = os.path.join(save_dir, f'images_{intervals}d.npy')
        metadata_filename = os.path.join(save_dir, f'charts_{intervals}d_metadata.feather')
        
        if not os.path.exists(images_filename) or not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Batch files not found in {save_dir}")
            
        print(f"Loading metadata from {metadata_filename}")
        metadata_df = pd.read_feather(metadata_filename)
        
        if tickers is not None:
            ticker_mask = metadata_df['ticker'].isin(tickers)
            metadata_df = metadata_df[ticker_mask].reset_index(drop=True)
            if len(metadata_df) == 0:
                raise ValueError(f"No data found for tickers: {tickers}")
        
        print(f"Loading {len(metadata_df)} images from {images_filename} via np.load (mmap)")
        all_images = np.load(images_filename, mmap_mode='r')
        total_images = all_images.shape[0]
        
        if tickers is not None:
            image_indices = metadata_df['image_idx'].values
            max_idx = image_indices.max()
            if max_idx >= total_images:
                raise ValueError(f"Image index {max_idx} exceeds available images {total_images}")
            images = all_images[image_indices]
            metadata_df = metadata_df.copy()
            metadata_df['image_idx'] = range(len(metadata_df))
        else:
            images = all_images[:len(metadata_df)]
        
        print(f"Loaded images shape: {images.shape}")
        print(f"Loaded metadata shape: {metadata_df.shape}")
        return images, metadata_df


class GenerateImages:
    def __init__(self, 
                 o_data: pd.DataFrame, 
                 l_data: pd.DataFrame, 
                 h_data: pd.DataFrame, 
                 c_data: pd.DataFrame, 
                 v_data: pd.DataFrame, 
                 **kwargs) -> None:
        
        market_data = MarketData(
            open=o_data,
            low=l_data, 
            high=h_data,
            close=c_data,
            volume=v_data
        )
        
        config = ChartConfig(**kwargs)
        self.generator = ChartGenerator(market_data, config)
        self.debugger = ChartDebugger(self.generator)
        self.batch_processor = ChartBatchProcessor(self.generator)

    def generate_image_files_batch(self, tickers: Optional[List[str]] = None) -> None:
        self.batch_processor.generate_batch_dataset(tickers=tickers)

    def load_batch_data(self, intervals: int, tickers: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        return self.batch_processor.load_batch_dataset(intervals, tickers=tickers)


def run_batch(frequencies: List[int] = [5, 20, 60], 
              ma_windows_map: Optional[Dict[int, Tuple[int, ...]]] = None):
    print("=== Running Batch Chart Generation ===")
    
    data_dir = DATA_ROOT
    loader = DataLoader(data_dir=str(data_dir))
    print("Available datasets:", loader.available())

    def load_df(name: str) -> pd.DataFrame:
        return loader.to_pandas(loader.load(name))

    open_data = load_df("open")
    low_data = load_df("low") 
    high_data = load_df("high")
    close_data = load_df("close")
    volume_data = load_df("volume")
    
    for freq in frequencies:
        print(f"\n=== Processing {freq}-day frequency ===")
        
        freq_ma_windows = ma_windows_map.get(freq) if ma_windows_map else None
        config = ChartConfig(
            intervals=freq,
            image_height=32 if freq == 5 else 64 if freq == 20 else 96,
            include_ma=True,
            ma_windows=freq_ma_windows,
            include_volume=True,
            img_save_dir=str(IMAGES_ROOT)
        )
        
        generator = GenerateImages(
            o_data=open_data,
            l_data=low_data,
            h_data=high_data,
            c_data=close_data,
            v_data=volume_data,
            **config.__dict__
        )
        
        print(f"Generating {freq}-day charts for all tickers...")
        generator.generate_image_files_batch()
    
    print("\n=== Batch generation completed ===")
    print("All chart datasets have been generated and saved.")


if __name__ == "__main__":
    ma_windows_map = {
        5: (5, 20, 60),
        20: (5, 20, 60),
        60: (5, 20, 60),
    }
    run_batch(frequencies=[20, 60], ma_windows_map=ma_windows_map)
