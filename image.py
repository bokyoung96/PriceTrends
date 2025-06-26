import os
import random
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
from PIL import Image, ImageDraw
from tqdm import tqdm
from loader import DataLoader


@dataclass(frozen=True)
class ChartConfig:
    intervals: int = 5
    image_height: int = 32
    include_ma: bool = True
    include_volume: bool = True
    background_color: int = 0
    chart_color: int = 255
    volume_chart_gap: int = 1
    img_save_dir: str = os.path.join(os.path.dirname(__file__), 'Images')


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
        self._ma_data: Optional[pd.DataFrame] = None

    @property
    def ma_data(self) -> pd.DataFrame:
        if self._ma_data is None:
            self._ma_data = self.market_data.close.rolling(
                window=self.config.intervals).mean()
        return self._ma_data

    def _extract_arrays(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> np.ndarray:
        date_slice = slice(start_date, end_date)

        arrays = [
            self.market_data.open.loc[date_slice, ticker].values,
            self.market_data.low.loc[date_slice, ticker].values,
            self.market_data.high.loc[date_slice, ticker].values,
            self.market_data.close.loc[date_slice, ticker].values,
        ]

        if self.config.include_ma:
            arrays.append(self.ma_data.loc[date_slice, ticker].values)

        if self.config.include_volume:
            arrays.append(
                self.market_data.volume.loc[date_slice, ticker].values)
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
        scaled_prices = (normalized_prices - min_price) / \
            (max_price - min_price) * (price_height - 1) + offset

        return scaled_prices

    def _normalize_volume_data(self, volume_array: np.ndarray, volume_height: int) -> np.ndarray:
        max_volume = np.nanmax(volume_array)
        if max_volume == 0 or np.isnan(max_volume) or max_volume <= 0:
            return np.zeros_like(volume_array)

        normalized_volume = (volume_array / max_volume) * (volume_height - 1)
        return np.nan_to_num(normalized_volume, nan=0.0)

    def _create_base_image(self, total_height: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("L", (self.config.intervals * 3,
                          total_height), self.config.background_color)
        return image, ImageDraw.Draw(image)

    def _draw_ohlc_chart(self, draw: ImageDraw.ImageDraw, price_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length):
            draw.point((i * 3, price_data[0][i]), fill=self.config.chart_color)
            draw.point(
                (i * 3 + 2, price_data[3][i]), fill=self.config.chart_color)
            draw.line((i * 3 + 1, price_data[1][i], i * 3 + 1,
                      price_data[2][i]), fill=self.config.chart_color)

    def _draw_ma_line(self, draw: ImageDraw.ImageDraw, ma_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length - 1):
            draw.line(
                (i * 3, ma_data[i], (i + 1) * 3, ma_data[i + 1]), fill=self.config.chart_color)

    def _draw_volume_bars(self, draw: ImageDraw.ImageDraw, volume_data: np.ndarray, data_length: int) -> None:
        for i in range(data_length):
            draw.line((i * 3 + 1, 0, i * 3 + 1,
                      volume_data[i]), fill=self.config.chart_color)

    def _calculate_label(self, ticker: str, estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> int:
        start_price = self.market_data.close.loc[estimation_start, ticker]
        end_price = self.market_data.close.loc[estimation_end, ticker]
        return 1 if (end_price / start_price) > 1 else 0

    def generate_chart_image(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                             estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> Tuple[Image.Image, int]:
        try:
            arrays = self._extract_arrays(ticker, start_date, end_date)

            price_arrays_to_check = arrays[:5] if self.config.include_ma and len(
                arrays) > 4 else arrays[:4]
            if np.isnan(price_arrays_to_check).any():
                raise ValueError(
                    "Price or MA data contains NaN, skipping chart.")

            data_length = len(
                self.market_data.close.loc[start_date:end_date].index)

            total_height, price_height, volume_height = self._calculate_heights()

            price_arrays_to_normalize = arrays[:5] if self.config.include_ma and len(
                arrays) > 4 else arrays[:4]
            normalized_data = self._normalize_price_data(
                price_arrays_to_normalize, price_height, volume_height).astype(int)
            normalized_prices = normalized_data[:4]

            image, draw = self._create_base_image(total_height)

            self._draw_ohlc_chart(draw, normalized_prices, data_length)

            if self.config.include_ma and len(normalized_data) > 4:
                normalized_ma = normalized_data[4]
                self._draw_ma_line(draw, normalized_ma, data_length)

            if self.config.include_volume:
                volume_data = self._normalize_volume_data(
                    arrays[-1], volume_height).astype(int)
                self._draw_volume_bars(draw, volume_data, data_length)

            final_image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = self._calculate_label(
                ticker, estimation_start, estimation_end)
            return final_image, label

        except Exception as e:
            raise RuntimeError(f"Failed to generate chart for {ticker}: {e}")


class ChartDebugger:
    def __init__(self, chart_generator: ChartGenerator) -> None:
        self.generator = chart_generator

    def save_chart_image(self, ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
                         estimation_start: pd.Timestamp, estimation_end: pd.Timestamp) -> None:
        try:
            image, label = self.generator.generate_chart_image(
                ticker, start_date, end_date, estimation_start, estimation_end)

            filename = f'{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_{label}.png'
            save_dir = os.path.join(self.generator.config.img_save_dir, str(
                self.generator.config.intervals))
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

        print(
            f"Processing {len(selected_tickers)} tickers with {self.generator.config.intervals}-day intervals.")

        metadata_list = []
        image_counter = 0

        ticker_date_ranges = {}
        total_combinations = 0
        for ticker in selected_tickers:
            ticker_dates = self.generator.ma_data[ticker].dropna().index
            max_idx = len(ticker_dates) - \
                (2 * self.generator.config.intervals) - 1
            if max_idx > 0:
                ticker_date_ranges[ticker] = (ticker_dates, max_idx)
                total_combinations += max_idx

        print(f"Total chart combinations to process: {total_combinations}")

        if total_combinations == 0:
            print("No valid chart combinations found to process.")
            return

        save_dir = os.path.join(self.generator.config.img_save_dir, str(
            self.generator.config.intervals))
        os.makedirs(save_dir, exist_ok=True)
        images_filename = os.path.join(
            save_dir, f'images_{self.generator.config.intervals}d.npy')

        image_height = self.generator.config.image_height
        image_width = self.generator.config.intervals * 3
        image_size = image_height * image_width

        images_array = np.memmap(
            images_filename, dtype=np.uint8, mode='w+', shape=(total_combinations, image_size))

        pbar = tqdm(total=total_combinations,
                    desc="Generating and saving charts")

        for ticker in selected_tickers:
            if ticker not in ticker_date_ranges:
                continue

            ticker_dates, max_idx = ticker_date_ranges[ticker]

            for i in range(max_idx):
                start_date = ticker_dates[i]
                end_date = ticker_dates[i +
                                        self.generator.config.intervals - 1]
                estimation_start = ticker_dates[i +
                                                self.generator.config.intervals]
                estimation_end = ticker_dates[i +
                                              self.generator.config.intervals + self.generator.config.intervals]

                try:
                    image, label = self.generator.generate_chart_image(
                        ticker, start_date, end_date, estimation_start, estimation_end)

                    images_array[image_counter] = np.array(image).flatten()

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
                    print(
                        f"SKIP: {ticker} {start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')} | {str(e)}")
                finally:
                    pbar.update(1)

        pbar.close()

        images_array.flush()
        del images_array

        if image_counter == 0:
            print("No charts were successfully generated.")
            if os.path.exists(images_filename):
                os.remove(images_filename)
            return

        if image_counter < total_combinations:
            print(f"Warning: Generated {image_counter} of {total_combinations} expected charts. "
                  "The .npy file on disk is oversized, but this is handled correctly during loading.")

        metadata_df = pd.DataFrame(metadata_list)
        metadata_filename = os.path.join(
            save_dir, f'charts_{self.generator.config.intervals}d_metadata.feather')

        with tqdm(total=1, desc="Saving metadata") as pbar_meta:
            metadata_df.to_feather(metadata_filename)
            pbar_meta.update(1)

        print(f"\nSaved {image_counter} charts to {save_dir}")
        print(f"Images: {images_filename}")
        print(f"Metadata: {metadata_filename}")
        print(
            f"Image shape per chart: {image_width} x {image_height}")

    def load_batch_dataset(self, intervals: int, tickers: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        save_dir = os.path.join(
            self.generator.config.img_save_dir, str(intervals))

        images_filename = os.path.join(save_dir, f'images_{intervals}d.npy')
        metadata_filename = os.path.join(
            save_dir, f'charts_{intervals}d_metadata.feather')

        if not os.path.exists(images_filename) or not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Batch files not found in {save_dir}")

        print(f"Loading metadata from {metadata_filename}")
        metadata_df = pd.read_feather(metadata_filename)

        if tickers is not None:
            ticker_mask = metadata_df['ticker'].isin(tickers)
            metadata_df = metadata_df[ticker_mask].reset_index(drop=True)
            if len(metadata_df) == 0:
                raise ValueError(f"No data found for tickers: {tickers}")

        if len(metadata_df) == 0:
            print("No images to load based on metadata.")
            return np.array([]), pd.DataFrame()

        image_height = self.generator.config.image_height
        image_width = intervals * 3
        image_size = image_width * image_height

        num_images_from_meta = len(metadata_df)

        print(
            f"Loading {num_images_from_meta} images from {images_filename} based on metadata.")

        all_images = np.memmap(
            images_filename,
            dtype=np.uint8,
            mode='r',
            shape=(num_images_from_meta, image_size)
        )

        if tickers is not None:
            image_indices = metadata_df['image_idx'].values

            original_memmap = np.memmap(
                images_filename,
                dtype=np.uint8,
                mode='r'
            ).reshape(-1, image_size)

            images = original_memmap[image_indices]

            metadata_df = metadata_df.copy()
            metadata_df['image_idx'] = range(len(metadata_df))
        else:
            images = all_images

        images = images.reshape(-1, image_height, image_width)
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


def run_batch(frequencies: List[int] = [5, 20, 60]):
    print("=== Running Batch Chart Generation ===")

    loader = DataLoader(data_dir=os.path.join(
        os.path.dirname(__file__), "DATA"))
    print("Available datasets:", loader.available())

    open_data = loader.load("open")
    low_data = loader.load("low")
    high_data = loader.load("high")
    close_data = loader.load("close")
    volume_data = loader.load("volume")

    for freq in frequencies:
        print(f"\n=== Processing {freq}-day frequency ===")

        config = ChartConfig(
            intervals=freq,
            image_height=32 if freq == 5 else 64 if freq == 20 else 96,
            include_ma=True,
            include_volume=True,
            img_save_dir=os.path.join(os.path.dirname(__file__), 'Images')
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


def run_single(ticker: str, intervals: int = 20):
    print(
        f"=== Generating Single Random Date Chart: {ticker} (random date) ===")

    loader = DataLoader(data_dir=os.path.join(
        os.path.dirname(__file__), "DATA"))

    open_data = loader.load("open")
    low_data = loader.load("low")
    high_data = loader.load("high")
    close_data = loader.load("close")
    volume_data = loader.load("volume")

    if ticker not in close_data.columns:
        available_tickers = close_data.columns.tolist()[:10]
        raise ValueError(
            f"Ticker '{ticker}' not found. Available tickers (first 10): {available_tickers}")

    config = ChartConfig(
        intervals=intervals,
        image_height=32 if intervals == 5 else 64 if intervals == 20 else 96,
        include_ma=True,
        include_volume=True,
        img_save_dir=os.path.join(os.path.dirname(__file__), 'Images')
    )

    generator = GenerateImages(
        o_data=open_data,
        l_data=low_data,
        h_data=high_data,
        c_data=close_data,
        v_data=volume_data,
        **config.__dict__
    )

    ticker_data = close_data[ticker].dropna()
    available_dates = ticker_data.index
    min_required_days = intervals * 3
    if len(available_dates) < min_required_days:
        raise ValueError(
            f"Not enough data for {ticker}. Need at least {min_required_days} days, got {len(available_dates)}")

    valid_start_indices = list(range(len(available_dates) - min_required_days))
    if len(valid_start_indices) == 0:
        raise ValueError(f"No valid start dates for {ticker}")

    start_idx = random.choice(valid_start_indices)
    start_date = available_dates[start_idx]

    end_idx = start_idx + intervals - 1
    end_date = available_dates[end_idx]

    estimation_start_idx = end_idx + 1
    estimation_end_idx = estimation_start_idx + intervals - 1
    estimation_start = available_dates[estimation_start_idx]
    estimation_end = available_dates[estimation_end_idx]

    try:
        generator.debugger.save_chart_image(
            ticker, start_date, end_date, estimation_start, estimation_end)

        print(f"✓ Chart generated successfully")
        print(f"  Ticker: {ticker}")
        print(
            f"  Chart period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({intervals} trading days)")
        print(
            f"  Estimation period: {estimation_start.strftime('%Y-%m-%d')} to {estimation_end.strftime('%Y-%m-%d')} ({intervals} trading days)")

        try:
            start_price = close_data.loc[estimation_start, ticker]
            end_price = close_data.loc[estimation_end, ticker]
            label = 1 if end_price / start_price > 1 else 0
        except KeyError:
            label = 0

        print(f"  Label: {label} ({'Up' if label == 1 else 'Down'})")

        save_dir = os.path.join(config.img_save_dir, str(intervals))
        filename = f'{ticker}_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}_{label}.png'
        save_path = os.path.join(save_dir, filename)
        print(f"✓ PNG saved to: {save_path}")

        return save_path

    except Exception as e:
        print(f"✗ Error generating chart: {e}")
        raise


if __name__ == "__main__":
    # run_batch(frequencies=[5, 20, 60])
    run_batch(frequencies=[20, 60])
