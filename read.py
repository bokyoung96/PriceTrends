import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List


class ChartViewer:
    def __init__(self, intervals: int, base_dir: str = 'Images') -> None:
        self.intervals = intervals
        self.base_dir = os.path.join(os.path.dirname(__file__), base_dir)
        self.image_height = self._get_image_height(intervals)
        self.image_width = intervals * 3
        self.image_shape = (self.image_height, self.image_width)
        self.image_size_bytes = self.image_width * self.image_height

        self.metadata: Optional[pd.DataFrame] = None
        self.images_memmap: Optional[np.ndarray] = None

    def _get_image_height(self, intervals: int) -> int:
        if intervals == 5:
            return 32
        if intervals == 20:
            return 64
        if intervals == 60:
            return 96
        raise ValueError(
            f"Unsupported interval value: {intervals}. Supported values are 5, 20, 60.")

    def load_batch_data(self) -> None:
        save_dir = os.path.join(self.base_dir, str(self.intervals))
        metadata_filename = os.path.join(
            save_dir, f'charts_{self.intervals}d_metadata.feather')
        images_filename = os.path.join(
            save_dir, f'charts_{self.intervals}d_images.dat')

        if not os.path.exists(metadata_filename) or not os.path.exists(images_filename):
            raise FileNotFoundError(
                f"Data files not found in {save_dir}. Please run the image generation first.")

        print(f"Loading metadata from: {metadata_filename}")
        self.metadata = pd.read_feather(metadata_filename)

        print(f"Memory-mapping images from: {images_filename}")
        self.images_memmap = np.memmap(
            images_filename, dtype=np.uint8, mode='r')

    def display_batch_charts(self, ticker: str, chart_numbers: Optional[List[int]] = None) -> None:
        if self.metadata is None or self.images_memmap is None:
            raise RuntimeError(
                "Data not loaded. Call load_batch_data() first.")

        ticker_metadata = self.metadata[self.metadata['ticker'] == ticker].reset_index(
            drop=True)
        if ticker_metadata.empty:
            print(
                f"No charts found for ticker '{ticker}' with interval {self.intervals}.")
            all_tickers = self.metadata['ticker'].unique()
            print(f"Available tickers: {', '.join(all_tickers[:10])}...")
            return

        num_charts = len(ticker_metadata)

        if not chart_numbers:
            print(f"Found {num_charts} charts for ticker '{ticker}'.")
            print(
                f"Please specify chart numbers to view, e.g., chart_numbers=[1, 2, 3].")
            return

        print(
            f"Displaying {len(chart_numbers)} specified charts for '{ticker}'...")

        for chart_number in chart_numbers:
            if not (1 <= chart_number <= num_charts):
                print(
                    f"\n--- \nWarning: Chart number {chart_number} is out of range (1-{num_charts}). Skipping.")
                continue

            row = ticker_metadata.iloc[chart_number - 1]

            print("\n" + "="*50)
            print(f"Metadata for Chart #{chart_number}/{num_charts}")
            print(f"  - Ticker:            {row.ticker}")
            print(f"  - Chart Period:      {row.start_date} to {row.end_date}")
            print(
                f"  - Estimation Period: {row.estimation_start} to {row.estimation_end}")
            print(
                f"  - Label:             {'Up' if row.label == 1 else 'Down'}")
            print(f"  - Internal Image Idx:  {row.image_idx}")
            print("="*50)

            start_offset = row.image_idx * self.image_size_bytes
            end_offset = start_offset + self.image_size_bytes

            img_array_flat = self.images_memmap[start_offset:end_offset]
            if img_array_flat.size != self.image_size_bytes:
                print(
                    f"Warning: Incomplete image data for index {row.image_idx}")
                continue

            img_array = img_array_flat.reshape(self.image_shape)

            plt.figure(figsize=(10, 5))
            plt.imshow(img_array, cmap='gray', aspect='auto')

            plt.title(f"Ticker: {row.ticker} - Chart #{chart_number}")
            plt.axis('off')
            plt.show()


def read_batch_charts(ticker: str,
                      intervals: int,
                      chart_numbers: Optional[List[int]] = None) -> None:
    try:
        viewer = ChartViewer(intervals=intervals)
        viewer.load_batch_data()
        viewer.display_batch_charts(ticker=ticker,
                                    chart_numbers=chart_numbers)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    read_batch_charts(ticker="A005930",
                      intervals=5,
                      chart_numbers=[100, 101, 150])
