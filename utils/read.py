import os
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.root import IMAGES_ROOT


class ChartViewer:
    def __init__(self, intervals: int) -> None:
        self.intervals = intervals
        self.base_dir = str(IMAGES_ROOT)
        self.image_height = self._get_image_height(intervals)
        self.image_width = intervals * 3
        self.image_shape = (self.image_height, self.image_width)

        self.metadata: Optional[pd.DataFrame] = None
        self.images: Optional[np.ndarray] = None

    def _get_image_height(self, intervals: int) -> int:
        if intervals == 5:
            return 32
        if intervals == 20:
            return 64
        if intervals == 60:
            return 96
        raise ValueError(f"Unsupported interval value: {intervals}. Supported values are 5, 20, 60.")

    def load_data(self) -> None:
        save_dir = os.path.join(self.base_dir, str(self.intervals))
        metadata_filename = os.path.join(save_dir, f'charts_{self.intervals}d_metadata.feather')
        npy_filename = os.path.join(save_dir, f'images_{self.intervals}d.npy')

        if not os.path.exists(metadata_filename):
            raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
        
        if not os.path.exists(npy_filename):
            raise FileNotFoundError(f"NPY file not found: {npy_filename}. Run convert.py first.")

        print(f"Loading metadata from: {metadata_filename}")
        self.metadata = pd.read_feather(metadata_filename)

        print(f"Memory-mapping images from: {npy_filename}")
        self.images = np.load(npy_filename, mmap_mode='r')
        print(f"Loaded {self.images.shape[0]} images with shape {self.images.shape[1:]}.")

    def display_charts(self, ticker: str, chart_numbers: Optional[List[int]] = None) -> None:
        if self.metadata is None or self.images is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        ticker_metadata = self.metadata[self.metadata['ticker'] == ticker].reset_index(drop=True)
        if ticker_metadata.empty:
            print(f"No charts found for ticker '{ticker}' with interval {self.intervals}.")
            all_tickers = self.metadata['ticker'].unique()
            print(f"Available tickers: {', '.join(all_tickers[:10])}...")
            return

        num_charts = len(ticker_metadata)

        if not chart_numbers:
            print(f"Found {num_charts} charts for ticker '{ticker}'.")
            print(f"Please specify chart numbers to view, e.g., chart_numbers=[1, 2, 3].")
            return
        
        print(f"Displaying {len(chart_numbers)} specified charts for '{ticker}'...")

        for chart_number in chart_numbers:
            if not (1 <= chart_number <= num_charts):
                print(f"\n--- \nWarning: Chart number {chart_number} is out of range (1-{num_charts}). Skipping.")
                continue

            row = ticker_metadata.iloc[chart_number - 1]
            
            print("\n" + "="*50)
            print(f"Metadata for Chart #{chart_number}/{num_charts}")
            print(f"  - Ticker:            {row.ticker}")
            print(f"  - Chart Period:      {row.start_date} to {row.end_date}")
            print(f"  - Estimation Period: {row.estimation_start} to {row.estimation_end}")
            print(f"  - Label:             {'Up' if row.label == 1 else 'Down'}")
            print(f"  - Internal Image Idx:  {row.image_idx}")
            print("="*50)
            
            img_array = self.images[row.image_idx]
            
            plt.figure(figsize=(10, 5))
            plt.imshow(img_array, cmap='gray', aspect='auto')
            
            plt.title(f"Ticker: {row.ticker} - Chart #{chart_number}")
            plt.axis('off')
            plt.show()

def read_charts(
    ticker: str, intervals: int, chart_numbers: Optional[List[int]] = None
) -> None:
    try:
        viewer = ChartViewer(intervals=intervals)
        viewer.load_data()
        viewer.display_charts(ticker=ticker, chart_numbers=chart_numbers)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}")

def load_full_dataset(intervals: int) -> tuple[np.ndarray, pd.DataFrame]:
    save_dir = IMAGES_ROOT / str(intervals)
    
    metadata_filename = save_dir / f'charts_{intervals}d_metadata.feather'
    npy_filename = save_dir / f'images_{intervals}d.npy'
    
    if not os.path.exists(metadata_filename):
        raise FileNotFoundError(f"Metadata file not found: {metadata_filename}")
    
    if not os.path.exists(npy_filename):
        raise FileNotFoundError(f"NPY file not found: {npy_filename}")
    
    print(f"Loading from: {save_dir}")
    print("-" * 50)
    
    metadata = pd.read_feather(metadata_filename)
    print(f"Metadata shape: {metadata.shape}")
    print(f"Metadata columns: {metadata.columns.tolist()}")
    print(f"Unique tickers: {metadata['ticker'].nunique()}")
    print(f"Label distribution: {metadata['label'].value_counts().to_dict()}")
    print()
    
    images = np.load(npy_filename, mmap_mode='r')
    print(f"Images shape: {images.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Memory usage: {images.nbytes / 1024**2:.1f} MB")
    
    if len(images.shape) == 3:
        print(f"Format: 3D (num_images, height, width)")
        print(f"Image dimensions: {images.shape[1]} x {images.shape[2]}")
    elif len(images.shape) == 2:
        print(f"Format: 2D (num_images, flattened_pixels)")
        expected_height = 32 if intervals == 5 else 64 if intervals == 20 else 96
        expected_width = intervals * 3
        print(f"Expected reshape: ({expected_height}, {expected_width})")
    
    print("-" * 50)
    return images, metadata

if __name__ == "__main__":
    read_charts(ticker="A005930", 
                intervals=60, 
                chart_numbers=[270, 6189])
    
    # images, metadata = load_full_dataset(intervals=5)
