import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from core.loader import DataLoader


Array = np.ndarray


@dataclass(frozen=True)
class RawTensorConfig:
    """Configuration for building raw OHLCV transformer tensors."""

    data_dir: Path = Path(__file__).resolve().parents[1] / "DATA"
    output_path: Optional[Path] = None
    lookback: int = 60
    stride: int = 5
    horizon: int = 20
    min_assets: int = 50
    min_valid_ratio: float = 0.95
    normalization: Literal["asset", "cross", "none"] = "asset"
    features: Tuple[str, ...] = ("log_ret", "hl_spread", "oc_gap", "volume_z")
    volume_window: int = 20
    zero_as_invalid: bool = False

    def __post_init__(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative.")
        if self.min_assets <= 0:
            raise ValueError("min_assets must be positive.")
        ratio = float(self.min_valid_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("min_valid_ratio must be in (0, 1].")
        object.__setattr__(self, "min_valid_ratio", ratio)


@dataclass(frozen=True)
class RawWindows:
    data: Array  # (num_windows, lookback, num_assets, num_features)
    mask: Array  # (num_windows, num_assets) bool mask for assets
    targets: Array  # (num_windows, num_assets)
    dates: Array  # (num_windows,)
    assets: Tuple[str, ...]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data": self.data,
                "mask": self.mask,
                "targets": self.targets,
                "dates": self.dates,
                "assets": self.assets,
            },
            path,
        )
        return path


class RawWindowBuilder:
    """Convert OHLCV parquet files into tensor windows for Transformer models."""

    def __init__(self, cfg: RawTensorConfig) -> None:
        self.cfg = cfg

    def run(self) -> RawWindows:
        frames = self._load_frames()
        panel, valid_mask, dates, assets = self._build_feature_panel(frames)
        windows = self._build_windows(panel, valid_mask, frames["close"], dates, assets)
        output_path = self._resolve_output_path()
        windows.save(output_path)
        print(f"[transformer] saved windows to {output_path}")
        return windows

    def _resolve_output_path(self) -> Path:
        if self.cfg.output_path is not None:
            return Path(self.cfg.output_path)
        default_dir = Path(__file__).resolve().parent / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"windows_lb{self.cfg.lookback}_hz{self.cfg.horizon}.pt"
        return default_dir / file_name

    def _load_frames(self) -> Dict[str, pd.DataFrame]:
        loader = DataLoader(str(self.cfg.data_dir))
        required = ["open", "high", "low", "close", "volume"]
        frames: Dict[str, pd.DataFrame] = {}
        base_index: Optional[pd.Index] = None
        base_columns: Optional[pd.Index] = None

        excluded = set(loader.get_excluded_tickers())

        for name in required:
            df = loader.to_pandas(loader.load(name))
            if base_index is None:
                base_index = df.index
            if base_columns is None:
                base_columns = df.columns

            df = df.reindex(index=base_index, columns=base_columns)
            if excluded:
                df = df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")
            frames[name] = df.astype(np.float64)

        assert base_index is not None and base_columns is not None
        return frames

    def _build_feature_panel(
        self, frames: Dict[str, pd.DataFrame]
    ) -> Tuple[Array, Array, Array, Tuple[str, ...]]:
        close = frames["close"]
        high = frames["high"]
        low = frames["low"]
        open_ = frames["open"]
        volume = frames["volume"]

        prev_close = close.shift(1)
        log_ret = np.log(close / prev_close)

        hl_spread = (high - low) / close.replace(0.0, np.nan)
        oc_gap = (open_ - prev_close) / prev_close

        vol_mean = volume.rolling(self.cfg.volume_window, min_periods=1).mean()
        volume_z = (volume / vol_mean.replace(0.0, np.nan)) - 1.0

        feature_map = {
            "log_ret": log_ret,
            "hl_spread": hl_spread,
            "oc_gap": oc_gap,
            "volume_z": volume_z,
        }

        arrays: List[Array] = []
        for name in self.cfg.features:
            if name not in feature_map:
                raise ValueError(f"Unknown feature '{name}'. Available: {list(feature_map)}")
            arrays.append(feature_map[name].to_numpy(copy=True))

        panel = np.stack(arrays, axis=-1)
        panel = self._apply_normalization(panel)

        valid_mask = np.isfinite(panel).all(axis=2)
        if self.cfg.zero_as_invalid:
            valid_mask &= panel.any(axis=2)

        panel = np.nan_to_num(panel, nan=0.0).astype(np.float32, copy=False)
        dates = close.index.to_numpy()
        assets = tuple(close.columns.astype(str))

        return panel, valid_mask, dates, assets

    def _apply_normalization(self, panel: Array) -> Array:
        norm = self.cfg.normalization.lower()
        if norm == "none":
            return panel

        data = panel.copy()
        if norm == "asset":
            mean = np.nanmean(data, axis=0, keepdims=True)
            std = np.nanstd(data, axis=0, keepdims=True)
        elif norm == "cross":
            mean = np.nanmean(data, axis=1, keepdims=True)
            std = np.nanstd(data, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown normalization '{self.cfg.normalization}'.")

        std = np.where(std == 0.0, 1.0, std)
        return (data - mean) / std

    def _build_windows(
        self,
        panel: Array,
        valid_mask: Array,
        close_df: pd.DataFrame,
        dates: Array,
        assets: Tuple[str, ...],
    ) -> RawWindows:
        lookback = self.cfg.lookback
        stride = self.cfg.stride
        horizon = self.cfg.horizon
        min_valid = int(np.ceil(lookback * self.cfg.min_valid_ratio))

        close_values = close_df.to_numpy()
        future_returns = (close_df.shift(-horizon) / close_df) - 1.0
        future_values = future_returns.to_numpy()

        num_steps = panel.shape[0]
        max_start = num_steps - lookback - horizon
        if max_start <= 0:
            raise ValueError("Not enough timesteps to build any window with current config.")

        window_list: List[Array] = []
        mask_list: List[Array] = []
        target_list: List[Array] = []
        date_list: List[np.datetime64] = []

        for start in range(0, max_start + 1, stride):
            end = start + lookback
            target_idx = end - 1

            window_valid = valid_mask[start:end]
            asset_valid_counts = window_valid.sum(axis=0)
            asset_mask = asset_valid_counts >= min_valid

            future_row = future_values[target_idx]
            base_row = close_values[target_idx]
            future_valid = np.isfinite(future_row) & np.isfinite(base_row)
            asset_mask &= future_valid

            if asset_mask.sum() < self.cfg.min_assets:
                continue

            window = panel[start:end].copy()
            window_list.append(window)
            mask_list.append(asset_mask.astype(bool, copy=False))
            target_list.append(future_row.astype(np.float32, copy=False))
            date_list.append(dates[target_idx])

        if not window_list:
            raise RuntimeError("No windows were generated. Check data quality or config.")

        data = np.stack(window_list, axis=0).astype(np.float32, copy=False)
        mask = np.stack(mask_list, axis=0).astype(bool, copy=False)
        targets = np.stack(target_list, axis=0).astype(np.float32, copy=False)
        date_array = np.asarray(date_list)

        return RawWindows(data=data, mask=mask, targets=targets, dates=date_array, assets=assets)


if __name__ == "__main__":
    builder = RawWindowBuilder(RawTensorConfig())
    builder.run()
