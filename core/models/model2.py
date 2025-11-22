import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from core.models.model1 import CNNModel, KoreanEquityDataset, init_weights
from core.params import CNNConfig, CNNParams
from core.spec import MarketMetric
from utils.root import DATA_ROOT, MODELS_ROOT


logger = logging.getLogger(__name__)


class ForeignDataset(Dataset):
    def __init__(
        self,
        windows: Sequence[int] = (5, 20, 60),
        data_root: Path | None = None,
        years: Sequence[int] | None = None,
        norm_years: Sequence[int] | None = None,
    ) -> None:
        if not windows:
            raise ValueError("At least one rolling window must be provided.")

        self.windows = tuple(dict.fromkeys(sorted(int(w) for w in windows)))
        self.data_root = Path(data_root) if data_root is not None else DATA_ROOT
        self.years = tuple(dict.fromkeys(int(y) for y in years)) if years is not None else None
        self.norm_years = (
            tuple(dict.fromkeys(int(y) for y in norm_years)) if norm_years is not None else None
        )
        self.window_paths = {
            window: self.data_root / f"FOREIGN_{window}.parquet" for window in self.windows
        }
        self.metric_path = self.data_root / MarketMetric.MKTCAP.parquet_filename

        self.metric_df = self._load_raw_data(self.metric_path)
        self.foreign_frames = {
            window: self._load_raw_data(path) for window, path in self.window_paths.items()
        }

        self.common_tickers = self._find_common_tickers()
        if not self.common_tickers:
            raise ValueError("No common tickers found across foreign metrics and market cap.")

        processed = {
            window: self._get_window_features(df) for window, df in self.foreign_frames.items()
        }
        self.feature_frame = self._stack_features(processed)
        if self.feature_frame.empty:
            raise ValueError("ForeignDataset contains no valid samples after preprocessing.")

        self.sample_index = list(self.feature_frame.index)
        self.feature_matrix = self.feature_frame.astype(np.float32).to_numpy()

    def __getitem__(self, idx: int) -> Dict[str, object]:
        date, ticker = self.sample_index[idx]
        features = torch.tensor(self.feature_matrix[idx], dtype=torch.float32)
        return {
            "foreign_features": features,
            "ticker": ticker,
            "date": pd.Timestamp(date),
        }

    def _load_raw_data(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Required parquet file not found: {path}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def _find_common_tickers(self) -> List[str]:
        years = self.years

        if years is not None:
            year_list = list(years)
            valid: set[str] = set()

            for df in self.foreign_frames.values():
                if df.empty:
                    continue
                year_arr = df.index.year
                mask = pd.Series(year_arr).isin(year_list).to_numpy()
                df_year = df.iloc[mask]
                if df_year.empty:
                    continue
                cols_with_data = df_year.columns[df_year.notna().any()].tolist()
                valid.update(cols_with_data)

            if not valid:
                return []

            metric = self.metric_df
            if not metric.empty:
                year_arr = metric.index.year
                mask = pd.Series(year_arr).isin(year_list).to_numpy()
                metric_year = metric.iloc[mask]
            else:
                metric_year = metric

            if metric_year.empty:
                return []

            metric_cols = set(metric_year.columns)
            valid &= metric_cols
            return sorted(valid)

        # Fallback: union over all years
        valid: set[str] = set()
        for df in self.foreign_frames.values():
            if df.empty:
                continue
            cols_with_data = df.columns[df.notna().any()].tolist()
            valid.update(cols_with_data)
        metric_cols = set(self.metric_df.columns)
        valid &= metric_cols
        return sorted(valid)

    def _get_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.common_tickers if c in df.columns]
        if not cols:
            return pd.DataFrame()

        df = df.loc[:, cols]
        metric = self.metric_df.loc[:, cols]
        metric = metric.reindex(df.index)
        ratio = df.divide(metric)
        ratio = ratio.replace([np.inf, -np.inf], np.nan)
        normalized = self._zscore_cols(ratio)
        if self.years is not None:
            year_arr = normalized.index.year
            mask = pd.Series(year_arr).isin(list(self.years)).to_numpy()
            normalized = normalized.iloc[mask]
        normalized = normalized.dropna(how="all")
        return normalized

    def _zscore_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.norm_years is not None:
            years = df.index.year
            mask = pd.Series(years).isin(self.norm_years).to_numpy()
            train_df = df.iloc[mask]
        else:
            train_df = df
        means = train_df.mean(axis=0)
        stds = train_df.std(axis=0, ddof=0).replace(0, np.nan)
        return (df - means) / stds

    def _stack_features(self, processed: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        stacked_frames = []
        for window in self.windows:
            window_df = processed[window]
            if window_df.empty:
                continue
            stacked = window_df.stack().to_frame(name=f"foreign_{window}")
            stacked.index = stacked.index.set_names(["date", "ticker"])
            stacked_frames.append(stacked)

        if not stacked_frames:
            return pd.DataFrame()

        feature_frame = pd.concat(stacked_frames, axis=1)
        feature_frame = feature_frame.dropna()
        feature_frame.sort_index(inplace=True)
        return feature_frame

    def get_frame(self) -> pd.DataFrame:
        return self.feature_frame.copy()


class ForeignFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        xavier: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)

        if xavier:
            self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class FusionDataset(Dataset):
    def __init__(
        self,
        ws: int,
        years: List[int],
        foreign_windows: Sequence[int] = (5, 20, 60),
        norm_years: Sequence[int] | None = None,
    ) -> None:
        self.ws = ws
        self.years = years

        self.image_dataset = KoreanEquityDataset(intervals=ws, years=years)
        self.foreign_dataset = ForeignDataset(
            windows=foreign_windows,
            years=years,
            norm_years=norm_years,
        )

        foreign_frame = self.foreign_dataset.get_frame()
        self.feature_columns = foreign_frame.columns.tolist()

        meta = self.image_dataset.metadata.copy()
        key_dates = pd.to_datetime(meta["end_date"]).dt.normalize()
        meta["key"] = list(zip(key_dates, meta["ticker"]))

        dates = pd.to_datetime(foreign_frame.index.get_level_values("date")).normalize()
        tickers = foreign_frame.index.get_level_values("ticker")
        keys = list(zip(dates, tickers))

        features_by_key = foreign_frame.copy()
        features_by_key["key"] = keys
        features_by_key.set_index("key", inplace=True)

        merged = meta.join(features_by_key[self.feature_columns], on="key", how="inner")
        if merged.empty:
            raise ValueError("FusionDataset has no samples after joining image and foreign features.")

        self.merged = merged.reset_index(drop=True)
        self.feature_dim = len(self.feature_columns)

        logger.info(
            "FusionDataset initialized with %d samples (ws=%d, years=%s)",
            len(self.merged),
            ws,
            years,
        )

    def __len__(self) -> int:
        return len(self.merged)

    def __getitem__(self, idx: int) -> Dict:
        row = self.merged.iloc[idx]

        original_idx = int(row.image_idx)
        img_array = self.image_dataset.images[original_idx]
        if img_array.dtype != np.float32:
            img_array = img_array.astype(np.float32, copy=False)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)

        features = torch.as_tensor(
            row[self.feature_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        label = torch.tensor(row.label, dtype=torch.long)

        return {
            "image": img_tensor,
            "foreign_features": features,
            "label": label,
            "StockID": row.ticker,
            "ending_date": str(row.end_date),
        }


def get_fusion_dataloaders(
    ws: int,
    train_years: List[int],
    foreign_windows: Sequence[int],
    config: CNNConfig,
    train_ratio: float = 0.7,
) -> Dict[str, DataLoader]:
    full_dataset = FusionDataset(
        ws=ws,
        years=train_years,
        foreign_windows=foreign_windows,
        norm_years=train_years,
    )

    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=1.0 - train_ratio,
        random_state=42,
        stratify=full_dataset.merged["label"],
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    logger.info(
        "Fusion train size: %d (%.0f%%), Validation size: %d (%.0f%%)",
        len(train_dataset),
        train_ratio * 100,
        len(val_dataset),
        (1 - train_ratio) * 100,
    )

    selector = DeviceSelector()
    device = selector.resolve()

    num_workers = getattr(config, "num_workers", 0)
    spawn_like = os.name == "nt"
    if not spawn_like:
        try:
            start_method = mp.get_start_method(allow_none=True)
        except RuntimeError:
            start_method = None
        spawn_like = start_method in (None, "spawn", "forkserver")
    if num_workers > 0 and spawn_like:
        logger.warning(
            "Detected spawn-based multiprocessing; forcing num_workers=0 because numpy.memmap datasets cannot be pickled safely."
        )
        num_workers = 0

    pin_memory = device.type == "cuda"
    dataloader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

    return {"train": train_loader, "validate": val_loader}


class FusionModel(nn.Module):
    def __init__(
        self,
        image_encoder: CNNModel,
        foreign_encoder: ForeignFeatureMLP,
        fusion_hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        lrelu: bool = True,
        xavier: bool = True,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.foreign_encoder = foreign_encoder

        self.fusion_hidden_dims = fusion_hidden_dims
        self.dropout_prob = dropout
        self.lrelu = lrelu

        image_dim = self._get_image_embedding_dim()
        foreign_dim = self._get_foreign_embedding_dim()
        fused_dim = image_dim + foreign_dim

        self.classifier = self._init_classifier(
            input_dim=fused_dim,
            hidden_dims=fusion_hidden_dims,
            dropout=dropout,
            lrelu=lrelu,
        )

        if xavier:
            self.apply(init_weights)

    def _get_image_embedding_dim(self) -> int:
        if not hasattr(self.image_encoder, "conv_layers"):
            raise ValueError("CNNModel must expose `conv_layers` for embedding computation.")
        if not hasattr(self.image_encoder, "input_size"):
            raise ValueError("CNNModel must expose `input_size` for embedding computation.")

        height, width = self.image_encoder.input_size
        dummy_input = torch.rand((1, 1, height, width))
        with torch.no_grad():
            x = self.image_encoder.conv_layers(dummy_input)
            x = x.view(x.size(0), -1)
        return x.shape[1]

    def _get_foreign_embedding_dim(self) -> int:
        last_linear: Optional[nn.Linear] = None
        for module in self.foreign_encoder.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("ForeignFeatureMLP must contain at least one nn.Linear layer.")
        return last_linear.out_features

    def _init_classifier(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
        lrelu: bool,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.01) if lrelu else nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 2))
        return nn.Sequential(*layers)

    def forward(
        self,
        image: torch.Tensor,
        foreign_features: torch.Tensor,
    ) -> torch.Tensor:
        x_image = self.image_encoder.conv_layers(image)
        image_embedding = x_image.view(x_image.size(0), -1)
        foreign_embedding = self.foreign_encoder(foreign_features)

        fused = torch.cat([image_embedding, foreign_embedding], dim=1)
        logits = self.classifier(fused)
        return logits


class FusionTrainer:
    def __init__(
        self,
        ws: int,
        pw: int,
        config: CNNConfig,
        fusion_model: FusionModel,
        exp_suffix: str = "fusion",
    ) -> None:
        self.ws = ws
        self.pw = pw
        self.config = config

        selector = DeviceSelector()
        self.device = selector.resolve()
        logger.info(selector.summary("FusionTrainer"))

        self.model = fusion_model.to(self.device)

        self.exp_name = f"korea_cnn_{ws}d{pw}p_{config.mode}_{exp_suffix}"
        self.model_dir = MODELS_ROOT / self.exp_name
        os.makedirs(self.model_dir, exist_ok=True)

    def train(
        self,
        dataloaders_dict: Dict[str, DataLoader],
        checkpoint_name: str = "fusion_checkpoint.pth.tar",
    ) -> None:
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_model_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_no_improve = 0

        model_save_path = self.model_dir / checkpoint_name

        for epoch in range(self.config.max_epoch):
            for phase in ["train", "validate"]:
                is_train = phase == "train"
                self.model.train() if is_train else self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                loader = dataloaders_dict[phase]
                grad_context = torch.enable_grad if is_train else torch.no_grad
                with grad_context():
                    for batch in tqdm(
                        loader,
                        desc=f"Fusion Epoch {epoch + 1}/{self.config.max_epoch} - {phase}",
                    ):
                        images = batch["image"].to(self.device, non_blocking=True)
                        foreign_features = batch["foreign_features"].to(
                            self.device, non_blocking=True
                        )
                        labels = batch["label"].to(self.device, non_blocking=True)

                        if is_train:
                            optimizer.zero_grad(set_to_none=True)

                        outputs = self.model(images, foreign_features)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if is_train:
                            loss.backward()
                            optimizer.step()

                        running_loss += loss.item() * images.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(loader.dataset)
                epoch_acc = running_corrects.double() / len(loader.dataset)
                logger.info("Fusion %s Loss: %.4f Acc: %.4f", phase, epoch_loss, epoch_acc)

                if not is_train:
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = self.model.state_dict()
                        epochs_no_improve = 0
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": best_model_state,
                                "loss": best_val_loss,
                            },
                            model_save_path,
                        )
                        logger.info("New best fusion model saved to %s", model_save_path)
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= 3:
                logger.info("Fusion early stopping triggered.")
                break


def main_fusion(windows: Optional[List[int]] = None) -> None:
    RUN_MODE = "TEST"
    params = CNNParams()
    default_windows = params.window_sizes
    target_windows = windows if windows is not None else default_windows

    for ws in target_windows:
        if ws not in default_windows:
            logger.warning("Skipping window %d: not defined in config.json", ws)
            continue

        config = params.get_config(RUN_MODE, ws)
        logger.info(
            "\n%s TRAINING FUSION MODEL: %dd%dp %s",
            "=" * 25,
            ws,
            config.pw,
            "=" * 25,
        )
        logger.info("--- Running in %s MODE ---", RUN_MODE)

        dataloaders = get_fusion_dataloaders(
            ws=ws,
            train_years=config.train_years,
            foreign_windows=(5, 20, 60),
            config=config,
        )

        fusion_ds = dataloaders["train"].dataset
        base_ds = fusion_ds.dataset if hasattr(fusion_ds, "dataset") else fusion_ds
        if not isinstance(base_ds, FusionDataset):
            raise TypeError("Expected FusionDataset as underlying dataset for fusion training.")

        feature_dim = base_ds.feature_dim

        paddings = [(int(fs[0] / 2), int(fs[1] / 2)) for fs in config.filter_sizes]

        cnn_encoder = CNNModel(
            layer_number=len(config.conv_channels),
            input_size=(base_ds.image_dataset.image_height, base_ds.image_dataset.image_width),
            inplanes=config.conv_channels[0],
            conv_layer_chanls=config.conv_channels,
            drop_prob=config.drop_prob,
            filter_size_list=config.filter_sizes,
            stride_list=[(1, 1)] * len(config.conv_channels),
            padding_list=paddings,
            dilation_list=[(1, 1)] * len(config.conv_channels),
            max_pooling_list=[(2, 1)] * len(config.conv_channels),
        )

        foreign_encoder = ForeignFeatureMLP(input_dim=feature_dim, embedding_dim=128, hidden_dim=128)

        fusion_model = FusionModel(
            image_encoder=cnn_encoder,
            foreign_encoder=foreign_encoder,
            fusion_hidden_dims=(256, 128),
            dropout=config.drop_prob,
        )

        trainer = FusionTrainer(ws=ws, pw=config.pw, config=config, fusion_model=fusion_model)
        trainer.train(dataloaders)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    WINDOWS_TO_TRAIN = [5, 20, 60]
    main_fusion(windows=WINDOWS_TO_TRAIN)
