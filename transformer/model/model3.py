import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.model1 import CNNModel, KoreanEquityDataset
from core.params import CNNConfig, CNNParams
from transformer.params import TransformerConfig, TransformerParams
from transformer.pipeline import Config as TfConfig
from transformer.pipeline import Pipeline as TfPipeline
from transformer.pipeline import Windows

from .model1 import GRN, VSN, PosEncoding

logger = logging.getLogger(__name__)


class CrashTransformer(nn.Module):
    def __init__(
        self,
        n_feat: int,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        drop: float = 0.1,
        n_class: int = 2,
        max_len: int = 5000,
    ):
        super().__init__()
        self.vsn = VSN(n_feat, 1, d_model, drop)
        self.pe = PosEncoding(d_model, max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=drop,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.shared_grn = GRN(d_model, d_ff, d_model, drop)
        self.norm = nn.LayerNorm(d_model)

        self.head_cls = nn.Linear(d_model, n_class)
        self.head_crash = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, 2),
        )

        self._init_weights()

    def _forward_backbone(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # NOTE: Backbone: VSN -> PE -> Encoder -> MeanPool -> GRN -> Norm
        x = self.vsn(x)
        x = self.pe(x)
        x = self.enc(x, src_key_padding_mask=mask)
        x = x.mean(dim=1)
        x = self.shared_grn(x)
        x = self.norm(x)
        return x

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NOTE: x: (batch, seq, n_feat)
        x = self._forward_backbone(x, mask=mask)

        logits = self.head_cls(x)
        crash_logits = self.head_crash(x)
        return logits, crash_logits


def load_transformer_cfg(mode: str = "TEST", timeframe: str = "MEDIUM") -> TransformerConfig:
    params = TransformerParams()
    return params.get_config(mode=mode, timeframe=timeframe)


def load_cnn_cfg(mode: str = "TEST", window: int = 5) -> CNNConfig:
    params = CNNParams()
    cfg = params.get_config(mode=mode, window_size=window)
    cfg.window = window
    return cfg


class MultiModalFusionDataset(Dataset):
    def __init__(
        self,
        windows: Windows,
        image_dataset: KoreanEquityDataset,
        crash_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        meta = image_dataset.metadata.copy()
        meta["key"] = list(
            zip(
                pd.to_datetime(meta["end_date"]).dt.normalize(),
                meta["ticker"],
            )
        )

        dates = pd.to_datetime(windows.dates).normalize()
        assets = pd.Series(windows.assets).astype(str)
        keys = list(zip(dates, assets))

        win_df = pd.DataFrame(
            {
                "key": keys,
                "seq_idx": np.arange(len(windows.data)),
                "ret_label": windows.targets_cls.astype(np.int64),
                "dd": windows.targets_dd.astype(np.float32),
            }
        )

        merged = meta.merge(win_df, on="key", how="inner")
        if merged.empty:
            raise ValueError("MultiModalFusionDataset has no overlapping samples. Check end_date alignment.")

        self.seq_data = windows.data.astype(np.float32)
        self.seq_len = self.seq_data.shape[1]
        self.seq_dim = self.seq_data.shape[2]

        self.image_data = image_dataset.images
        self.image_shape = (image_dataset.image_height, image_dataset.image_width)

        self.seq_indices = merged["seq_idx"].to_numpy(dtype=np.int64)
        self.image_indices = merged["image_idx"].to_numpy(dtype=np.int64)
        self.labels = merged["ret_label"].to_numpy(dtype=np.int64)
        dd = merged["dd"].to_numpy(dtype=np.float32)
        self.crash_labels = (dd >= crash_threshold).astype(np.int64)
        logger.info("Crash label 1 ratio: %.4f", float(self.crash_labels.mean()))
        logger.info("Crash label 0 ratio: %.4f", float((1 - self.crash_labels).mean()))
        self.dates = pd.to_datetime(merged["end_date"]).to_numpy()
        self.assets = merged["ticker"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq_idx = self.seq_indices[idx]
        img_idx = self.image_indices[idx]

        seq = torch.from_numpy(self.seq_data[seq_idx])
        img = self.image_data[img_idx]
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        assert img_tensor.dim() == 3 and img_tensor.size(0) == 1, "Expected grayscale image with shape (1,H,W)"

        return {
            "input": seq,
            "image": img_tensor,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "crash_label": torch.tensor(self.crash_labels[idx], dtype=torch.long),
            "date": str(self.dates[idx]),
            "asset": self.assets[idx],
        }


def get_multimodal_loaders(
    tf_cfg: TransformerConfig,
    cnn_cfg: CNNConfig,
    batch: int = 32,
    ratio: float = 0.8,
    workers: int = 0,
    crash_threshold: float = 0.1,
) -> Tuple[Dict[str, DataLoader], Tuple[int, int], Tuple[int, int]]:
    tf_config = TfConfig(
        lookback=tf_cfg.lookback,
        stride=tf_cfg.stride,
        horizon=tf_cfg.horizon,
        features=tuple(tf_cfg.features),
        min_assets=tf_cfg.min_assets,
        norm=tf_cfg.norm,
        label_type=tf_cfg.label_type,
        threshold=tf_cfg.threshold,
        train_years=tf_cfg.train_years,
        test_years=tf_cfg.test_years,
    )
    pipe = TfPipeline(tf_config)
    path = pipe.get_path()
    wins = Windows.load(path) if path.exists() else pipe.run()

    years_for_images: List[int] = list(tf_cfg.train_years or [])
    if tf_cfg.test_years:
        years_for_images.extend(tf_cfg.test_years)
    years_for_images = sorted(list(dict.fromkeys(years_for_images)))
    image_ds = KoreanEquityDataset(intervals=cnn_cfg.window, years=years_for_images)

    ds = MultiModalFusionDataset(wins, image_ds, crash_threshold=crash_threshold)

    idx = np.arange(len(ds))
    if tf_cfg.train_years is not None:
        year_arr = pd.to_datetime(ds.dates).year.astype(int)
        mask = np.isin(year_arr, list(tf_cfg.train_years))
        train_ds = Subset(ds, idx[mask])
        val_ds = Subset(ds, idx[~mask])
    else:
        split = int(len(ds) * ratio)
        train_ds = Subset(ds, idx[:split])
        val_ds = Subset(ds, idx[split:])

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        "validate": DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers),
    }

    seq_shape = (ds.seq_len, ds.seq_dim)
    image_shape = ds.image_shape
    return loaders, seq_shape, image_shape


class MultiModalCrash(nn.Module):
    def __init__(self, transformer_cfg: TransformerConfig, cnn_cfg: CNNConfig) -> None:
        super().__init__()
        if not hasattr(transformer_cfg, "n_feat") or not hasattr(transformer_cfg, "seq_len"):
            raise ValueError("Transformer config must include n_feat and seq_len for multimodal model.")
        if not hasattr(cnn_cfg, "image_shape") or not hasattr(cnn_cfg, "window"):
            raise ValueError("CNN config must include image_shape and window for multimodal model.")

        self.d_model = transformer_cfg.d_model
        self.d_ff = transformer_cfg.d_ff
        self.drop = transformer_cfg.drop

        max_len = transformer_cfg.seq_len + 100
        self.seq_encoder = CrashTransformer(
            n_feat=transformer_cfg.n_feat,
            d_model=transformer_cfg.d_model,
            nhead=transformer_cfg.nhead,
            n_layers=transformer_cfg.n_layers,
            d_ff=transformer_cfg.d_ff,
            drop=transformer_cfg.drop,
            n_class=2,
            max_len=max_len,
        )

        paddings = [(fs[0] // 2, fs[1] // 2) for fs in cnn_cfg.filter_sizes]
        self.cnn_encoder = CNNModel(
            layer_number=len(cnn_cfg.conv_channels),
            input_size=cnn_cfg.image_shape,
            inplanes=cnn_cfg.conv_channels[0],
            conv_layer_chanls=cnn_cfg.conv_channels,
            drop_prob=cnn_cfg.drop_prob,
            filter_size_list=cnn_cfg.filter_sizes,
            stride_list=[(1, 1)] * len(cnn_cfg.conv_channels),
            padding_list=paddings,
            dilation_list=[(1, 1)] * len(cnn_cfg.conv_channels),
            max_pooling_list=[(2, 1)] * len(cnn_cfg.conv_channels),
        )
        self.cnn_encoder.fc = nn.Identity()

        cnn_dim = self._get_cnn_dim()

        self.cnn_norm = nn.LayerNorm(cnn_dim)
        self.cnn_proj = nn.Linear(cnn_dim, self.d_model)

        self.seq_proj = nn.Linear(self.d_model, self.d_model)
        self.proj_dropout = nn.Dropout(self.drop)

        self.fusion_norm = nn.LayerNorm(self.d_model * 2)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_ff),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(self.d_ff, self.d_model),
        )

        self.head_norm = nn.LayerNorm(self.d_model)
        self.return_head = nn.Linear(self.d_model, 2)
        self.crash_head = nn.Linear(self.d_model, 2)

        self._init_weights()

    def _get_cnn_dim(self) -> int:
        if not hasattr(self.cnn_encoder, "conv_layers"):
            raise ValueError("CNN encoder must expose conv_layers.")
        height, width = self.cnn_encoder.input_size
        dummy = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            x = self.cnn_encoder.conv_layers(dummy)
            x = x.view(x.size(0), -1)
        return x.shape[1]

    def _forward_cnn(self, image: torch.Tensor) -> torch.Tensor:
        x = self.cnn_encoder.conv_layers(image)
        x = x.view(x.size(0), -1)
        x = self.cnn_norm(x)
        x = self.cnn_proj(x)
        x = self.proj_dropout(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module is self.crash_head:
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq = batch["input"]
        image = batch["image"]
        mask = batch.get("mask")

        seq_feat = self.seq_encoder._forward_backbone(seq, mask=mask)
        seq_feat = self.seq_proj(seq_feat)
        seq_feat = self.proj_dropout(seq_feat)

        cnn_feat = self._forward_cnn(image)

        fused = torch.cat([seq_feat, cnn_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_mlp(fused)

        head_input = self.head_norm(fused)
        return_logits = self.return_head(head_input)
        crash_logits = self.crash_head(head_input)

        crash_probs = torch.softmax(crash_logits, dim=-1)
        return_probs = torch.softmax(return_logits, dim=-1)

        return {
            "return": return_logits,
            "crash": crash_logits,
            "return_logits": return_logits,
            "crash_logits": crash_logits,
            "return_probs": return_probs,
            "crash_probs": crash_probs,
            "seq_feat": seq_feat,
            "cnn_feat": cnn_feat,
        }
