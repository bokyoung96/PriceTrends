import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from transformer.params import TransformerConfig
from transformer.pipeline import Config as TfConfig
from transformer.pipeline import Pipeline as TfPipeline
from transformer.pipeline import Windows

from .model1 import GRN, VSN, PosEncoding

logger = logging.getLogger(__name__)


def load_cnn_cfgs(mode: str = "TEST", windows: Sequence[int] | None = None) -> List[CNNConfig]:
    params = CNNParams()
    target_windows = params.window_sizes if windows is None else list(dict.fromkeys(windows))
    target_windows = sorted(target_windows)
    cfgs: List[CNNConfig] = []
    for ws in target_windows:
        cfg = params.get_config(mode=mode, window_size=ws)
        cfg.window = ws
        cfgs.append(cfg)
    return cfgs


class MultiWindowFusionDataset(Dataset):
    """
    Aligns transformer windows with one or more CNN chart datasets.
    This dataset is dedicated to return classification only (no MDD/crash labels).
    """

    def __init__(
        self,
        windows: Windows,
        image_datasets: Dict[int, KoreanEquityDataset],
    ) -> None:
        super().__init__()
        if not image_datasets:
            raise ValueError("image_datasets must not be empty.")

        meta_frames: List[pd.DataFrame] = []
        intervals = sorted(image_datasets.keys())

        for interval in intervals:
            ds = image_datasets[interval]
            meta = ds.metadata.copy()
            meta["key"] = list(
                zip(pd.to_datetime(meta["end_date"]).dt.normalize(), meta["ticker"].astype(str))
            )
            meta_frames.append(
                meta[["key", "image_idx"]].rename(columns={"image_idx": f"image_idx_{interval}"})
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

        merged = win_df
        for meta in meta_frames:
            merged = merged.merge(meta, on="key", how="inner")

        if merged.empty:
            raise ValueError("MultiWindowFusionDataset has no overlapping samples across intervals.")

        self.seq_data = windows.data.astype(np.float32)
        self.seq_len = self.seq_data.shape[1]
        self.seq_dim = self.seq_data.shape[2]

        self.image_datasets = image_datasets
        self.intervals = intervals
        self.image_shapes = {
            interval: (ds.image_height, ds.image_width) for interval, ds in image_datasets.items()
        }

        self.seq_indices = merged["seq_idx"].to_numpy(dtype=np.int64)
        self.label_arr = merged["ret_label"].to_numpy(dtype=np.int64)
        self.image_indices: List[np.ndarray] = []
        for interval in intervals:
            col = f"image_idx_{interval}"
            self.image_indices.append(merged[col].to_numpy(dtype=np.int64))

        self.dates = pd.to_datetime([k[0] for k in merged["key"]]).to_numpy()
        self.assets = pd.Series([k[1] for k in merged["key"]], dtype=str).to_numpy()

    def __len__(self) -> int:
        return len(self.label_arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq_idx = self.seq_indices[idx]
        seq = torch.from_numpy(self.seq_data[seq_idx])

        images: List[torch.Tensor] = []
        for interval, image_idx in zip(self.intervals, self.image_indices):
            ds = self.image_datasets[interval]
            img = ds.images[image_idx[idx]]
            if img.dtype != np.float32:
                img = img.astype(np.float32, copy=False)
            img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
            images.append(img_tensor)

        batch: Dict[str, Any] = {
            "input": seq,
            "images": images,
            "label": torch.tensor(self.label_arr[idx], dtype=torch.long),
        }
        batch["date"] = str(self.dates[idx])
        batch["asset"] = self.assets[idx]
        return batch


def get_multimodal_v2_loaders(
    tf_cfg: TransformerConfig,
    cnn_cfgs: Sequence[CNNConfig],
    batch: int = 32,
    ratio: float = 0.8,
    workers: int = 0,
) -> Tuple[Dict[str, DataLoader], Tuple[int, int], Dict[int, Tuple[int, int]]]:
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

    image_datasets: Dict[int, KoreanEquityDataset] = {}
    for cfg in cnn_cfgs:
        ds = KoreanEquityDataset(intervals=cfg.window, years=years_for_images)
        image_datasets[int(cfg.window)] = ds
        cfg.image_shape = (ds.image_height, ds.image_width)  # type: ignore[attr-defined]

    ds = MultiWindowFusionDataset(wins, image_datasets)

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
    image_shapes = ds.image_shapes
    return loaders, seq_shape, image_shapes


class MultiWindowCNNEncoder(nn.Module):
    def __init__(self, cnn_cfgs: Sequence[CNNConfig]) -> None:
        super().__init__()
        if not cnn_cfgs:
            raise ValueError("cnn_cfgs must not be empty.")

        self.encoders = nn.ModuleList()
        self.dims: List[int] = []
        for cfg in cnn_cfgs:
            paddings = [(fs[0] // 2, fs[1] // 2) for fs in cfg.filter_sizes]
            encoder = CNNModel(
                layer_number=len(cfg.conv_channels),
                input_size=cfg.image_shape,  # type: ignore[arg-type]
                inplanes=cfg.conv_channels[0],
                conv_layer_chanls=cfg.conv_channels,
                drop_prob=cfg.drop_prob,
                filter_size_list=cfg.filter_sizes,
                stride_list=[(1, 1)] * len(cfg.conv_channels),
                padding_list=paddings,
                dilation_list=[(1, 1)] * len(cfg.conv_channels),
                max_pooling_list=[(2, 1)] * len(cfg.conv_channels),
            )
            encoder.fc = nn.Identity()
            self.encoders.append(encoder)
            self.dims.append(self._get_cnn_dim(encoder, cfg))

    @staticmethod
    def _get_cnn_dim(encoder: CNNModel, cfg: CNNConfig) -> int:
        if not hasattr(encoder, "conv_layers"):
            raise ValueError("CNN encoder must expose conv_layers.")
        height, width = cfg.image_shape  # type: ignore[misc]
        dummy = torch.zeros(1, 1, height, width)
        with torch.no_grad():
            x = encoder.conv_layers(dummy)
            x = x.view(x.size(0), -1)
        return x.shape[1]

    def forward(self, images: Sequence[torch.Tensor]) -> torch.Tensor:
        feats = []
        for img, enc in zip(images, self.encoders):
            x = enc.conv_layers(img)
            x = x.view(x.size(0), -1)
            feats.append(x)
        return torch.cat(feats, dim=-1)


class MultiModalFusion(nn.Module):
    """
    Transformer + multi-window CNN fusion model without MDD/crash heads.
    """

    def __init__(self, transformer_cfg: TransformerConfig, cnn_cfgs: Sequence[CNNConfig]) -> None:
        super().__init__()
        if not hasattr(transformer_cfg, "n_feat") or not hasattr(transformer_cfg, "seq_len"):
            raise ValueError("Transformer config must include n_feat and seq_len for multimodal model.")
        for cfg in cnn_cfgs:
            if not hasattr(cfg, "image_shape") or not hasattr(cfg, "window"):
                raise ValueError("CNN configs must include image_shape and window for multimodal model.")

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

        self.cnn_encoder = MultiWindowCNNEncoder(cnn_cfgs)
        total_cnn_dim = sum(self.cnn_encoder.dims)

        self.cnn_norm = nn.LayerNorm(total_cnn_dim)
        self.cnn_proj = nn.Linear(total_cnn_dim, self.d_model)

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

        self._init_weights()

    def _forward_cnn(self, images: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(images) != len(self.cnn_encoder.encoders):
            raise ValueError(
                f"Expected {len(self.cnn_encoder.encoders)} image tensors, got {len(images)}."
            )
        x = self.cnn_encoder(images)
        x = self.cnn_norm(x)
        x = self.cnn_proj(x)
        x = self.proj_dropout(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq = batch["input"]
        images = batch["images"]
        mask = batch.get("mask")

        seq_feat = self.seq_encoder._forward_backbone(seq, mask=mask)
        seq_feat = self.seq_proj(seq_feat)
        seq_feat = self.proj_dropout(seq_feat)

        cnn_feat = self._forward_cnn(images)

        fused = torch.cat([seq_feat, cnn_feat], dim=-1)
        fused = self.fusion_norm(fused)
        fused = self.fusion_mlp(fused)

        head_input = self.head_norm(fused)
        return_logits = self.return_head(head_input)
        return_probs = torch.softmax(return_logits, dim=-1)

        return {
            "return": return_logits,
            "return_logits": return_logits,
            "return_probs": return_probs,
            "seq_feat": seq_feat,
            "cnn_feat": cnn_feat,
        }


class CrashTransformer(nn.Module):
    """
    Lightweight re-export of model3 CrashTransformer for reuse without importing the entire module.
    """

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

        self._init_weights()

    def _forward_backbone(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        x = self._forward_backbone(x, mask=mask)
        logits = self.head_cls(x)
        return logits, logits
