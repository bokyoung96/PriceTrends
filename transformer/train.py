import logging
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from core.models.model1 import KoreanEquityDataset
from transformer.model.model1 import Transformer
from transformer.model.model2 import MultiHeadTransformer
from transformer.model.model3 import (MultiModalCrash, get_multimodal_loaders,
                                      load_cnn_cfg, load_transformer_cfg)
from transformer.model.registry import MODEL_REGISTRY
from transformer.params import build_name
from transformer.pipeline import Config, get_loaders
from utils.root import MODELS_ROOT

logger = logging.getLogger(__name__)


def get_tf_config(tcfg) -> Config:
    return Config(
        lookback=tcfg.lookback,
        stride=tcfg.stride,
        horizon=tcfg.horizon,
        features=tuple(tcfg.features),
        min_assets=tcfg.min_assets,
        norm=tcfg.norm,
        label_type=tcfg.label_type,
        threshold=tcfg.threshold,
        train_years=tcfg.train_years,
        test_years=tcfg.test_years,
    )


def get_cnn_loaders(cnn_cfg, batch: int, ratio: float = 0.8, workers: int = 0) -> Dict[str, DataLoader]:
    ds = KoreanEquityDataset(cnn_cfg.window, cnn_cfg.train_years)
    cnn_cfg.image_shape = (ds.image_height, ds.image_width)

    train_idx, val_idx = train_test_split(
        list(range(len(ds))),
        test_size=1.0 - ratio,
        random_state=42,
        stratify=ds.metadata["label"],
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    return {
        "train": DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers),
        "validate": DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers),
    }


class Trainer:
    def __init__(
        self,
        model_type: str = "transformer",
        mode: str = "TEST",
        timeframe: str = "MEDIUM",
        cnn_window: int = 5,
    ):
        self.model_type = model_type.lower()
        self.mode = mode
        self.timeframe = timeframe
        self.cnn_window = cnn_window

        self.tf_cfg = load_transformer_cfg(mode=mode, timeframe=timeframe)
        self.cnn_cfg = load_cnn_cfg(mode=mode, window=cnn_window) if self.model_type in ("cnn", "multimodal_crash") else None

        sel = DeviceSelector()
        self.dev = sel.resolve()
        logger.info(sel.summary("Trainer"))

        name = build_name(self.tf_cfg.mode, self.model_type)
        self.dir = MODELS_ROOT / name
        os.makedirs(self.dir, exist_ok=True)
        self.name = name

    def _build_loaders(
        self,
        batch: int,
        ratio: float,
        workers: int,
        crash_threshold: float,
    ) -> Dict[str, DataLoader]:
        if self.model_type in ("multi", "transformer"):
            cfg = get_tf_config(self.tf_cfg)
            loaders = get_loaders(cfg, batch=batch, ratio=ratio, workers=workers)
            sample = loaders["train"].dataset[0]["input"]
            self.tf_cfg.n_feat = sample.shape[1]
            self.tf_cfg.seq_len = sample.shape[0]
            return loaders

        if self.model_type == "cnn" and self.cnn_cfg is not None:
            return get_cnn_loaders(self.cnn_cfg, batch=batch, ratio=ratio, workers=workers)

        if self.model_type == "multimodal_crash" and self.cnn_cfg is not None:
            loaders, seq_shape, image_shape = get_multimodal_loaders(
                self.tf_cfg,
                self.cnn_cfg,
                batch=batch,
                ratio=ratio,
                workers=workers,
                crash_threshold=crash_threshold,
            )
            self.tf_cfg.seq_len, self.tf_cfg.n_feat = seq_shape
            self.cnn_cfg.image_shape = image_shape
            return loaders

        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _build_model(self) -> nn.Module:
        model_cls = MODEL_REGISTRY.get(self.model_type)
        if model_cls is None:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        if self.model_type == "multi":
            return MultiHeadTransformer(
                n_feat=self.tf_cfg.n_feat,
                d_model=self.tf_cfg.d_model,
                nhead=self.tf_cfg.nhead,
                n_layers=self.tf_cfg.n_layers,
                d_ff=self.tf_cfg.d_ff,
                drop=self.tf_cfg.drop,
                n_class=2,
                max_len=self.tf_cfg.seq_len + 100,
            )
        if self.model_type == "transformer":
            return Transformer(
                n_feat=self.tf_cfg.n_feat,
                d_model=self.tf_cfg.d_model,
                nhead=self.tf_cfg.nhead,
                n_layers=self.tf_cfg.n_layers,
                d_ff=self.tf_cfg.d_ff,
                drop=self.tf_cfg.drop,
                n_class=2,
                max_len=self.tf_cfg.seq_len + 100,
            )
        if self.model_type == "cnn":
            paddings = [(fs[0] // 2, fs[1] // 2) for fs in self.cnn_cfg.filter_sizes]
            return model_cls(
                layer_number=len(self.cnn_cfg.conv_channels),
                input_size=self.cnn_cfg.image_shape,
                inplanes=self.cnn_cfg.conv_channels[0],
                conv_layer_chanls=self.cnn_cfg.conv_channels,
                drop_prob=self.cnn_cfg.drop_prob,
                filter_size_list=self.cnn_cfg.filter_sizes,
                stride_list=[(1, 1)] * len(self.cnn_cfg.conv_channels),
                padding_list=paddings,
                dilation_list=[(1, 1)] * len(self.cnn_cfg.conv_channels),
                max_pooling_list=[(2, 1)] * len(self.cnn_cfg.conv_channels),
            )
        if self.model_type == "multimodal_crash":
            return MultiModalCrash(self.tf_cfg, self.cnn_cfg)

        raise ValueError(f"Unsupported model_type for training: {self.model_type}")

    def train(
        self,
        epochs: int = 20,
        lr: float = 1e-4,
        batch: int = 64,
        ratio: float = 0.8,
        workers: int = 0,
        lambda_crash: float = 1.0,
        crash_threshold: float = 0.1,
    ):
        loaders = self._build_loaders(batch=batch, ratio=ratio, workers=workers, crash_threshold=crash_threshold)
        model = self._build_model().to(self.dev)

        opt = optim.Adam(model.parameters(), lr=lr)
        crit_cls = nn.CrossEntropyLoss()
        crit_crash = nn.CrossEntropyLoss()
        crit_dd = nn.MSELoss()

        best_loss = float("inf")

        for ep in range(epochs):
            for phase in ["train", "validate"]:
                model.train() if phase == "train" else model.eval()

                run_loss = 0.0
                run_acc = 0
                total = 0

                loader = loaders[phase]

                with torch.set_grad_enabled(phase == "train"):
                    pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs} - {phase}")
                    for batch_data in pbar:
                        loss = 0.0
                        main_logits = None

                        opt.zero_grad()

                        if self.model_type == "multimodal_crash":
                            seq = batch_data["input"].to(self.dev)
                            img = batch_data["image"].to(self.dev)
                            out = model({"input": seq, "image": img})
                            main_logits = out.get("return")
                            crash_logits = out.get("crash")

                            if main_logits is not None:
                                loss += crit_cls(main_logits, batch_data["label"].to(self.dev))
                            if crash_logits is not None:
                                loss += lambda_crash * crit_crash(
                                    crash_logits, batch_data["crash_label"].to(self.dev)
                                )
                        elif self.model_type == "cnn":
                            logits = model(batch_data["image"].to(self.dev))
                            main_logits = logits
                            loss = crit_cls(logits, batch_data["label"].to(self.dev))
                        elif self.model_type == "multi":
                            x = batch_data["input"].to(self.dev)
                            y = batch_data["label"].to(self.dev)
                            y_dd = batch_data["dd"].to(self.dev)
                            logits, dd_pred = model(x)
                            main_logits = logits
                            loss = crit_cls(logits, y) + lambda_crash * crit_dd(dd_pred, y_dd)
                        else:
                            x = batch_data["input"].to(self.dev)
                            y = batch_data["label"].to(self.dev)
                            logits = model(x)
                            main_logits = logits
                            loss = crit_cls(logits, y)

                        preds = None
                        if main_logits is not None:
                            _, preds = torch.max(main_logits, 1)

                        if phase == "train":
                            loss.backward()
                            opt.step()

                        batch_size = (
                            batch_data.get("input", batch_data.get("image")).size(0)
                        )
                        run_loss += loss.item() * batch_size
                        if preds is not None and "label" in batch_data:
                            labels = batch_data["label"].to(self.dev)
                            run_acc += torch.sum(preds == labels)
                        total += batch_size

                        pbar.set_postfix({"loss": loss.item()})

                ep_loss = run_loss / max(total, 1)
                ep_acc = run_acc.float() / max(total, 1)

                logger.info(f"{phase} Loss: {ep_loss:.4f} Acc: {ep_acc:.4f}")

                if phase == "validate":
                    if ep_loss < best_loss:
                        best_loss = ep_loss
                        torch.save(model.state_dict(), self.dir / "checkpoint0.pth")
                        logger.info(f"Saved best: {best_loss:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    model_type = "multimodal_crash"
    mode = "TEST"
    timeframe = "MEDIUM"
    cnn_window = 5

    trainer = Trainer(model_type=model_type, mode=mode, timeframe=timeframe, cnn_window=cnn_window)
    trainer.train(
        epochs=trainer.tf_cfg.max_epoch,
        batch=trainer.tf_cfg.batch_size,
        lr=trainer.tf_cfg.lr,
        workers=0,
        lambda_crash=1.0,
        crash_threshold=0.05,
    )
