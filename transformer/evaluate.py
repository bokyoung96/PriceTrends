import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from transformer.model.model1 import Transformer
from transformer.model.model2 import MultiHeadTransformer
from transformer.model.model3 import (CrashTransformer, MultiModalCrash,
                                      get_multimodal_loaders, load_cnn_cfg,
                                      load_transformer_cfg)
from transformer.params import TransformerParams, build_name
from transformer.pipeline import Config, get_loaders
from utils.root import MODELS_ROOT, RESULTS_ROOT

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        cfg: Config,
        name: str = "transformer",
        model_type: str = "transformer",
        timeframe: str = "MEDIUM",
        cnn_window: int = 5,
        crash_threshold: float = 0.1,
    ):
        self.cfg = cfg
        self.name = name
        self.model_type = model_type.lower()
        self.timeframe = timeframe
        self.cnn_window = cnn_window
        self.crash_threshold = crash_threshold

        sel = DeviceSelector()
        self.dev = sel.resolve()

        self.dir = MODELS_ROOT / self.name
        self.path = self.dir / "checkpoint0.pth"

    def _build_loader_and_model(
        self,
        batch: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        d_ff: int,
        drop: float,
        crash_threshold: float | None = None,
    ):
        if self.model_type == "multimodal_crash":
            tf_cfg = load_transformer_cfg(mode="TEST", timeframe=self.timeframe)
            cnn_cfg = load_cnn_cfg(mode="TEST", window=self.cnn_window)
            ct = self.crash_threshold if crash_threshold is None else crash_threshold
            loaders, seq_shape, image_shape = get_multimodal_loaders(
                tf_cfg,
                cnn_cfg,
                batch=batch,
                workers=0,
                crash_threshold=ct,
            )
            tf_cfg.seq_len, tf_cfg.n_feat = seq_shape
            cnn_cfg.image_shape = image_shape
            model = MultiModalCrash(tf_cfg, cnn_cfg).to(self.dev)
            loader = loaders["validate"]
            return loader, model

        loaders = get_loaders(self.cfg, batch=batch, workers=0)
        loader = loaders["validate"]

        sample = loaders["train"].dataset[0]["input"]
        n_feat = sample.shape[1]
        seq_len = sample.shape[0]

        if self.model_type == "crash":
            model = CrashTransformer(
                n_feat=n_feat,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                d_ff=d_ff,
                drop=drop,
                n_class=2,
                max_len=seq_len + 100,
            ).to(self.dev)
        elif self.model_type == "multi":
            model = MultiHeadTransformer(
                n_feat=n_feat,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                d_ff=d_ff,
                drop=drop,
                n_class=2,
                max_len=seq_len + 100,
            ).to(self.dev)
        else:
            model = Transformer(
                n_feat=n_feat,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                d_ff=d_ff,
                drop=drop,
                n_class=2,
                max_len=seq_len + 100,
            ).to(self.dev)

        return loader, model

    def run(
        self,
        batch: int = 256,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        drop: float = 0.1,
        k: float = 0.5,
        crash_threshold: float | None = None,
    ) -> pd.DataFrame:

        loader, model = self._build_loader_and_model(
            batch, d_model, nhead, n_layers, d_ff, drop, crash_threshold=crash_threshold
        )

        if not self.path.exists():
            raise FileNotFoundError(f"No checkpoint: {self.path}")

        logger.info(f"Loading model from {self.path}")
        model.load_state_dict(torch.load(self.path, map_location=self.dev))
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        all_crash_probs = []
        all_dd_pred = []
        all_dates = []
        all_assets = []

        with torch.no_grad():
            for b in tqdm(loader, desc="Evaluating"):
                if self.model_type == "multimodal_crash":
                    seq = b["input"].to(self.dev)
                    img = b["image"].to(self.dev)
                    out = model({"input": seq, "image": img})
                    logits = out["return"]
                    crash_logits = out["crash"]
                    y = b["label"].to(self.dev)
                    d = b["date"]
                    a = b["asset"]
                    crash_prob = F.softmax(crash_logits, dim=1)[:, 1]
                    all_crash_probs.extend(crash_prob.cpu().numpy())
                else:
                    x = b["input"].to(self.dev)
                    y = b["label"].to(self.dev)
                    d = b["date"]
                    a = b["asset"]

                    if self.model_type == "crash":
                        logits, crash_logits = model(x)
                        crash_prob = F.softmax(crash_logits, dim=1)[:, 1]
                        all_crash_probs.extend(crash_prob.cpu().numpy())
                    elif self.model_type == "multi":
                        logits, dd_pred = model(x)
                        all_dd_pred.extend(dd_pred.cpu().numpy())
                    else:
                        logits = model(x)

                prob = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
                all_dates.extend(d)
                all_assets.extend(a)

        acc = (pd.Series(all_preds) == pd.Series(all_labels)).mean()
        logger.info(f"Accuracy: {acc:.4f}")

        logger.info("\n" + classification_report(all_labels, all_preds, target_names=["Down", "Up"]))
        logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(all_labels, all_preds)))

        probs = pd.DataFrame(all_probs, columns=["prob_down", "prob_up"])
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(all_dates),
                "asset": all_assets,
                "label": all_labels,
                "prediction": all_preds,
                "prob_down": probs["prob_down"],
                "prob_up": probs["prob_up"],
            }
        )

        if self.model_type == "crash" or self.model_type == "multimodal_crash":
            crash_prob = pd.Series(all_crash_probs, name="crash_prob")
            df["crash_prob"] = crash_prob.values
            df["score"] = df["prob_up"] - k * df["crash_prob"]
        elif self.model_type == "multi":
            dd_pred = pd.Series(all_dd_pred, name="dd_pred")
            df["dd_pred"] = dd_pred.values
            df["score"] = df["prob_up"] - k * df["dd_pred"]

        return df
    
    def save(self, df: pd.DataFrame, mode: str):
        res_dir = RESULTS_ROOT / self.name
        res_dir.mkdir(parents=True, exist_ok=True)
        
        path = res_dir / f"price_trends_results_{self.name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved results to {path}")
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    params = TransformerParams()
    tcfg = params.get_config(mode="TEST", timeframe="MEDIUM")

    model_type = "multimodal_crash"
    crash_threshold = 0.05
    
    cfg = Config(
        lookback=tcfg.lookback,
        stride=tcfg.stride,
        horizon=tcfg.horizon,
        features=tuple(tcfg.features),
        min_assets=tcfg.min_assets,
        norm=tcfg.norm,
        label_type=tcfg.label_type,
        threshold=tcfg.threshold,
        train_years=tcfg.train_years,
        test_years=tcfg.test_years
    )
    
    name = build_name(tcfg.mode, model_type)
    ev = Evaluator(
        cfg,
        name=name,
        model_type=model_type,
        timeframe="MEDIUM",
        cnn_window=5,
        crash_threshold=crash_threshold,
    )
    
    df = ev.run(
        batch=1024,
        d_model=tcfg.d_model,
        nhead=tcfg.nhead,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        drop=tcfg.drop,
        k=1,
        crash_threshold=crash_threshold,
    )
    
    ev.save(df, mode="TEST")
