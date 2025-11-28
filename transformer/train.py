import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from transformer.model.model1 import Transformer
from transformer.model.model2 import MultiHeadTransformer
from transformer.params import TransformerParams, build_name
from transformer.pipeline import Config, get_loaders
from utils.root import MODELS_ROOT

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: Config, name: str = "transformer", model_type: str = "transformer"):
        self.cfg = cfg
        self.name = name
        self.model_type = model_type
        
        sel = DeviceSelector()
        self.dev = sel.resolve()
        logger.info(sel.summary("Trainer"))
        
        self.dir = MODELS_ROOT / self.name
        os.makedirs(self.dir, exist_ok=True)
        
    def train(self, epochs: int = 20, lr: float = 1e-4, batch: int = 64, 
              d_model: int = 64, nhead: int = 4, n_layers: int = 2, d_ff: int = 128, drop: float = 0.1,
              lambda_dd: float = 1.0):
        loaders = get_loaders(self.cfg, batch=batch, ratio=0.8, workers=0)
        
        sample = loaders['train'].dataset[0]['input']
        n_feat = sample.shape[1]
        seq_len = sample.shape[0]
        
        logger.info(f"Feats: {n_feat}, Len: {seq_len}")
        
        if self.model_type == "multi":
            model = MultiHeadTransformer(
                n_feat=n_feat,
                d_model=d_model,
                nhead=nhead,
                n_layers=n_layers,
                d_ff=d_ff,
                drop=drop,
                n_class=2,
                max_len=seq_len + 100
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
                max_len=seq_len + 100
            ).to(self.dev)
        
        opt = optim.Adam(model.parameters(), lr=lr)
        crit_cls = nn.CrossEntropyLoss()
        crit_dd = nn.MSELoss()
        
        best_loss = float('inf')
        
        for ep in range(epochs):
            for phase in ['train', 'validate']:
                if phase == 'train': model.train()
                else: model.eval()
                    
                run_loss = 0.0
                run_acc = 0
                total = 0
                
                loader = loaders[phase]
                
                with torch.set_grad_enabled(phase == 'train'):
                    pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs} - {phase}")
                    for batch_data in pbar:
                        x = batch_data['input'].to(self.dev)
                        y = batch_data['label'].to(self.dev)
                        y_dd = batch_data['dd'].to(self.dev)
                        
                        opt.zero_grad()
                        
                        if self.model_type == "multi":
                            logits, dd_pred = model(x)
                            loss_cls = crit_cls(logits, y)
                            loss_dd = crit_dd(dd_pred, y_dd)
                            loss = loss_cls + lambda_dd * loss_dd
                        else:
                            logits = model(x)
                            loss = crit_cls(logits, y)
                        
                        _, preds = torch.max(logits, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            opt.step()
                            
                        run_loss += loss.item() * x.size(0)
                        run_acc += torch.sum(preds == y.data)
                        total += x.size(0)
                        
                        pbar.set_postfix({'loss': loss.item()})
                
                ep_loss = run_loss / total
                ep_acc = run_acc.float() / total
                
                logger.info(f"{phase} Loss: {ep_loss:.4f} Acc: {ep_acc:.4f}")
                
                if phase == 'validate':
                    if ep_loss < best_loss:
                        best_loss = ep_loss
                        torch.save(model.state_dict(), self.dir / "checkpoint0.pth")
                        logger.info(f"Saved best: {best_loss:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
        
    params = TransformerParams()
    tcfg = params.get_config(mode="TEST", timeframe="LONG")

    model_type = "multi"
    
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
    trainer = Trainer(cfg, name=name, model_type=model_type)
    trainer.train(
        epochs=tcfg.max_epoch,
        batch=tcfg.batch_size,
        lr=tcfg.lr,
        d_model=tcfg.d_model,
        nhead=tcfg.nhead,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        drop=tcfg.drop,
        lambda_dd=1.0
    )
