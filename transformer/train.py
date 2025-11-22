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
from utils.root import MODELS_ROOT
from transformer.model import Transformer
from transformer.pipeline import Config, get_loaders
from transformer.params import TransformerParams

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: Config, name: str = "transformer"):
        self.cfg = cfg
        self.name = name
        
        sel = DeviceSelector()
        self.dev = sel.resolve()
        logger.info(sel.summary("Trainer"))
        
        self.dir = MODELS_ROOT / self.name
        os.makedirs(self.dir, exist_ok=True)
        
    def train(self, epochs: int = 20, lr: float = 1e-4, batch: int = 64, 
              d_model: int = 64, nhead: int = 4, n_layers: int = 2, d_ff: int = 128, drop: float = 0.1):
        loaders = get_loaders(self.cfg, batch=batch, ratio=0.8, workers=0)
        
        sample = loaders['train'].dataset[0]['input']
        n_feat = sample.shape[1]
        seq_len = sample.shape[0]
        
        logger.info(f"Feats: {n_feat}, Len: {seq_len}")
        
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
        crit = nn.CrossEntropyLoss()
        
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
                        
                        opt.zero_grad()
                        
                        out = model(x)
                        loss = crit(out, y)
                        
                        _, preds = torch.max(out, 1)
                        
                        if phase == 'train':
                            loss.backward()
                            opt.step()
                            
                        run_loss += loss.item() * x.size(0)
                        run_acc += torch.sum(preds == y.data)
                        total += x.size(0)
                        
                        pbar.set_postfix({'loss': loss.item()})
                
                ep_loss = run_loss / total
                ep_acc = run_acc.double() / total
                
                logger.info(f"{phase} Loss: {ep_loss:.4f} Acc: {ep_acc:.4f}")
                
                if phase == 'validate':
                    if ep_loss < best_loss:
                        best_loss = ep_loss
                        torch.save(model.state_dict(), self.dir / "best.pth")
                        logger.info(f"Saved best: {best_loss:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
        
    params = TransformerParams()
    tcfg = params.get_config(mode="TEST", timeframe="MEDIUM")
    
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
    
    trainer = Trainer(cfg, name=f"transformer_{tcfg.mode}")
    trainer.train(
        epochs=tcfg.max_epoch,
        batch=tcfg.batch_size,
        lr=tcfg.lr,
        d_model=tcfg.d_model,
        nhead=tcfg.nhead,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        drop=tcfg.drop
    )
