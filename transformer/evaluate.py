import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from utils.root import MODELS_ROOT, RESULTS_ROOT
from transformer.model import Transformer
from transformer.pipeline import Config, get_loaders
from transformer.params import TransformerParams

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config, name: str = "transformer"):
        self.cfg = cfg
        self.name = name
        
        sel = DeviceSelector()
        self.dev = sel.resolve()
        
        self.dir = MODELS_ROOT / self.name
        self.path = self.dir / "checkpoint0.pth"
        
    def run(self, batch: int = 256, d_model: int = 64, nhead: int = 4, 
            n_layers: int = 2, d_ff: int = 128, drop: float = 0.1) -> pd.DataFrame:
        
        loaders = get_loaders(self.cfg, batch=batch, workers=0)
        loader = loaders['validate']
        
        sample = loaders['train'].dataset[0]['input']
        n_feat = sample.shape[1]
        seq_len = sample.shape[0]
        
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
        
        if not self.path.exists():
            raise FileNotFoundError(f"No checkpoint: {self.path}")
            
        logger.info(f"Loading model from {self.path}")
        model.load_state_dict(torch.load(self.path, map_location=self.dev))
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_dates = []
        all_assets = []
        
        with torch.no_grad():
            for b in tqdm(loader, desc="Evaluating"):
                x = b['input'].to(self.dev)
                y = b['label'].to(self.dev)
                d = b['date']
                a = b['asset']
                
                out = model(x)
                prob = F.softmax(out, dim=1)
                _, preds = torch.max(out, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())
                all_dates.extend(d)
                all_assets.extend(a)
                
        acc = (pd.Series(all_preds) == pd.Series(all_labels)).mean()
        logger.info(f"Accuracy: {acc:.4f}")
        
        logger.info("\n" + classification_report(all_labels, all_preds, target_names=['Down', 'Up']))
        logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(all_labels, all_preds)))
        
        probs = pd.DataFrame(all_probs, columns=['prob_down', 'prob_up'])
        
        df = pd.DataFrame({
            "date": pd.to_datetime(all_dates),
            "asset": all_assets,
            "label": all_labels,
            "prediction": all_preds,
            "prob_down": probs['prob_down'],
            "prob_up": probs['prob_up']
        })
        
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
    
    name = f"transformer_{tcfg.mode}"
    ev = Evaluator(cfg, name=name)
    
    df = ev.run(
        batch=1024,
        d_model=tcfg.d_model,
        nhead=tcfg.nhead,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        drop=tcfg.drop
    )
    
    ev.save(df, mode="TEST")
