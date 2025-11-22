import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from utils.root import MODELS_ROOT, SCORES_ROOT
from transformer.model import Transformer
from transformer.pipeline import Config, Pipeline, StockDataset

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, cfg: Config, name: str = "transformer_v1"):
        self.cfg = cfg
        self.name = name
        
        sel = DeviceSelector()
        self.dev = sel.resolve()
        
        self.dir = MODELS_ROOT / self.name
        self.path = self.dir / "best.pth"
        
    def run(self, batch: int = 256, d_model: int = 64, nhead: int = 4, 
            n_layers: int = 2, d_ff: int = 128, drop: float = 0.1) -> pd.DataFrame:
        pipe = Pipeline(self.cfg)
        path = pipe.get_path()
        
        if path.exists():
            logger.info(f"Loading {path}")
            wins = pipe.run()
        else:
            logger.info("Building...")
            wins = pipe.run()
            
        ds = StockDataset(wins)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
        
        sample = ds[0]['input']
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
            
        model.load_state_dict(torch.load(self.path, map_location=self.dev))
        model.eval()
        
        scores, dates, assets = [], [], []
        
        with torch.no_grad():
            for b in tqdm(loader, desc="Eval"):
                x = b['input'].to(self.dev)
                d = b['date']
                a = b['asset']
                
                out = model(x)
                prob = torch.softmax(out, dim=1)
                s = prob[:, 1].cpu().numpy()
                
                scores.extend(s)
                dates.extend(d)
                assets.extend(a)
                
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "asset": assets,
            "score": scores
        })
        
        return df.pivot(index="date", columns="asset", values="score").sort_index()
    
    def save(self, df: pd.DataFrame, name: str):
        SCORES_ROOT.mkdir(parents=True, exist_ok=True)
        path = SCORES_ROOT / name
        df.to_parquet(path)
        logger.info(f"Saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from transformer.params import TransformerParams
    
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
    
    ev = Evaluator(cfg, name=f"transformer_{tcfg.mode}")
    df = ev.run(
        batch=tcfg.batch_size,
        d_model=tcfg.d_model,
        nhead=tcfg.nhead,
        n_layers=tcfg.n_layers,
        d_ff=tcfg.d_ff,
        drop=tcfg.drop
    )
    ev.save(df, f"price_trends_score_transformer_{tcfg.mode}_lb{tcfg.lookback}_hz{tcfg.horizon}.parquet")
