import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class TransformerConfig:
    mode: str
    train_years: List[int]
    test_years: List[int]
    batch_size: int
    max_epoch: int
    lr: float
    lookback: int
    stride: int
    horizon: int
    min_assets: int
    d_model: int
    nhead: int
    n_layers: int
    d_ff: int
    drop: float
    features: List[str]
    norm: str
    label_type: str
    threshold: float


class TransformerParams:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def get_config(self, mode: str = "TEST", timeframe: str = "MEDIUM") -> TransformerConfig:
        mode_config = self.config['mode_configs'][mode]
        timeframe_config = self.config['timeframe_configs'][timeframe]
        model_config = timeframe_config['model']
        
        return TransformerConfig(
            mode=f"{mode.lower()}_{timeframe.lower()}",
            train_years=list(mode_config['train_years']),
            test_years=list(mode_config['test_years']),
            batch_size=int(mode_config['batch_size']),
            max_epoch=int(mode_config['max_epoch']),
            lr=float(mode_config['lr']),
            lookback=int(timeframe_config['lookback']),
            stride=int(timeframe_config['stride']),
            horizon=int(timeframe_config['horizon']),
            min_assets=int(timeframe_config['min_assets']),
            d_model=int(model_config['d_model']),
            nhead=int(model_config['nhead']),
            n_layers=int(model_config['n_layers']),
            d_ff=int(model_config['d_ff']),
            drop=float(model_config['drop']),
            features=list(self.config['features']),
            norm=str(self.config['norm']),
            label_type=str(self.config['label_type']),
            threshold=float(self.config['threshold'])
        )
    
    @property
    def modes(self) -> List[str]:
        return list(self.config['mode_configs'].keys())
    
    @property
    def timeframes(self) -> List[str]:
        return list(self.config['timeframe_configs'].keys())


def build_name(mode: str, model_type: str, base: str = "transformer") -> str:
    suffix = "_multi" if model_type.lower() == "multi" else ""
    return f"{base}_{mode.lower()}{suffix}"
