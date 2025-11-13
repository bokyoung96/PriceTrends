import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CNNConfig:
    mode: str
    train_years: List[int]
    ensem_size: int
    batch_size: int
    max_epoch: int
    lr: float
    drop_prob: float
    conv_channels: List[int]
    pw: int
    filter_sizes: List[Tuple[int, int]]
    test_years: Optional[List[int]] = None

    def with_test_years(self, years: List[int]) -> "CNNConfig":
        self.test_years = list(years)
        return self


class CNNParams:
    def __init__(self, config_path: str = None) -> None:
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_config(self, mode: str, window_size: int) -> CNNConfig:
        mode_config = self.config['mode_configs'][mode]
        window_config = self.config['window_configs'][str(window_size)]

        filter_sizes = [
            (int(kernel[0]), int(kernel[1]))
            for kernel in window_config['filter_sizes'][mode]
        ]

        return CNNConfig(
            mode=mode_config['mode'],
            train_years=list(mode_config['train_years']),
            ensem_size=int(mode_config['ensem_size']),
            batch_size=int(mode_config['batch_size']),
            max_epoch=int(mode_config['max_epoch']),
            lr=float(mode_config['lr']),
            drop_prob=float(mode_config['drop_prob']),
            conv_channels=list(mode_config['conv_channels']),
            pw=int(window_config['pw']),
            filter_sizes=filter_sizes,
        )

    def get_test_years(self) -> List[int]:
        return list(self.config['test_years'])

    def get_evaluation_pairs(self, mode: str) -> List[Tuple[int, int]]:
        eval_config = self.config.get('evaluation_windows', {})
        windows = eval_config.get(mode, self.window_sizes)
        pairs: List[Tuple[int, int]] = []
        for window in windows:
            key = str(window)
            window_cfg = self.config['window_configs'].get(key)
            if window_cfg is None:
                raise ValueError(f"Window size {window} is not defined in config.window_configs")
            prediction_window = window_cfg['pw']
            pairs.append((int(window), int(prediction_window)))
        return pairs

    @property
    def modes(self) -> List[str]:
        return list(self.config['mode_configs'].keys())

    @property
    def window_sizes(self) -> List[int]:
        return [int(ws) for ws in self.config['window_configs'].keys()]
