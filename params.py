import json
import os
from typing import Dict, List, Tuple, Any


class CNNParams:
    def __init__(self, config_path: str = None) -> None:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), 'config.json')

        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_config(self, mode: str, window_size: int) -> Dict[str, Any]:
        mode_config = self.config['mode_configs'][mode].copy()
        window_config = self.config['window_configs'][str(window_size)].copy()

        config = mode_config.copy()
        config.update(window_config)
        config['filter_sizes'] = [
            tuple(fs) for fs in window_config['filter_sizes'][mode]]
        return config

    def get_test_years(self) -> List[int]:
        return self.config['test_years']

    @property
    def modes(self) -> List[str]:
        return list(self.config['mode_configs'].keys())

    @property
    def window_sizes(self) -> List[int]:
        return [int(ws) for ws in self.config['window_configs'].keys()]
