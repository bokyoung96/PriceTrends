import logging
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.root import RESULTS_ROOT, SCORES_ROOT
from transformer.params import TransformerParams

logger = logging.getLogger(__name__)


class ScoreMaker:
    def __init__(self, mode: str = "TEST", name: str = "transformer"):
        self.mode = mode.lower()
        self.name = name
        self.res_dir = RESULTS_ROOT / self.name
        
    def run(self) -> pd.DataFrame:
        path = self.res_dir / f"price_trends_results_{self.name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No results found at {path}. Run evaluate.py first.")
            
        logger.info(f"Loading results from {path}")
        df = pd.read_parquet(path)
        
        scores = df.pivot(index="date", columns="asset", values="prob_up").sort_index()
        return scores
    
    def save(self, df: pd.DataFrame):
        SCORES_ROOT.mkdir(parents=True, exist_ok=True)
        path = SCORES_ROOT / f"price_trends_score_{self.name}.parquet"
        df.to_parquet(path)
        logger.info(f"Saved scores to {path}")
        return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    params = TransformerParams()
    tcfg = params.get_config(mode="TEST", timeframe="MEDIUM")
    
    name = f"transformer_{tcfg.mode}"
    
    maker = ScoreMaker(mode="TEST", name=name)
    scores = maker.run()
    maker.save(scores)
