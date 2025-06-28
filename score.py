import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from functools import cached_property


class ResultLoader:
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = Path(__file__).parent / results_dir

    def raw_results(self, i: int, r: int) -> pd.DataFrame:
        file_path = self.results_dir / f"test_results_i{i}_r{r}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_parquet(file_path)

    def load_results(self, i: int, r: int) -> Dict[str, pd.DataFrame]:
        df = self.raw_results(i, r).reset_index(drop=False)
        df['ending_date'] = pd.to_datetime(
            df['ending_date'].astype(str), format='%Y%m%d')

        pivot_dict = {}
        for col in ['label', 'prediction', 'prob_down', 'prob_up']:
            pivot_df = df.pivot(index='ending_date',
                                columns='StockID', values=col)
            pivot_dict[col] = pivot_df
        return pivot_dict

    def available_models(self) -> List[Tuple[int, int]]:
        """Get list of available (i, r) model combinations"""
        models = []
        for file_path in self.results_dir.glob("test_results_i*_r*.parquet"):
            name = file_path.stem
            parts = name.replace("test_results_", "").split("_")
            i = int(parts[0][1:])
            r = int(parts[1][1:])
            models.append((i, r))
        return sorted(models)

    @cached_property
    def avg_prob(self) -> pd.DataFrame:
        prob_ups = []
        models = self.available_models()
        for i, r in tqdm(models, desc="Loading models"):
            try:
                data = self.load_results(i, r)
                prob_ups.append(data['prob_up'])
            except FileNotFoundError:
                print(f"Model I{i}/R{r} not found, skipping...")
                continue

        if len(prob_ups) < 2:
            raise ValueError("Need at least 2 models for average")

        common_dates = prob_ups[0].index
        common_ids = prob_ups[0].columns

        for prob_up in prob_ups[1:]:
            common_dates = common_dates.intersection(prob_up.index)
            common_ids = common_ids.intersection(prob_up.columns)

        print(
            f"Common dates: {len(common_dates)}, Common stocks: {len(common_ids)}")

        aligned_probs = []
        for prob_up in prob_ups:
            aligned = prob_up.loc[common_dates, common_ids]
            aligned_probs.append(aligned)

        avgs = sum(aligned_probs) / len(aligned_probs)
        return avgs

    def __call__(self, i: int, r: int = None) -> Dict[str, pd.DataFrame]:
        if r is None:
            r = i
        return self.load_results(i, r)


if __name__ == "__main__":
    loader = ResultLoader()
    print(loader.avg_prob)
