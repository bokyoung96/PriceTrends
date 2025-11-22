import logging
import sys
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.root import RESULTS_ROOT, SCORES_ROOT

logger = logging.getLogger(__name__)


class FusionResultRepository:
    def __init__(self, mode: str = "TEST") -> None:
        self.mode = mode.lower()
        self.base_dir = RESULTS_ROOT

    def fetch(self, input_days: int, return_days: int) -> pd.DataFrame:
        file_path = (
            self.base_dir
            / "fusion"
            / f"price_trends_results_{self.mode}_i{input_days}_r{return_days}_fusion.parquet"
        )
        if not file_path.exists():
            raise FileNotFoundError(
                f"No fusion result file found for I{input_days}/R{return_days} "
                f"(mode={self.mode}) in {self.base_dir}"
            )
        return pd.read_parquet(file_path)


class FusionResultAnalyzer:
    def __init__(self, repository: FusionResultRepository, pairs: Sequence[Tuple[int, int]]) -> None:
        normalized = list(self._normalize_pairs(pairs))
        if not normalized:
            raise ValueError("At least one (input_days, return_days) pair must be provided.")
        self.repo = repository
        self.pairs = normalized

    @staticmethod
    def _normalize_pairs(pairs: Sequence[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
        def to_pair(item: Tuple[int, int]) -> Tuple[int, int]:
            if isinstance(item, tuple) and len(item) == 2:
                return int(item[0]), int(item[1])
            raise ValueError(f"Invalid pair: {item}")

        if isinstance(pairs, tuple) and len(pairs) == 2 and all(
            isinstance(x, int) for x in pairs
        ):
            return [to_pair(pairs)]

        normalized: list[Tuple[int, int]] = []
        for item in pairs:
            normalized.append(to_pair(item))
        return normalized

    def _pivot_results(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        df = df.reset_index(drop=False)
        df["ending_date"] = pd.to_datetime(df["ending_date"].astype(str), format="%Y%m%d")
        columns = ["label", "prediction", "prob_down", "prob_up"]
        return {
            col: df.pivot(index="ending_date", columns="StockID", values=col) for col in columns
        }

    def load_pivots(self, input_days: int, return_days: int) -> Dict[str, pd.DataFrame]:
        df = self.repo.fetch(input_days, return_days)
        return self._pivot_results(df)

    def get_prob_up(self, input_days: int, return_days: int) -> pd.DataFrame:
        return self.load_pivots(input_days, return_days)["prob_up"]

    @cached_property
    def prob_up(self) -> Dict[Tuple[int, int], pd.DataFrame]:
        data: Dict[Tuple[int, int], pd.DataFrame] = {}
        for pair in self.pairs:
            try:
                data[pair] = self.get_prob_up(*pair)
            except FileNotFoundError as exc:
                logger.warning("Fusion model I%d/R%d not found: %s", pair[0], pair[1], exc)
        if not data:
            raise ValueError("No fusion probability data could be loaded for the provided pairs.")
        return data

    @cached_property
    def prob_up_avg(self) -> Optional[pd.DataFrame]:
        prob_ups = list(self.prob_up.values())
        if len(prob_ups) < 2:
            return None

        common_dates = prob_ups[0].index
        common_ids = prob_ups[0].columns
        for prob_up in prob_ups[1:]:
            common_dates = common_dates.intersection(prob_up.index)
            common_ids = common_ids.intersection(prob_up.columns)

        logger.info("Fusion common dates: %d, common stocks: %d", len(common_dates), len(common_ids))

        aligned = [prob_up.loc[common_dates, common_ids] for prob_up in prob_ups]
        return sum(aligned) / len(aligned)

    def save(
        self,
        output_dir: Optional[Path] = None,
        include_average: bool = True,
    ) -> Dict[Tuple[int, int], pd.DataFrame]:
        target_dir = output_dir or SCORES_ROOT
        target_dir.mkdir(parents=True, exist_ok=True)

        for (input_days, return_days), df in self.prob_up.items():
            out_path = (
                target_dir
                / f"price_trends_score_{self.repo.mode}_i{input_days}_r{return_days}_fusion.parquet"
            )
            df.to_parquet(out_path)
            logger.info("Saved fusion scores %s", out_path)

        if include_average:
            avg_df = self.prob_up_avg
            if avg_df is None:
                logger.info("Fusion average probability not available (need at least 2 models).")
            else:
                out_path = (
                    target_dir
                    / f"price_trends_score_{self.repo.mode}_ensemble_fusion.parquet"
                )
                df = avg_df
                df.to_parquet(out_path)
                logger.info("Saved fusion ensemble average %s", out_path)

        return self.prob_up


def main(
    mode: str = "TEST",
    pairs: Sequence[Tuple[int, int]] = ((5, 5), (20, 20), (60, 60)),
    include_average: bool = True,
) -> FusionResultAnalyzer:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    repo = FusionResultRepository(mode=mode)
    analyzer = FusionResultAnalyzer(repo, pairs)
    analyzer.save(include_average=include_average)
    return analyzer


if __name__ == "__main__":
    analyzer = main(mode='TEST',
                    pairs=((5, 5), (20, 20), (60, 60)),
                    include_average=True)

