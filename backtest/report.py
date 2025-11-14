from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import BacktestConfig
from .portfolio import TradeRecord


@dataclass(frozen=True)
class QuantileReport:
    """Holds time-series, trade records, and summary stats for one quantile."""

    quantile_id: int
    equity_curve: pd.Series
    period_returns: pd.Series
    trades: List[TradeRecord]
    stats: Dict[str, float]


@dataclass
class BacktestReport:
    """Aggregates QuantileReport objects and exposes convenience helpers."""

    config: BacktestConfig
    quantiles: Dict[int, QuantileReport]

    def equity_frame(self) -> pd.DataFrame:
        series = {f"q{qid}": rpt.equity_curve for qid, rpt in self.quantiles.items()}
        return pd.DataFrame(series).sort_index()

    def return_frame(self) -> pd.DataFrame:
        series = {f"q{qid}": rpt.period_returns for qid, rpt in self.quantiles.items()}
        return pd.DataFrame(series).sort_index()

    def summary_table(self) -> pd.DataFrame:
        table = pd.DataFrame(
            {
                f"q{qid}": rpt.stats
                for qid, rpt in self.quantiles.items()
            }
        ).T
        table.index.name = "quantile"
        return table

    def render_summary(self) -> str:
        table = self.summary_table()
        return table.to_string(float_format=lambda x: f"{x:0.4f}")

    def save(self, output_dir: Path | None = None) -> Path:
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.equity_frame().to_csv(out_dir / "equity_curve.csv")
        self.return_frame().to_csv(out_dir / "period_returns.csv")
        self.summary_table().to_csv(out_dir / "summary.csv")
        return out_dir

