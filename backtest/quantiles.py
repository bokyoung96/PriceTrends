from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileBuckets:
    """Stores the tickers assigned to each quantile for a single rebalance date."""

    labels: Dict[int, Tuple[str, ...]]
    total_assets: int
    skipped: bool = False
    reason: str | None = None

    def tickers_for(self, bucket: int) -> Tuple[str, ...]:
        return self.labels.get(bucket, tuple())

    def has_assets(self) -> bool:
        return not self.skipped and self.total_assets > 0


class QuantileAssigner:
    """Splits cross-sectional scores into equally populated buckets."""

    def __init__(self, quantiles: int, min_assets: int, allow_partial: bool = False) -> None:
        self.quantiles = quantiles
        self.min_assets = min_assets
        self.allow_partial = allow_partial

    def assign(self, scores: pd.Series) -> QuantileBuckets:
        clean_scores = scores.dropna()
        asset_count = len(clean_scores)
        min_required = self.quantiles if self.allow_partial else self.min_assets

        if asset_count < self.quantiles:
            return QuantileBuckets(
                labels={i: tuple() for i in range(self.quantiles)},
                total_assets=asset_count,
                skipped=True,
                reason="Not enough assets to form quantiles.",
            )

        if asset_count < min_required:
            return QuantileBuckets(
                labels={i: tuple() for i in range(self.quantiles)},
                total_assets=asset_count,
                skipped=True,
                reason=f"Asset count {asset_count} fell below minimum {min_required}.",
            )

        ranks = clean_scores.rank(method="first")
        percentiles = ranks / asset_count
        edges = np.linspace(0, 1, self.quantiles + 1)
        bucket_ids = np.digitize(percentiles.to_numpy(), edges[1:-1], right=True)

        working: Dict[int, list[str]] = {i: [] for i in range(self.quantiles)}
        for ticker, bucket in zip(clean_scores.index, bucket_ids):
            working[int(bucket)].append(ticker)
        bucket_map = {k: tuple(v) for k, v in working.items()}

        return QuantileBuckets(
            labels=bucket_map,
            total_assets=asset_count,
            skipped=False,
            reason=None,
        )
