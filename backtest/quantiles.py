from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileAllocation:
    groups: Dict[int, Tuple[str, ...]]
    total_assets: int
    skipped: bool = False
    reason: str | None = None

    def tickers_for(self, bucket: int) -> Tuple[str, ...]:
        return self.groups.get(bucket, tuple())

    def has_assets(self) -> bool:
        return not self.skipped and self.total_assets > 0


class QuantileAllocator:
    def __init__(self, quantiles: int, min_assets: int, allow_partial: bool = False, min_score: float | None = None) -> None:
        if quantiles < 1:
            raise ValueError("At least one quantile is required.")
        self.quantiles = quantiles
        self.min_assets = min_assets
        self.allow_partial = allow_partial
        self.min_score = min_score

    def assign(self, scores: pd.Series) -> QuantileAllocation:
        clean_scores = scores.dropna()
        asset_count = len(clean_scores)
        if asset_count == 0:
            return self._empty_allocation(reason="No assets had valid scores.")

        if self.min_score is not None:
            clean_scores = clean_scores[clean_scores >= self.min_score]
            asset_count = len(clean_scores)
            if asset_count == 0:
                return self._empty_allocation(
                    reason=f"No assets met minimum score >= {self.min_score}.",
                    total=0,
                )

        if asset_count < self.quantiles:
            return self._empty_allocation(reason="Not enough assets to form quantiles.")

        min_required = self.quantiles if self.allow_partial else self.min_assets
        if asset_count < min_required:
            return self._empty_allocation(
                reason=f"Asset count {asset_count} fell below minimum {min_required}.",
                total=asset_count,
            )

        ranks = clean_scores.rank(method="first")
        percentiles = ranks / asset_count
        edges = np.linspace(0, 1, self.quantiles + 1)
        bucket_ids = np.digitize(percentiles.to_numpy(), edges[1:-1], right=True)

        working: Dict[int, list[str]] = {i: [] for i in range(self.quantiles)}
        for ticker, bucket in zip(clean_scores.index, bucket_ids):
            working[int(bucket)].append(str(ticker))
        groups = {bucket: tuple(tickers) for bucket, tickers in working.items()}
        return QuantileAllocation(groups=groups, total_assets=asset_count)

    def _empty_allocation(self, reason: str, total: int = 0) -> QuantileAllocation:
        return QuantileAllocation(
            groups={i: tuple() for i in range(self.quantiles)},
            total_assets=total,
            skipped=True,
            reason=reason,
        )
