from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Sequence, Tuple

import pandas as pd

from backtest.quantiles import QuantileAllocator

GroupSelection = Tuple[str, ...]


@dataclass(frozen=True)
class PortfolioGroupDefinition:
    identifier: str
    label: str


@dataclass(frozen=True)
class PortfolioAllocationResult:
    selections: Dict[str, GroupSelection]
    total_assets: int
    skipped: bool = False
    message: str | None = None

    def tickers_for(self, identifier: str) -> GroupSelection:
        return self.selections.get(identifier, tuple())


class PortfolioGroupingStrategy(ABC):
    @abstractmethod
    def groups(self) -> Sequence[PortfolioGroupDefinition]:
        ...

    @abstractmethod
    def allocate(self, scores: pd.Series) -> PortfolioAllocationResult:
        ...


class QuantileGroupingStrategy(PortfolioGroupingStrategy):
    def __init__(
        self,
        *,
        quantiles: int,
        min_assets: int,
        allow_partial: bool = False,
        min_score: float | None = None,
        enabled_quantiles: Sequence[int] | None = None,
    ) -> None:
        self._allocator = QuantileAllocator(
            quantiles=quantiles,
            min_assets=min_assets,
            allow_partial=allow_partial,
            min_score=min_score,
        )
        if enabled_quantiles:
            ids = sorted({int(q) for q in enabled_quantiles})
        else:
            ids = list(range(quantiles))
        if not ids:
            raise ValueError("At least one quantile must be tracked by the grouping strategy.")
        for idx in ids:
            if idx < 0 or idx >= quantiles:
                raise ValueError(f"Quantile id {idx} is outside the range [0, {quantiles}).")
        self._quantile_ids = tuple(ids)
        self._label_map = {idx: f"q{idx + 1}" for idx in self._quantile_ids}

    def groups(self) -> Sequence[PortfolioGroupDefinition]:
        return [
            PortfolioGroupDefinition(identifier=self._label_map[idx], label=f"Quantile {idx + 1}")
            for idx in self._quantile_ids
        ]

    def allocate(self, scores: pd.Series) -> PortfolioAllocationResult:
        allocation = self._allocator.assign(scores)
        selections = {}
        for idx in self._quantile_ids:
            label = self._label_map[idx]
            selections[label] = allocation.tickers_for(idx)
        return PortfolioAllocationResult(
            selections=selections,
            total_assets=allocation.total_assets,
            skipped=allocation.skipped,
            message=allocation.reason,
        )


class FixedCountGroupingStrategy(PortfolioGroupingStrategy):
    def __init__(
        self,
        *,
        quantiles: int,
        top_n: int,
        bottom_n: int,
        min_assets: int,
        allow_partial: bool = False,
        min_score: float | None = None,
        enabled_quantiles: Sequence[int] | None = None,
    ) -> None:
        if quantiles < 2:
            raise ValueError("Fixed-count grouping requires quantiles >= 2.")
        if top_n <= 0 or bottom_n <= 0:
            raise ValueError("top_n and bottom_n must be positive integers.")
        self.quantiles = int(quantiles)
        self.top_n = int(top_n)
        self.bottom_n = int(bottom_n)
        self.min_assets = int(min_assets)
        self.allow_partial = bool(allow_partial)
        self.min_score = min_score

        bottom_id = 0
        top_id = self.quantiles - 1
        if enabled_quantiles:
            ids = sorted({int(q) for q in enabled_quantiles})
        else:
            ids = [bottom_id, top_id]
        allowed = {bottom_id, top_id}
        for idx in ids:
            if idx not in allowed:
                raise ValueError(
                    "Fixed-count grouping only supports the bottom (q1) and top (qN) groups."
                )
        self._quantile_ids = tuple(ids)
        self._label_map = {bottom_id: "q1", top_id: f"q{self.quantiles}"}

    def groups(self) -> Sequence[PortfolioGroupDefinition]:
        return [
            PortfolioGroupDefinition(identifier=self._label_map[idx], label=f"Quantile {idx + 1}")
            for idx in self._quantile_ids
        ]

    def allocate(self, scores: pd.Series) -> PortfolioAllocationResult:
        clean_scores = scores.dropna()
        if self.min_score is not None:
            clean_scores = clean_scores[clean_scores >= self.min_score]
        asset_count = len(clean_scores)
        if asset_count == 0:
            return self._empty_allocation("No assets had valid scores.")

        required = self.top_n + self.bottom_n
        if asset_count < required:
            return self._empty_allocation(
                f"Asset count {asset_count} fell below required {required} for top/bottom selection.",
                total=asset_count,
            )
        if not self.allow_partial and asset_count < self.min_assets:
            return self._empty_allocation(
                f"Asset count {asset_count} fell below minimum {self.min_assets}.",
                total=asset_count,
            )

        sorted_scores = clean_scores.sort_values(ascending=True, kind="mergesort")
        bottom = sorted_scores.iloc[: self.bottom_n]
        top = sorted_scores.iloc[-self.top_n :]
        selections = {self._label_map[idx]: tuple() for idx in self._quantile_ids}
        bottom_id = 0
        top_id = self.quantiles - 1
        if bottom_id in self._quantile_ids:
            selections[self._label_map[bottom_id]] = tuple(str(t) for t in bottom.index)
        if top_id in self._quantile_ids:
            selections[self._label_map[top_id]] = tuple(str(t) for t in top.index)

        return PortfolioAllocationResult(
            selections=selections,
            total_assets=asset_count,
            skipped=False,
            message=None,
        )

    def _empty_allocation(self, reason: str, total: int = 0) -> PortfolioAllocationResult:
        selections = {self._label_map[idx]: tuple() for idx in self._quantile_ids}
        return PortfolioAllocationResult(
            selections=selections,
            total_assets=total,
            skipped=True,
            message=reason,
        )


class ExplicitGroupingStrategy(PortfolioGroupingStrategy):
    def __init__(
        self,
        selectors: Mapping[str, Callable[[pd.Series], Sequence[str]]],
        *,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        if not selectors:
            raise ValueError("ExplicitGroupingStrategy requires at least one selector.")
        self._selectors = {key: selectors[key] for key in selectors}
        self._labels = dict(labels or {})

    def groups(self) -> Sequence[PortfolioGroupDefinition]:
        items = []
        for identifier in self._selectors:
            label = self._labels.get(identifier, identifier)
            items.append(PortfolioGroupDefinition(identifier=identifier, label=label))
        return items

    def allocate(self, scores: pd.Series) -> PortfolioAllocationResult:
        selections: Dict[str, GroupSelection] = {}
        clean_scores = scores.dropna()
        for identifier, selector in self._selectors.items():
            tickers = selector(clean_scores)
            selections[identifier] = tuple(str(t) for t in tickers)
        return PortfolioAllocationResult(selections=selections, total_assets=len(clean_scores))


class SectorNeutralGroupingStrategy(PortfolioGroupingStrategy):
    def __init__(
        self,
        sector_panel: pd.DataFrame,
        inner_strategy: PortfolioGroupingStrategy,
    ) -> None:
        self.sector_panel = sector_panel
        self.inner_strategy = inner_strategy

    def groups(self) -> Sequence[PortfolioGroupDefinition]:
        return self.inner_strategy.groups()

    def allocate(self, scores: pd.Series) -> PortfolioAllocationResult:
        date = scores.name
        if not isinstance(date, pd.Timestamp):
            return PortfolioAllocationResult(
                selections={g.identifier: tuple() for g in self.groups()},
                total_assets=0,
                skipped=True,
                message="Cannot determine date from scores Series to look up sectors.",
            )

        try:
            if date in self.sector_panel.index:
                sector_map = self.sector_panel.loc[date]
            else:
                idx = self.sector_panel.index.searchsorted(date, side='right') - 1
                if idx < 0:
                    return PortfolioAllocationResult(
                        selections={g.identifier: tuple() for g in self.groups()},
                        total_assets=0,
                        skipped=True,
                        message=f"No sector data available before {date}.",
                    )
                sector_map = self.sector_panel.iloc[idx]
        except Exception as e:
             return PortfolioAllocationResult(
                selections={g.identifier: tuple() for g in self.groups()},
                total_assets=0,
                skipped=True,
                message=f"Error looking up sector data: {e}",
            )

        valid_tickers = scores.index.intersection(sector_map.index)
        if valid_tickers.empty:
            return PortfolioAllocationResult(
                selections={g.identifier: tuple() for g in self.groups()},
                total_assets=0,
                skipped=True,
                message="No tickers found in sector map.",
            )

        aggregated_selections: Dict[str, list[str]] = {g.identifier: [] for g in self.groups()}
        total_assets = 0
        active_sectors = sector_map.loc[valid_tickers]
        
        for sector, sector_tickers in active_sectors.groupby(active_sectors):
            sector_scores = scores.loc[sector_tickers.index]
            sector_result = self.inner_strategy.allocate(sector_scores)
            
            if not sector_result.skipped:
                for group_id, tickers in sector_result.selections.items():
                    aggregated_selections[group_id].extend(tickers)
                total_assets += sector_result.total_assets

        final_selections = {k: tuple(v) for k, v in aggregated_selections.items()}
        
        return PortfolioAllocationResult(
            selections=final_selections,
            total_assets=total_assets,
            skipped=total_assets == 0,
            message=None if total_assets > 0 else "No assets allocated across sectors.",
        )
