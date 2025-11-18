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
        enabled_quantiles: Sequence[int] | None = None,
    ) -> None:
        self._allocator = QuantileAllocator(quantiles=quantiles, min_assets=min_assets, allow_partial=allow_partial)
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
