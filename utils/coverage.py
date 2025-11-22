from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.model1 import KoreanEquityDataset
from core.models.model2 import ForeignDataset


@dataclass
class CoverageSummary:
    image_tickers: set[str]
    foreign_tickers: set[str]

    @property
    def intersection(self) -> set[str]:
        return self.image_tickers & self.foreign_tickers

    @property
    def missing_in_foreign(self) -> set[str]:
        return self.image_tickers - self.foreign_tickers

    @property
    def missing_in_image(self) -> set[str]:
        return self.foreign_tickers - self.image_tickers


class FusionCoverageAnalyzer:
    def __init__(
        self,
        ws: int,
        years: Optional[Sequence[int]] = None,
        foreign_windows: Sequence[int] = (5, 20, 60),
    ) -> None:
        if years is None:
            years = list(range(2010, 2030))
        self.ws = ws
        self.years: list[int] = list(years)
        self.foreign_windows = tuple(sorted(set(foreign_windows)))

        self._image_meta: Optional[pd.DataFrame] = None
        self._foreign_long: Optional[pd.DataFrame] = None

    @property
    def image_meta(self) -> pd.DataFrame:
        if self._image_meta is None:
            ds = KoreanEquityDataset(intervals=self.ws, years=self.years)
            meta = ds.metadata.copy()
            meta["end_date"] = pd.to_datetime(meta["end_date"].astype(str)).dt.normalize()
            self._image_meta = meta
        return self._image_meta

    @property
    def foreign_long(self) -> pd.DataFrame:
        if self._foreign_long is None:
            foreign_ds = ForeignDataset(
                windows=self.foreign_windows,
                years=self.years,
            )
            self._foreign_long = foreign_ds.get_frame()
        return self._foreign_long

    def date_counts(self) -> Tuple[pd.Series, pd.Series]:
        img_dates = self.image_meta["end_date"]
        foreign_dates = pd.to_datetime(
            self.foreign_long.index.get_level_values("date")
        ).normalize()

        img_counts = img_dates.value_counts().sort_index()
        foreign_counts = foreign_dates.value_counts().sort_index()
        return img_counts, foreign_counts

    def ticker_summary(self) -> CoverageSummary:
        img_tickers = set(self.image_meta["ticker"].astype(str).unique())
        foreign_tickers = set(
            self.foreign_long.index.get_level_values("ticker").astype(str).unique()
        )
        return CoverageSummary(image_tickers=img_tickers, foreign_tickers=foreign_tickers)

    def plot_date_coverage(
        self,
        ax: Optional[Axes] = None,
        show: bool = True,
        limit_dates: Optional[Iterable[pd.Timestamp]] = None,
    ) -> Axes:
        img_counts, foreign_counts = self.date_counts()

        if limit_dates is not None:
            limit_index = pd.to_datetime(list(limit_dates))
            img_counts = img_counts.loc[img_counts.index.intersection(limit_index)]
            foreign_counts = foreign_counts.loc[foreign_counts.index.intersection(limit_index)]

        if ax is None:
            fig: Figure
            fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(img_counts.index, img_counts.values, label="Image samples per day")
        ax.plot(foreign_counts.index, foreign_counts.values, label="Foreign samples per day")
        ax.set_title(f"Date Coverage Comparison (ws={self.ws})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sample Count")
        ax.legend()
        ax.grid(True)

        if show:
            plt.show()
        return ax

    def plot_ticker_coverage(
        self,
        ax: Optional[Axes] = None,
        show: bool = True,
    ) -> Axes:
        summary = self.ticker_summary()
        img_count = len(summary.image_tickers)
        foreign_count = len(summary.foreign_tickers)

        if ax is None:
            fig: Figure
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(["Image tickers", "Foreign tickers"], [img_count, foreign_count])
        ax.set_title(f"Ticker Coverage Comparison (ws={self.ws})")
        ax.set_ylabel("Count")

        if show:
            plt.show()
        return ax

    def print_summary(self) -> CoverageSummary:
        summary = self.ticker_summary()
        intersect = summary.intersection

        print("\n========== COVERAGE SUMMARY ==========")
        print(f"Image tickers:          {len(summary.image_tickers):,}")
        print(f"Foreign tickers:        {len(summary.foreign_tickers):,}")
        print(f"Intersection tickers:   {len(intersect):,}")
        print(f"Missing in foreign:     {len(summary.missing_in_foreign):,}")
        print(f"Missing in image:       {len(summary.missing_in_image):,}")
        print("=======================================\n")

        return summary

    def visualize_all(self, show: bool = True) -> CoverageSummary:
        self.plot_date_coverage(show=show)
        self.plot_ticker_coverage(show=show)
        return self.print_summary()


if __name__ == "__main__":
    analyzer = FusionCoverageAnalyzer(ws=5, years=[2000, 2001, 2002, 2003, 2004, 2005,
                                                   2006, 2007, 2008, 2009, 2010, 2011])
    analyzer.visualize_all(show=True)
