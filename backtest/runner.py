from __future__ import annotations

import sys
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig
from backtest.data_sources import BacktestDataset
from backtest.engine import BacktestEngine
from backtest.report import BacktestReport, PortfolioReport

GroupSelector = str | int | Sequence[str | int] | None
ScoreInput = Path | str | Sequence[Path | str] | None
LABEL_TOKEN_PREFIXES = ("test", "origin", "fusion", "i", "r", "transformer", "short", "medium", "long", "lp")


def _label_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    tokens: list[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        lower = part.lower()
        if lower.startswith(LABEL_TOKEN_PREFIXES):
            if lower == "lp" and i + 1 < len(parts) and parts[i + 1].isdigit():
                tokens.append(f"{part}_{parts[i + 1]}")
                i += 2
                continue
            tokens.append(part)
        i += 1
    return "_".join(tokens) if tokens else path.stem


def _parse_score_inputs(value: ScoreInput) -> tuple[Path, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return tuple(Path(item) for item in value)
    return (Path(value),)


@dataclass
class BacktestJob:
    label: str
    config: BacktestConfig
    scores: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    dataset: BacktestDataset | None = None
    report: BacktestReport | None = None

    def execute(self) -> BacktestReport:
        loader = self.config.data_loader(scores=self.scores, prices=self.prices)
        dataset = loader.build()
        engine = BacktestEngine(config=self.config, dataset=dataset)
        report = engine.run()
        self.dataset = dataset
        self.report = report
        return report


class BacktestSuite:
    def __init__(self, jobs: Sequence[BacktestJob], group_selector: GroupSelector = None) -> None:
        if not jobs:
            raise ValueError("At least one job must be provided.")
        self.jobs = list(jobs)
        self.group_selector = group_selector
        self.report: BacktestReport | None = None

    def run(self) -> BacktestReport:
        for job in self.jobs:
            job.execute()
        if len(self.jobs) == 1:
            report = self.jobs[0].report
            if report is None:
                raise RuntimeError("Job execution did not yield a report.")
            self.report = report
            return report
        combined = self.combine_jobs(self.jobs, self.group_selector)
        self.report = combined
        return combined

    @staticmethod
    def combine_jobs(jobs: Sequence[BacktestJob], selector: GroupSelector) -> BacktestReport:
        items: list[tuple[str, BacktestReport]] = []
        for job in jobs:
            if job.report is None:
                raise RuntimeError(f"Job '{job.label}' has no report to combine.")
            items.append((job.label, job.report))
        if not items:
            raise RuntimeError("No completed jobs available for comparison.")
        first_label, first_report = items[0]
        group_ids = BacktestSuite._select_group_ids(selector, first_report)

        combined: Dict[str, PortfolioReport] = {}
        labels: Dict[str, str] = {}
        for label, report in items:
            for group_id in group_ids:
                if group_id not in report.groups:
                    raise ValueError(f"Group '{group_id}' not available in '{label}' report.")
                source = report.groups[group_id]
                composite_id = f"{label}_{group_id}_{len(combined)}"
                combined[composite_id] = PortfolioReport(
                    group_id=composite_id,
                    equity_curve=source.equity_curve,
                    period_returns=source.period_returns,
                    trades=source.trades,
                    stats=source.stats,
                )
                labels[composite_id] = f"{label}:{group_id}"

        score_suffix = "_".join(label for label, _ in items)
        group_suffix = "_".join(group_ids)
        suffix = "_".join(part for part in (score_suffix, group_suffix) if part)
        combined_config = first_report.config.with_overrides(
            scores_path=Path(suffix),
        )
        bench = None
        for _, report in items:
            if report.bench_equity is not None and not report.bench_equity.empty:
                bench = report.bench_equity
                break
        return BacktestReport(config=combined_config, groups=combined, bench_equity=bench, labels=labels)

    @staticmethod
    def _select_group_ids(selector: GroupSelector, report: BacktestReport) -> list[str]:
        available = sorted(report.groups.keys())
        if not available:
            raise ValueError("Report contains no groups to select from.")

        def parse(value: str | int) -> str:
            if isinstance(value, int):
                return f"q{value + 1}"
            token = str(value).strip()
            if not token:
                raise ValueError("Group selector strings cannot be empty.")
            return token

        if selector is None:
            return [available[-1]]
        if isinstance(selector, Sequence) and not isinstance(selector, (str, bytes)):
            group_ids = [parse(v) for v in selector]
            if not group_ids:
                raise ValueError("At least one group selector must be provided.")
        else:
            group_ids = [parse(selector)]
        unique: list[str] = []
        for value in group_ids:
            if value in unique:
                continue
            unique.append(value)
        for value in unique:
            if value not in available:
                raise ValueError(f"Group '{value}' not found in the source report.")
        return unique


class Backtester:
    """High-level facade for running one or multiple PriceTrends backtests."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._base_config = config or BacktestConfig()
        self._jobs: list[BacktestJob] = []
        self._suite: BacktestSuite | None = None
        self._report: BacktestReport | None = None

    def run(
        self,
        score_paths: ScoreInput = None,
        *,
        group_selector: GroupSelector = None,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        **overrides: Any,
    ) -> BacktestReport:
        overrides = dict(overrides)
        parsed_paths = _parse_score_inputs(score_paths)
        config = self._config_with(overrides, parsed_paths)
        jobs = self._build_jobs(config, scores=scores, prices=prices)
        suite = BacktestSuite(jobs, group_selector)
        report = suite.run()
        self._jobs = jobs
        self._suite = suite
        self._report = report
        return report

    def run_single(
        self,
        *,
        scores_path: Path | str | None = None,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        group_selector: GroupSelector = None,
        **overrides: Any,
    ) -> BacktestReport:
        return self.run(score_paths=scores_path, group_selector=group_selector, scores=scores, prices=prices, **overrides)

    def run_batch(
        self,
        score_paths: ScoreInput = None,
        *,
        group_selector: GroupSelector = None,
        **overrides: Any,
    ) -> BacktestReport:
        overrides = dict(overrides)
        provided = score_paths if score_paths is not None else self._base_config.scores_path
        parsed = _parse_score_inputs(provided)
        if parsed is None or len(parsed) < 2:
            raise ValueError("At least two score files are required for run_batch().")
        return self.run(score_paths=parsed, group_selector=group_selector, **overrides)

    def latest_report(self) -> BacktestReport:
        if self._report is None:
            raise RuntimeError("Backtest has not been executed yet.")
        return self._report

    def save_results(self, output_dir=None, filename: str | None = None):
        return self.latest_report().save(output_dir=output_dir, filename=filename)

    def summary(self) -> str:
        return self.latest_report().render_summary()

    @property
    def score_df(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._extract_frame("scores")

    @property
    def price_df(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        return self._extract_frame("prices")

    @property
    def hit_rate_df(self) -> pd.DataFrame:
        summary = self.latest_report().summary_table()
        win = summary["win_rate"].rename("hit_rate")
        return win.to_frame()

    @property
    def cumulative_returns(self) -> pd.Series:
        summary = self.latest_report().summary_table()
        return summary["total_return"]

    @property
    def equity_df(self) -> pd.DataFrame:
        return self.latest_report().equity_frame()

    @property
    def period_return_df(self) -> pd.DataFrame:
        return self.latest_report().return_frame()

    @property
    def daily_return_df(self) -> pd.DataFrame:
        return self.latest_report().daily_return_frame()

    @property
    def monthly_return_df(self) -> pd.DataFrame | None:
        report = self.latest_report()

        matrices = {}
        for group_id, group_report in report.groups.items():
             matrix = report._monthly_return_matrix(group_report.equity_curve, months=1200)
             if matrix is not None:
                 matrices[group_id] = matrix
        
        if report.bench_equity is not None and not report.bench_equity.empty:
            bench_matrix = report._monthly_return_matrix(report.bench_equity, months=1200)
            if bench_matrix is not None:
                matrices["benchmark"] = bench_matrix

        return pd.concat(matrices, axis=0) if matrices else None
    def daily_pnl_df(self) -> pd.DataFrame:
        return self.latest_report().daily_pnl_frame()

    def batch_reports(self) -> dict[str, BacktestReport]:
        if self._suite is None or len(self._jobs) < 2:
            raise RuntimeError("No batch backtest has been executed yet.")
        reports: dict[str, BacktestReport] = {}
        for job in self._jobs:
            if job.report is None:
                raise RuntimeError(f"Batch run for '{job.label}' has no report.")
            reports[job.label] = job.report
        return reports

    def _config_with(
        self,
        overrides: dict[str, Any],
        score_paths: tuple[Path, ...] | None,
    ) -> BacktestConfig:
        updates = dict(overrides)
        if score_paths is not None:
            updates["scores_path"] = score_paths if len(score_paths) != 1 else score_paths[0]
        if updates:
            return self._base_config.with_overrides(**updates)
        return self._base_config

    def _build_jobs(
        self,
        config: BacktestConfig,
        *,
        scores: pd.DataFrame | None,
        prices: pd.DataFrame | None,
    ) -> list[BacktestJob]:
        paths = list(config.scores_path)
        if not paths:
            raise ValueError("No score files available for the backtest.")
        if scores is not None and len(paths) > 1:
            raise ValueError("A scores DataFrame cannot be used when comparing multiple files.")

        jobs: list[BacktestJob] = []
        score_payload = scores
        price_payload = prices
        for path in paths:
            label = _label_from_path(path)
            job_config = config.with_overrides(scores_path=path)
            jobs.append(
                BacktestJob(
                    label=label,
                    config=job_config,
                    scores=score_payload,
                    prices=price_payload,
                )
            )
            score_payload = None
            price_payload = None
        return jobs

    def _extract_frame(self, attr: Literal["scores", "prices"]) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._jobs:
            raise RuntimeError("Backtest has not been executed yet.")
        if len(self._jobs) == 1:
            dataset = self._jobs[0].dataset
            if dataset is None:
                raise RuntimeError("Single run dataset is not available.")
            return getattr(dataset, attr).copy()
        frames: dict[str, pd.DataFrame] = {}
        for job in self._jobs:
            dataset = job.dataset
            if dataset is None:
                raise RuntimeError(f"Batch run for '{job.label}' has no dataset.")
            frames[job.label] = getattr(dataset, attr).copy()
        return frames
