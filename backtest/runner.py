from __future__ import annotations

import sys
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence, cast

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig
from backtest.data_sources import BacktestDataset, BacktestDatasetBuilder
from backtest.engine import BacktestEngine
from backtest.quantiles import BucketAllocator
from backtest.report import BacktestReport, BucketReport, SimulationReport

BucketSelector = str | int | Sequence[str | int] | None


class Backtester:
    """Object-oriented facade for running PriceTrends backtests."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._base_config = config or BacktestConfig()
        self._report: BacktestReport | None = None
        self._dataset: BacktestDataset | None = None
        self._batch_result: BatchResult | None = None

    def run(
        self,
        score_paths: Sequence[Path | str] | None = None,
        *,
        bucket: BucketSelector = None,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        **overrides: Any,
    ) -> BacktestReport:
        overrides = dict(overrides)
        explicit = score_paths or overrides.pop("score_paths", None)

        base_config = self._config_with(overrides)
        candidates = base_config.resolve_score_paths(explicit)
        if not candidates:
            raise ValueError("No score files available for the backtest.")

        if len(candidates) == 1:
            return self.run_single(
                scores_path=candidates[0],
                scores=scores,
                prices=prices,
                **overrides,
            )

        if scores is not None:
            raise ValueError("A scores DataFrame cannot be used when comparing multiple files.")
        executor = BatchBacktest(config=base_config, score_paths=candidates)
        result = executor.run(bucket=bucket)
        self._dataset = None
        self._batch_result = result
        self._report = result.report
        return result.report

    def run_single(
        self,
        *,
        scores_path: Path | str | None = None,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
        **overrides: Any,
    ) -> BacktestReport:
        overrides = dict(overrides)
        if scores_path is not None:
            overrides["scores_path"] = Path(scores_path)
        config = self._config_with(overrides)
        executor = SingleBacktest(config, scores=scores, prices=prices)
        report = executor.run()
        self._dataset = executor.dataset
        self._batch_result = None
        self._report = report
        return report

    def run_batch(
        self,
        score_paths: Sequence[Path | str] | None = None,
        *,
        bucket: BucketSelector = None,
        **overrides: Any,
    ) -> BacktestReport:
        overrides = dict(overrides)
        config = self._config_with(overrides)
        candidates = config.resolve_score_paths(score_paths)
        if len(candidates) < 2:
            raise ValueError("At least two score files are required for run_batch().")
        return self.run(score_paths=candidates, bucket=bucket, **overrides)

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
        if self._dataset is None:
            raise RuntimeError("daily_return_df is only available after a single-score run.")
        dataset = self._dataset
        equity = self.equity_df
        expanded = equity.reindex(dataset.dates).ffill()
        return expanded.pct_change().dropna()

    def _config_with(self, overrides: dict[str, Any]) -> BacktestConfig:
        if overrides:
            return self._base_config.with_overrides(**overrides)
        return self._base_config

    def _extract_frame(self, attr: Literal["scores", "prices"]) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if self._batch_result is not None:
            frames: dict[str, pd.DataFrame] = {}
            for label, run in self._batch_result.runs.items():
                dataset = run.dataset
                if dataset is None:
                    raise RuntimeError(f"Batch run for '{label}' has no dataset.")
                frames[label] = getattr(dataset, attr).copy()
            return frames
        if self._dataset is not None:
            return getattr(self._dataset, attr).copy()
        raise RuntimeError("Backtest has not been executed yet.")

    def batch_reports(self) -> dict[str, BacktestReport]:
        if self._batch_result is None:
            raise RuntimeError("No batch backtest has been executed yet.")
        reports: dict[str, BacktestReport] = {}
        for label, run in self._batch_result.runs.items():
            if run.report is None:
                raise RuntimeError(f"Batch run for '{label}' has no report.")
            reports[label] = run.report
        return reports


class SingleBacktest:
    """Executes a single backtest run for a given configuration."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        scores: pd.DataFrame | None = None,
        prices: pd.DataFrame | None = None,
    ) -> None:
        self.config = config
        self._scores = scores
        self._prices = prices
        self.dataset: BacktestDataset | None = None
        self.report: BacktestReport | None = None

    def run(self) -> BacktestReport:
        dataset = self._build_dataset()
        report = self._run_engine(dataset)
        self.dataset = dataset
        self.report = report
        return report

    def _build_dataset(self) -> BacktestDataset:
        self.config.ensure_io_paths(
            scores_in_memory=self._scores is not None,
            prices_in_memory=self._prices is not None,
        )
        builder = BacktestDatasetBuilder(
            scores_source=self._scores if self._scores is not None else self.config.scores_path,
            close_source=self._prices if self._prices is not None else self.config.close_path,
        )
        return builder.build()

    def _run_engine(self, dataset: BacktestDataset) -> BacktestReport:
        assigner = BucketAllocator(
            quantiles=self.config.quantiles,
            min_assets=self.config.min_assets,
            allow_partial=self.config.allow_partial_buckets,
        )
        engine = BacktestEngine(config=self.config, dataset=dataset, assigner=assigner)
        return engine.run()


@dataclass(frozen=True)
class BatchResult:
    report: BacktestReport
    runs: Mapping[str, SingleBacktest]


class BatchBacktest:
    """Executes multiple backtests and returns a combined comparison report."""

    def __init__(self, config: BacktestConfig, score_paths: Sequence[Path]) -> None:
        if len(score_paths) < 2:
            raise ValueError("BatchBacktest requires at least two score files.")
        self._base_config = config
        self._score_paths = [Path(p) for p in score_paths]

    def run(self, bucket: BucketSelector = None) -> BatchResult:
        runs: dict[str, SingleBacktest] = {}
        for path in self._score_paths:
            cfg = self._base_config.with_overrides(scores_path=path)
            single = SingleBacktest(cfg)
            single.run()
            tokens = [tok for tok in path.stem.split("_") if tok.lower().startswith(("i", "r"))]
            label = "_".join(tokens) if tokens else path.stem
            runs[label] = single
        combined = self.combine_reports(runs, bucket)
        return BatchResult(report=combined, runs=runs)

    def combine_reports(
        self,
        runs: Mapping[str, SingleBacktest],
        bucket: BucketSelector,
    ) -> BacktestReport:
        items: list[tuple[str, BacktestReport]] = []
        for label, run in runs.items():
            if run.report is None:
                raise RuntimeError(f"Batch member '{label}' did not produce a report.")
            items.append((label, run.report))
        if not items:
            raise RuntimeError("Batch comparison requires at least one completed run.")
        first_label, first_report = items[0]

        def parse_selector(value: str | int) -> int:
            if isinstance(value, int):
                return value
            token = value.strip().lower()
            if not token:
                raise ValueError("Bucket selector strings cannot be empty.")
            if token.startswith("q"):
                return max(0, int(token[1:]) - 1)
            return int(token)

        if bucket is None:
            bucket_ids = [max(first_report.quantiles.keys())]
        elif isinstance(bucket, SequenceABC) and not isinstance(bucket, (str, bytes)):
            bucket_ids = [parse_selector(sel) for sel in bucket]
            if not bucket_ids:
                raise ValueError("At least one bucket selector must be provided.")
        else:
            bucket_ids = [parse_selector(cast(str | int, bucket))]
        bucket_ids = list(dict.fromkeys(bucket_ids))
        bucket_tags = [f"q{idx + 1}" for idx in bucket_ids]
        run_tags = [label for label, _ in items]

        combined: Dict[int, BucketReport] = {}
        labels: Dict[int, str] = {}
        for label, report in items:
            for bucket_id in bucket_ids:
                if bucket_id not in report.quantiles:
                    raise ValueError(f"Bucket {bucket_id} not available in '{label}' report.")
                source = report.quantiles[bucket_id]
                idx = len(combined)
                combined[idx] = BucketReport(
                    bucket_id=idx,
                    equity_curve=source.equity_curve,
                    period_returns=source.period_returns,
                    trades=source.trades,
                    stats=source.stats,
                )
                bucket_tag = f"q{bucket_id + 1}"
                labels[idx] = f"{label}_{bucket_tag}"

        score_suffix = "_".join(run_tags)
        bucket_suffix = "_".join(bucket_tags)
        suffix_parts = [score_suffix] + ([bucket_suffix] if bucket_suffix else [])
        suffix = "_".join(part for part in suffix_parts if part)
        config = first_report.config.with_overrides(
            quantiles=len(combined),
            scores_path=Path(suffix),
        )
        bench = None
        for _, report in items:
            if report.bench_equity is not None and not report.bench_equity.empty:
                bench = report.bench_equity
                break
        return SimulationReport(config=config, quantiles=combined, bench_equity=bench, labels=labels)
