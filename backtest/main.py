from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import (BacktestConfig, BenchmarkType, score_path,
                             transformer_score_path)
from backtest.runner import Backtester
from backtest.validate import run_validation_example
from core.spec import MarketUniverse

logger = logging.getLogger(__name__)
DEFAULT_UNIVERSE: MarketUniverse | None = MarketUniverse.KOSPI200

LAST_RUNNER: "ExampleRunner | None" = None
LAST_RESULTS: Dict[str, Backtester] = {}


def _build_config(portfolio_weighting: str = "mc", **overrides) -> BacktestConfig:
    base: dict[str, object] = {}
    base["constituent_universe"] = DEFAULT_UNIVERSE
    base["portfolio_weighting"] = portfolio_weighting
    base.update(overrides)
    return BacktestConfig(**base)


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    scores: Tuple[Path, ...]
    group_selector: Any | None = None
    overrides: Dict[str, object] = field(default_factory=dict)
    log_summary: bool = True
    save_report: bool = True
    sector_neutral: bool = False
    output_filename: str | None = None
    label_prefix: str | None = None


class ExampleRunner:
    def __init__(self, base_opts: Dict[str, object] | None = None) -> None:
        self.base_opts = base_opts or {}
        self.latest: Backtester | None = None

    def run_spec(self, spec: ExampleSpec) -> Backtester:
        cfg_kwargs = dict(self.base_opts)
        cfg_kwargs.update(spec.overrides)
        cfg_kwargs["scores_path"] = spec.scores
        cfg_kwargs["sector_neutral"] = spec.sector_neutral
        if spec.label_prefix:
            cfg_kwargs["label_prefix"] = spec.label_prefix
        cfg = _build_config(**cfg_kwargs)
        tester = Backtester(cfg)
        report = tester.run(group_selector=spec.group_selector)
        if spec.log_summary:
            logger.info("%s summary:\n%s", spec.name, report.summary_table())
        if spec.save_report:
            output_path = report.save(filename=spec.output_filename)
            logger.info("Saved %s report to %s", spec.name, output_path)
        self.latest = tester
        return tester

    def run_named(self, name: str) -> Backtester:
        if name not in EXAMPLES:
            raise KeyError(f"Unknown example: {name}")
        return self.run_spec(EXAMPLES[name])


def _cnn_scores(input_days: int = 20, return_days: int = 20) -> Tuple[Path, ...]:
    return (
        score_path(input_days, return_days, mode="ORIGIN", fusion=False),
        score_path(input_days, return_days, mode="TEST", fusion=False),
        score_path(input_days, return_days, mode="TEST", fusion=True),
    )


def _transformer_scores_default(mode: str = "TEST") -> Tuple[Path, ...]:
    return (
        transformer_score_path(mode=mode, timeframe="MEDIUM"),
        transformer_score_path(mode=mode, timeframe="LONG"),
    )


def _transformer_scores_lp(mode: str = "TEST") -> Tuple[Path, ...]:
    return (
        transformer_score_path(mode=mode, timeframe="MEDIUM"),
        transformer_score_path(mode=mode, timeframe="LONG"),
        transformer_score_path(mode=mode, name="transformer_test_medium_lp_126"),
        transformer_score_path(mode=mode, name="transformer_test_medium_lp_252"),
    )


def _transformer_scores_mmcrash(mode: str = "TEST") -> Tuple[Path, ...]:
    name = f"transformer_{mode.lower()}_medium_mmcrash"
    return (transformer_score_path(name=name),)


def _transformer_scores_mmfusion(mode: str = "TEST") -> Tuple[Path, ...]:
    name = f"transformer_{mode.lower()}_medium_mmfusion"
    return (transformer_score_path(name=name),)


def _transformer_scores_mfd(mode: str = "TEST", timeframe: str = "MEDIUM") -> Tuple[Path, ...]:
    name = f"transformer_{mode.lower()}_{timeframe.lower()}_mfd"
    return (transformer_score_path(name=name),)


def _transformer_scores_mfd_bottom_pct(*, bottom_pct: int) -> Tuple[Path, ...]:
    name = f"transformer_medium_mfd_{int(bottom_pct)}"
    return (transformer_score_path(name=name),)


EXAMPLES: Dict[str, ExampleSpec] = {
    "cnn_single_test": ExampleSpec(
        name="cnn_single_test",
        scores=(score_path(20, 20, mode="TEST", fusion=False),),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
    ),
    "cnn_single_fusion": ExampleSpec(
        name="cnn_single_fusion",
        scores=(score_path(20, 20, mode="TEST", fusion=True),),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
    ),
    "cnn_batch": ExampleSpec(
        name="cnn_batch",
        scores=_cnn_scores(),
        group_selector=("q1", "q5"),
    ),
    "transformer_medium": ExampleSpec(
        name="transformer_medium",
        scores=(transformer_score_path(mode="TEST", timeframe="MEDIUM"),),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
    ),
    "transformer_long": ExampleSpec(
        name="transformer_long",
        scores=(transformer_score_path(mode="TEST", timeframe="LONG"),),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
    ),
    "transformer_lp": ExampleSpec(
        name="transformer_lp",
        scores=_transformer_scores_lp(),
        group_selector="q5",
    ),
    "full_comparison": ExampleSpec(
        name="full_comparison",
        scores=_cnn_scores() + _transformer_scores_default() + _transformer_scores_lp() + _transformer_scores_mmfusion(),
        group_selector="q5",
    ),
    "transformer_long_short": ExampleSpec(
        name="transformer_long_short",
        scores=_transformer_scores_default(),
        group_selector=("q1", "q5", "net"),
        overrides={
            "active_quantiles": (0, 4),
            "long_short_mode": "net",
            "short_quantiles": (0,),
            "label_prefix": "long_short",
            "dollar_neutral_net": True,
        },
        output_filename="backtest_long_short.png",
    ),
    "transformer_long_short_sector_neutral": ExampleSpec(
        name="transformer_long_short_sector_neutral",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=80),
        group_selector=("q1", "q5", "net"),
        sector_neutral=True,
        overrides={
            "active_quantiles": (0, 4),
            "long_short_mode": "net",
            "short_quantiles": (0,),
            "min_assets": 5,
            "label_prefix": "long_short_sn_mfd_80",
            "dollar_neutral_net": True,
        },
        output_filename="backtest_long_short_sn_mfd_80.png",
    ),
    "transformer_medium_multi": ExampleSpec(
        name="transformer_medium_multi",
        scores=(transformer_score_path(name="transformer_test_medium_multi"),),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
        overrides={
            "active_quantiles": (0, 1, 2, 3, 4),
            "label_prefix": "multi",
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": None,
        },
        output_filename="backtest_transformer_medium_multi.png",
    ),
    "transformer_medium_mmcrash": ExampleSpec(
        name="transformer_medium_mmcrash",
        scores=_transformer_scores_mmcrash(),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
        overrides={
            "active_quantiles": (0, 1, 2, 3, 4),
            "label_prefix": "mmcrash",
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": None,
            "min_assets": 10,
            "min_score": 0.0,
        },
        output_filename="backtest_transformer_medium_mmcrash.png",
    ),
    "transformer_medium_mmfusion": ExampleSpec(
        name="transformer_medium_mmfusion",
        scores=_transformer_scores_mmfusion(),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
        overrides={
            "active_quantiles": (0, 1, 2, 3, 4,),
            "label_prefix": "mmfusion",
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "min_assets": 40,
            "min_score": 0.6,
            "allow_partial_buckets": True,
        },
        output_filename="backtest_transformer_medium_mmfusion.png",
    ),
    "transformer_medium_mfd": ExampleSpec(
        name="transformer_medium_mfd",
        scores=_transformer_scores_mfd(mode="TEST", timeframe="MEDIUM"),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
        overrides={
            "active_quantiles": (0, 1, 2, 3, 4),
            "label_prefix": "mfd",
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
        },
        output_filename="backtest_transformer_medium_mfd.png",
    ),
    "transformer_medium_mfd_min_score": ExampleSpec(
        name="transformer_medium_mfd_min_score",
        scores=_transformer_scores_mfd(mode="TEST", timeframe="MEDIUM"),
        group_selector=("q1", "q2", "q3", "q4", "q5"),
        overrides={
            "active_quantiles": (0, 1, 2, 3, 4),
            "label_prefix": "mfd_min_score",
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "min_score": 0.55,
        },
        output_filename="backtest_transformer_medium_mfd_min_score.png",
    ),
       "transformer_medium_mfd_10": ExampleSpec(
        name="transformer_medium_mfd_10",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=10),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_10_q5.png",
    ),
    "transformer_medium_mfd_20": ExampleSpec(
        name="transformer_medium_mfd_20",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=20),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_20_q5.png",
    ),
    "transformer_medium_mfd_30": ExampleSpec(
        name="transformer_medium_mfd_30",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=30),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_30_q5.png",
    ),
    "transformer_medium_mfd_40": ExampleSpec(
        name="transformer_medium_mfd_40",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=40),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_40_q5.png",
    ),
    "transformer_medium_mfd_50": ExampleSpec(
        name="transformer_medium_mfd_50",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=50),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_50_q5.png",
    ),
    "transformer_medium_mfd_60": ExampleSpec(
        name="transformer_medium_mfd_60",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=60),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_60_q5.png",
    ),
    "transformer_medium_mfd_70": ExampleSpec(
        name="transformer_medium_mfd_70",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=70),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_70_q5.png",
    ),
    "transformer_medium_mfd_80": ExampleSpec(
        name="transformer_medium_mfd_80",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=80),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_80_q5.png",
    ),
    "transformer_medium_mfd_90": ExampleSpec(
        name="transformer_medium_mfd_90",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=90),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_90_q5.png",
    ),
    "transformer_medium_mfd_100": ExampleSpec(
        name="transformer_medium_mfd_100",
        scores=_transformer_scores_mfd_bottom_pct(bottom_pct=100),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
        },
        output_filename="backtest_transformer_medium_mfd_100_q5.png",
    ),
    "transformer_medium_mfd_q5_sweep": ExampleSpec(
        name="transformer_medium_mfd_q5_sweep",
        scores=tuple(
            transformer_score_path(name=f"transformer_medium_mfd_{p}")
            for p in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        ),
        group_selector="q5",
        overrides={
            "active_quantiles": (4,),
            "portfolio_weighting": "eq",
            "constituent_universe": MarketUniverse.KOSPI200,
            "benchmark_symbol": BenchmarkType.KOSPI200,
            "allow_partial_buckets": True,
            "min_assets": 5,
            "report_title": "transformer_medium_mfd_sweep",
        },
        output_filename="transformer_medium_mfd_sweep.png",
    ),
}


def main(selected_examples: Tuple[str, ...] | None = None) -> Dict[str, Backtester]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not selected_examples:
        raise ValueError("selected_examples must be provided and non-empty.")
    targets = selected_examples

    apply_trading_costs = False
    buy_cost_bps = 2.0
    sell_cost_bps = 2.0
    tax_bps = 15.0
    entry_lag = 0
    entry_price_mode = "close"
    benchmark = BenchmarkType.KOSPI200EQ
    portfolio_weighting = "eq"

    runner = ExampleRunner(
        base_opts=dict(
            rebalance_frequency="M",
            portfolio_weighting=portfolio_weighting,
            apply_trading_costs=apply_trading_costs,
            buy_cost_bps=buy_cost_bps,
            sell_cost_bps=sell_cost_bps,
            tax_bps=tax_bps,
            entry_lag=entry_lag,
            entry_price_mode=entry_price_mode,
            benchmark_symbol=BenchmarkType.KOSPI200,
            start_date="2020-01-31",
        )
    )

    results: Dict[str, Backtester] = {}
    for name in targets:
        try:
            results[name] = runner.run_named(name)
        except Exception as exc:
            logger.warning("%s run failed: %s", name, exc)

    global LAST_RUNNER, LAST_RESULTS
    LAST_RUNNER = runner
    LAST_RESULTS = dict(results)

    # NOTE: For validation examples
    # try:
    #     run_validation_example(
    #         rebalance_frequency="M",
    #         start_date="2012-01-01",
    #     )
    # except Exception as exc:
    #     logger.warning("Validation example failed: %s", exc)
    return results


if __name__ == "__main__":
    examples = ("transformer_long_short_sector_neutral",)
    main(examples)
