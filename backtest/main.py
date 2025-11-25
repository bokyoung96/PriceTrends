from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig, BenchmarkType, score_path, transformer_score_path
from backtest.runner import Backtester
from backtest.validate import run_validation_example
from core.spec import MarketUniverse

logger = logging.getLogger(__name__)
DEFAULT_UNIVERSE: MarketUniverse | None = MarketUniverse.KOSPI200


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


class ExampleRunner:
    def __init__(self, base_opts: Dict[str, object] | None = None) -> None:
        self.base_opts = base_opts or {}
        self.latest: Backtester | None = None

    def run_spec(self, spec: ExampleSpec) -> Backtester:
        cfg_kwargs = dict(self.base_opts)
        cfg_kwargs.update(spec.overrides)
        cfg_kwargs["scores_path"] = spec.scores
        cfg = _build_config(**cfg_kwargs)
        tester = Backtester(cfg)
        report = tester.run(group_selector=spec.group_selector)
        if spec.log_summary:
            logger.info("%s summary:\n%s", spec.name, report.summary_table())
        if spec.save_report:
            output_path = report.save()
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
        # transformer_score_path(mode=mode, name="transformer_test_medium_lp_252"),
    )


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
        scores=_cnn_scores() + _transformer_scores_default() + _transformer_scores_lp(),
        group_selector="q5",
    ),
}


def main(selected_examples: Tuple[str, ...] | None = None) -> None:
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
            benchmark_symbol=benchmark,
            start_date="2012-01-31",
        )
    )

    for name in targets:
        try:
            runner.run_named(name)
        except Exception as exc:
            logger.warning("%s run failed: %s", name, exc)

    # NOTE: For validation examples
    # try:
    #     run_validation_example(
    #         rebalance_frequency="M",
    #         start_date="2012-01-01",
    #     )
    # except Exception as exc:
    #     logger.warning("Validation example failed: %s", exc)


if __name__ == "__main__":
    examples = ("transformer_lp",)
    main(examples)
