from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.spec import MarketUniverse
from backtest.config import BacktestConfig, BenchmarkType, score_path
from backtest.runner import Backtester

logger = logging.getLogger(__name__)
DEFAULT_UNIVERSE: MarketUniverse | None = MarketUniverse.KOSPI200


class ExampleRegistry:
    def __init__(self) -> None:
        self.latest_single: Backtester | None = None
        self.latest_batch: Backtester | None = None


REGISTRY = ExampleRegistry()


def _build_config(portfolio_weighting: str = "mc", **overrides) -> BacktestConfig:
    base: dict[str, object] = {}
    if DEFAULT_UNIVERSE is not None:
        base["constituent_universe"] = DEFAULT_UNIVERSE
    base["portfolio_weighting"] = portfolio_weighting
    base.update(overrides)
    return BacktestConfig(**base)


def run_single_example(
    input_days: int = 20,
    return_days: int = 20,
    mode: str = "TEST",
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=score_path(
            input_days,
            return_days,
            mode=mode,
            fusion=False,
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(cfg)
    report = tester.run(group_selector=("q1", "q2", "q3", "q4", "q5"))
    logger.info("Single summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved single report to %s", output_path)
    REGISTRY.latest_single = tester
    return tester


def run_single_fusion_example(
    input_days: int = 20,
    return_days: int = 20,
    mode: str = "TEST",
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=score_path(
            input_days,
            return_days,
            mode=mode,
            fusion=True,
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(cfg)
    report = tester.run(group_selector=("q1", "q2", "q3", "q4", "q5"))
    logger.info("Fusion single summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved fusion single report to %s", output_path)
    REGISTRY.latest_single = tester
    return tester


def run_batch_example(
    input_days: int = 20,
    return_days: int = 20,
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    config = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=(
            score_path(input_days, return_days, mode="TEST", fusion=False),
            score_path(input_days, return_days, mode="ORIGIN", fusion=False),
            score_path(input_days, return_days, mode="TEST", fusion=True),
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(config)
    report = tester.run(group_selector=("q1", "q5"))
    logger.info("Batch summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved batch comparison report to %s", output_path)
    REGISTRY.latest_batch = tester
    return tester


def run_ensemble_example(
    mode: str = "TEST",
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=score_path(
            # NOTE: input / return as dummy variable 0
            input_days=0,
            return_days=0,
            mode=mode,
            fusion=False,
            ensemble=True,
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(cfg)
    report = tester.run(group_selector=("q1", "q2", "q3", "q4", "q5"))
    logger.info("Ensemble (%s) summary:\n%s", mode, report.summary_table())
    output_path = report.save()
    logger.info("Saved ensemble (%s) report to %s", mode, output_path)
    return tester


def run_ensemble_fusion_example(
    mode: str = "TEST",
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=score_path(
            input_days=0,
            return_days=0,
            mode=mode,
            fusion=True,
            ensemble=True,
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(cfg)
    report = tester.run(group_selector=("q1", "q2", "q3", "q4", "q5"))
    logger.info("Ensemble Fusion (%s) summary:\n%s", mode, report.summary_table())
    output_path = report.save()
    logger.info("Saved ensemble fusion (%s) report to %s", mode, output_path)
    return tester


def run_ensemble_batch_example(
    rebalance_frequency: str = "M",
    portfolio_weighting: str = "mc",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    benchmark: BenchmarkType | str = BenchmarkType.KOSPI200,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    config = _build_config(
        portfolio_weighting=portfolio_weighting,
        scores_path=(
            score_path(0, 0, mode="TEST", fusion=False, ensemble=True),
            score_path(0, 0, mode="ORIGIN", fusion=False, ensemble=True),
            score_path(0, 0, mode="TEST", fusion=True, ensemble=True),
        ),
        rebalance_frequency=rebalance_frequency,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark_symbol=benchmark,
        start_date=start_date,
        end_date=end_date,
    )
    tester = Backtester(config)
    report = tester.run(group_selector=("q1", "q5"))
    logger.info("Ensemble Batch summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved ensemble batch comparison report to %s", output_path)
    REGISTRY.latest_batch = tester
    return tester


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run_single_example()
    except Exception as exc:
        logger.warning("Single example failed: %s", exc)
    try:
        run_ensemble_example()
    except Exception as exc:
        logger.warning("Ensemble example failed: %s", exc)
    try:
        run_ensemble_fusion_example()
    except Exception as exc:
        logger.warning("Ensemble fusion example failed: %s", exc)
    try:
        run_ensemble_batch_example()
    except Exception as exc:
        logger.warning("Ensemble batch example failed: %s", exc)


if __name__ == "__main__":
    apply_trading_costs = False
    buy_cost_bps = 2.0
    sell_cost_bps = 2.0
    tax_bps = 15.0
    entry_lag = 0
    entry_price_mode = "close"
    benchmark = BenchmarkType.KOSPI200EQ
    portfolio_weighting = "eq"

    # tester = run_single_example(
    #     input_days=20,
    #     return_days=20,
    #     mode='TEST',
    #     rebalance_frequency="M",
    #     portfolio_weighting=portfolio_weighting,
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    #     benchmark=benchmark,
    # )

    # tester = run_single_example(
    #     input_days=20,
    #     return_days=20,
    #     mode='ORIGIN',
    #     rebalance_frequency="M",
    #     portfolio_weighting=portfolio_weighting,
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    #     benchmark=benchmark,
    # )
    
    # tester = run_single_fusion_example(
    #     input_days=20,
    #     return_days=20,
    #     rebalance_frequency="M",
    #     portfolio_weighting=portfolio_weighting,
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    #     benchmark=benchmark,
    # )

    tester = run_ensemble_fusion_example(
        rebalance_frequency="M",
        portfolio_weighting=portfolio_weighting,
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        benchmark=benchmark,
    )

    # # Batch comparison (CNN test / origin / fusion)
    # tester = run_batch_example(
    #     input_days=20,
    #     return_days=20,
    #     rebalance_frequency="M",
    #     portfolio_weighting=portfolio_weighting,
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    #     benchmark=benchmark,
    # )

    # # Ensemble Batch comparison
    # tester = run_ensemble_batch_example(
    #     rebalance_frequency="M",
    #     portfolio_weighting=portfolio_weighting,
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    #     benchmark=benchmark,
    # )
