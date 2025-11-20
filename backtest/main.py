from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.spec import MarketUniverse
from backtest.config import BacktestConfig, score_path
from backtest.runner import Backtester

logger = logging.getLogger(__name__)
DEFAULT_UNIVERSE: MarketUniverse | None = MarketUniverse.KOSPI200


class ExampleRegistry:
    def __init__(self) -> None:
        self.latest_single: Backtester | None = None
        self.latest_batch: Backtester | None = None


REGISTRY = ExampleRegistry()


def _build_config(**overrides) -> BacktestConfig:
    base: dict[str, object] = {}
    if DEFAULT_UNIVERSE is not None:
        base["constituent_universe"] = DEFAULT_UNIVERSE
    base["portfolio_weighting"] = "eq"
    base.update(overrides)
    return BacktestConfig(**base)


def run_single_example(
    input_days: int = 20,
    return_days: int = 20,
    mode: str = "TEST",
    rebalance_frequency: str = "M",
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
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
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    cfg = _build_config(
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
    apply_trading_costs: bool = False,
    buy_cost_bps: float = 0.0,
    sell_cost_bps: float = 0.0,
    tax_bps: float = 0.0,
    entry_lag: int = 0,
    entry_price_mode: str = "close",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    config = _build_config(
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        run_single_example()
    except Exception as exc:
        logger.warning("Single example failed: %s", exc)
    try:
        run_batch_example()
    except Exception as exc:
        logger.warning("Batch example failed: %s", exc)


if __name__ == "__main__":
    apply_trading_costs = True
    buy_cost_bps = 2.0
    sell_cost_bps = 2.0
    tax_bps = 15.0
    entry_lag = 0
    entry_price_mode = "close"

    # tester = run_single_example(
    #     input_days=20,
    #     return_days=20,
    #     mode='TEST',
    #     rebalance_frequency="M",
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    # )

    tester = run_single_example(
        input_days=20,
        return_days=20,
        mode='ORIGIN',
        rebalance_frequency="M",
        apply_trading_costs=apply_trading_costs,
        buy_cost_bps=buy_cost_bps,
        sell_cost_bps=sell_cost_bps,
        tax_bps=tax_bps,
        entry_lag=entry_lag,
        entry_price_mode=entry_price_mode,
        start_date="2015-01-31",
        end_date="2024-12-31",
    )
    
    # tester = run_single_fusion_example(
    #     input_days=20,
    #     return_days=20,
    #     rebalance_frequency="M",
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    # )

    # # Batch comparison (CNN test / origin / fusion)
    # tester = run_batch_example(
    #     input_days=20,
    #     return_days=20,
    #     rebalance_frequency="M",
    #     apply_trading_costs=apply_trading_costs,
    #     buy_cost_bps=buy_cost_bps,
    #     sell_cost_bps=sell_cost_bps,
    #     tax_bps=tax_bps,
    #     entry_lag=entry_lag,
    #     entry_price_mode=entry_price_mode,
    # )
