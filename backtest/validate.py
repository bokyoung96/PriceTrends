from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from backtest.config import BacktestConfig, BenchmarkType, PortfolioWeights
from backtest.runner import Backtester
from utils.root import DATA_ROOT

logger = logging.getLogger(__name__)


def run_validation_example(
    rebalance_frequency: str = "M",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Backtester:
    # NOTE: Use floating mktcap for weights
    mktcap_path = DATA_ROOT / "MKTCAP.parquet"

    cfg = BacktestConfig(
        scores_path=mktcap_path,
        portfolio_weighting=PortfolioWeights.MARKET_CAP,
        rebalance_frequency=rebalance_frequency,
        quantiles=1,
        active_quantiles=[0],
        benchmark_symbol=BenchmarkType.KOSPI200,
        start_date=start_date,
        end_date=end_date,
    )
    
    tester = Backtester(cfg)
    report = tester.run(group_selector="q1")
    
    logger.info("Validation Summary:\n%s", report.summary_table())
    
    if report.bench_equity is not None:
        strategy_returns = report.groups['q1'].equity_curve.pct_change().dropna()
        bench_returns = report.bench_equity.pct_change().dropna()
        
        common_index = strategy_returns.index.intersection(bench_returns.index)
        if not common_index.empty:
            corr = strategy_returns.loc[common_index].corr(bench_returns.loc[common_index])
            logger.info(f"Correlation with Benchmark: {corr:.4f}")
            
            if corr > 0.99:
                logger.info("SUCCESS: High correlation with benchmark. Backtest logic validated.")
            else:
                logger.warning("WARNING: Correlation with benchmark is lower than expected.")
        else:
            logger.warning("No overlapping dates between strategy and benchmark for correlation calculation.")
    else:
        logger.warning("WARNING: Benchmark equity is not available in the report. Check benchmark data or alignment.")
            
    output_path = report.save()
    logger.info("Saved validation report to %s", output_path)
    return tester
