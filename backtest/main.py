from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from core.const import MarketUniverse
from backtest.config import BacktestConfig
from backtest.runner import Backtester

logger = logging.getLogger(__name__)
DEFAULT_UNIVERSE: MarketUniverse | None = MarketUniverse.KOSPI200
LATEST_BATCH_TESTER: Backtester | None = None


def _build_config(**overrides) -> BacktestConfig:
    base: dict[str, object] = {}
    if DEFAULT_UNIVERSE is not None:
        base["constituent_universe"] = DEFAULT_UNIVERSE
    base.update(overrides)
    return BacktestConfig(**base)


def run_single_example() -> None:
    tester = Backtester(_build_config())
    report = tester.run_single()
    logger.info("Single-score summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved single-score report to %s", output_path)


def run_batch_example() -> Backtester:
    config = _build_config(
        scores_path=(
            Path("scores/price_trends_score_test_i20_r20.parquet"),
            Path("scores/price_trends_score_origin_i20_r20.parquet"),
        )
    )
    tester = Backtester(config)
    report = tester.run_batch(bucket=("q1", "q5"))
    logger.info("Batch summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved batch comparison report to %s", output_path)
    global LATEST_BATCH_TESTER
    LATEST_BATCH_TESTER = tester
    logger.info("Stored batch tester on backtest.main.LATEST_BATCH_TESTER for downstream inspection.")
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        batch = run_batch_example()
    except Exception as exc:
        logger.warning("Batch example failed: %s", exc)
