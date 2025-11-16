from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from backtest.config import BacktestConfig
from backtest.runner import Backtester

logger = logging.getLogger(__name__)


def run_single_example() -> None:
    tester = Backtester()
    report = tester.run_single()
    logger.info("Single-score summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved single-score report to %s", output_path)


def run_batch_example() -> None:
    config = BacktestConfig(
        scores_path=(
            Path("scores/price_trends_score_test_i20_r20.parquet"),
            Path("scores/price_trends_score_test_i60_r60.parquet"),
        )
    )
    tester = Backtester(config)
    report = tester.run_batch(bucket=("q1", "q5"))
    logger.info("Batch summary:\n%s", report.summary_table())
    output_path = report.save()
    logger.info("Saved batch comparison report to %s", output_path)


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
    main()
