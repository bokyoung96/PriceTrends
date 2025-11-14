from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from backtest.config import BacktestConfig
from backtest.costs import ExecutionCostModel
from backtest.data_sources import BacktestDataset
from backtest.engine import BacktestEngine
from backtest.portfolio import QuantilePortfolio
from backtest.quantiles import QuantileAssigner


def test_quantile_assignment_balanced() -> None:
    scores = pd.Series([0.1, 0.4, 0.9, 0.2, 0.7], index=list("abcde"))
    assigner = QuantileAssigner(quantiles=5, min_assets=5)

    buckets = assigner.assign(scores)
    assert buckets.has_assets()
    assert buckets.total_assets == 5
    assert set(buckets.tickers_for(4)) == {"c"}
    assert len(buckets.tickers_for(0)) == 1


def test_engine_tracks_capital_through_periods() -> None:
    dates = pd.to_datetime(["2021-01-04", "2021-02-01", "2021-03-01"])
    tickers = ["A", "B", "C"]
    scores = pd.DataFrame(
        [
            [0.1, 0.2, 0.9],
            [0.4, 0.7, 0.3],
            [0.6, 0.1, 0.2],
        ],
        index=dates,
        columns=tickers,
    )
    prices = pd.DataFrame(
        [
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 33.0],
            [12.0, 19.0, 36.0],
        ],
        index=dates,
        columns=tickers,
    )
    dataset = BacktestDataset(scores=scores, prices=prices)

    config = BacktestConfig(
        scores_path=Path("scores.parquet"),
        close_path=Path("close.parquet"),
        initial_capital=100.0,
        quantiles=2,
        rebalance_frequency="M",
        min_assets=2,
        active_quantiles=(1,),
    )
    assigner = QuantileAssigner(quantiles=2, min_assets=2)
    engine = BacktestEngine(config=config, dataset=dataset, assigner=assigner)

    report = engine.run()
    top_bucket = report.quantiles[1]

    assert top_bucket.equity_curve.iloc[-1] == pytest.approx(107.2673, rel=1e-4)
    assert top_bucket.period_returns.iloc[0] == pytest.approx(0.075, rel=1e-4)
    assert top_bucket.period_returns.iloc[1] == pytest.approx(-0.0021645, rel=1e-4)


def test_portfolio_respects_trading_costs() -> None:
    entry = pd.Series({"A": 10.0})
    exit_ = pd.Series({"A": 11.0})
    cost_model = ExecutionCostModel(enabled=True, buy_bps=2.0, sell_bps=2.0, tax_bps=15.0)

    portfolio = QuantilePortfolio(quantile_id=0, starting_capital=100.0, cost_model=cost_model)
    start = pd.Timestamp("2021-01-01")
    end = pd.Timestamp("2021-02-01")
    portfolio.mark_initial(start)
    portfolio.rebalance(
        enter_date=start,
        exit_date=end,
        tickers=("A",),
        entry_prices=entry,
        exit_prices=exit_,
    )

    round_trip = (1 - 0.0002) * 1.1 * (1 - 0.0017)
    assert portfolio.trades[-1].capital_out == pytest.approx(100.0 * round_trip, rel=1e-6)
    assert portfolio.trades[-1].period_return == pytest.approx(round_trip - 1, rel=1e-6)


def test_portfolio_holds_on_missing_exit_prices() -> None:
    entry = pd.Series({"A": 10.0, "B": 20.0})
    exit_ = pd.Series({"A": 11.0, "B": pd.NA})

    portfolio = QuantilePortfolio(quantile_id=0, starting_capital=100.0)
    start = pd.Timestamp("2022-01-03")
    end = pd.Timestamp("2022-02-03")
    portfolio.mark_initial(start)
    portfolio.rebalance(
        enter_date=start,
        exit_date=end,
        tickers=("A", "B"),
        entry_prices=entry,
        exit_prices=exit_,
    )

    expected_relative = ((11.0 / 10.0) + 1.0) / 2
    assert portfolio.trades[-1].capital_out == pytest.approx(100.0 * expected_relative, rel=1e-6)
    assert "Held positions for halted tickers" in (portfolio.trades[-1].note or "")


def test_exit_costs_scaled_by_tradable_fraction() -> None:
    entry = pd.Series({"A": 10.0, "B": 20.0})
    exit_ = pd.Series({"A": 12.0, "B": pd.NA})
    cost_model = ExecutionCostModel(enabled=True, sell_bps=100.0)  # 1% sell cost

    portfolio = QuantilePortfolio(quantile_id=0, starting_capital=100.0, cost_model=cost_model)
    start = pd.Timestamp("2022-03-01")
    end = pd.Timestamp("2022-04-01")
    portfolio.mark_initial(start)
    portfolio.rebalance(
        enter_date=start,
        exit_date=end,
        tickers=("A", "B"),
        entry_prices=entry,
        exit_prices=exit_,
    )

    gross_relative = ((12.0 / 10.0) + 1.0) / 2  # B is flat because exit missing
    tradable_fraction = 0.5  # only A exits
    expected = 100.0 * gross_relative * (1 - 0.01 * tradable_fraction)
    assert portfolio.trades[-1].capital_out == pytest.approx(expected, rel=1e-6)
