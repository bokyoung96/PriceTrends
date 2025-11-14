# PriceTrends Score Backtester

This module turns any PriceTrends score matrix (dates × tickers × probabilities) into a configurable quintile backtest that deploys capital via `price * quantity` using `DATA/close.parquet`.

## Defaults & knobs

- Loads scores from `scores/price_trends_score_test_i20_r20.parquet` and close prices from `DATA/close.parquet`.
- Splits each rebalance date into 5 quantiles (0 = lowest scores, 4 = highest).
- Allocates 100,000,000 KRW per active quantile by default, spreading capital equally across the tickers inside the bucket.
- Rebalances monthly (`freq='M'`) and enforces a 30-asset minimum before forming buckets (toggle via `allow_partial_buckets`).
- Persists equity/return/summary CSVs under top-level `bt/` (sibling of `results/`) unless you override `output_dir`.
- Trading costs are disabled by default; toggle `apply_trading_costs=True` and set `buy_cost_bps`, `sell_cost_bps`, `tax_bps` (e.g., 2/2/15 bps for slippage + tax) to bake them into the portfolio sizing.
  When costs are on, rebalances only deploy the post-cost capital and subtract exit fees/taxes from realised proceeds.
- Missing exit prices are treated as temporary trading halts: those tickers stay marked at their entry price (0% return) and are excluded from exit costs.

Everything above is adjustable by instantiating `BacktestConfig` or passing overrides to `run_backtest`.

## Quick start

```python
from backtest.runner import run_backtest

# All arguments are optional; pass only what you want to override.
report = run_backtest(
    initial_capital=150_000_000,
    quantiles=5,
    rebalance_frequency="M",
    active_quantiles=(4,),  # focus on top quintile
    apply_trading_costs=True,
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    tax_bps=15.0,
)

print(report.render_summary())
report.save()
```

`run_backtest` wires together dataset loading, quantile assignment, and the engine in a single call. Pass paths (e.g., `scores_path="scores/another_run.parquet"`) or any `BacktestConfig` field as keyword overrides. Results are identical to calling the lower-level classes manually.

## API usage

```python
from backtest.config import BacktestConfig
from backtest.data_sources import BacktestDatasetBuilder
from backtest.engine import BacktestEngine
from backtest.quantiles import QuantileAssigner

config = BacktestConfig(initial_capital=50_000_000, rebalance_frequency="MS")
dataset = BacktestDatasetBuilder(config.scores_path, config.close_path).build()
assigner = QuantileAssigner(config.quantiles, config.min_assets, config.allow_partial_buckets)

report = BacktestEngine(config, dataset, assigner).run()
print(report.summary_table())
report.save()
```

## Tests

Synthetic regression tests live under `backtest/tests/`. Run them with:

```bash
cd PriceTrends
pytest backtest/tests -q
```
