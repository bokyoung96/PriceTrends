# PriceTrends

Research framework for stock price trend prediction with CNN/Transformer models and a flexible backtesting engine.

---

## Overview
- **CNN pipeline** (`core/`, `prediction/`): OHLCV → chart images → CNN scoring.
- **Transformer pipeline** (`transformer/`): Raw time-series features + custom Transformer/VSN, multiple timeframes (SHORT/MEDIUM/LONG).
- **Backtest engine** (`backtest/`): Event-driven simulator with costs, benchmarks, grouping/quantiles, and reporting.

---

## Modules
| Module | Description |
| --- | --- |
| `core/` | Data loading/preprocessing and CNN model definitions. |
| `prediction/` | Image generation, CNN evaluation, and scoring. |
| `transformer/` | Transformer model, feature engineering, training scripts. |
| `backtest/` | Portfolio engine, grouping, reporting, and examples. |
| `daily/` | Daily operational scripts. |
| `utils/` | Path helpers and misc utilities. |

---

## Key Features
- **Transformer pipeline**: memmap windowing for large lookbacks; `config.json` separates `mode` (TEST/PRODUCTION) and `timeframe` (SHORT/MEDIUM/LONG).
- **Backtesting**: quantile grouping, entry lag, trading costs/tax, sector-neutral option, long/short (legs + net), benchmark support.
- **Reporting**: equity/drawdown, summary stats, monthly returns, and saved PNG/HTML artifacts.

Note: Long/short net is self-financing (leg PnL ÷ gross then compounded); it can differ from simple `(q1+q5)/2` averages due to compounding/period alignment. See `improvements.md` for details.

---

## Quick Start
Prerequisites: Python 3.8+, install requirements in a virtualenv/conda env.

1) Train Transformer (example)
```bash
python transformer/train.py
```

2) Run a backtest (entry point)
```bash
python backtest/main.py
```
Tweak `backtest/main.py` examples for scores, rebal freq, costs, long/short mode, etc.

---

## Project Structure
```
PriceTrends/
  backtest/        # Engine, grouping, reporting, examples
  core/            # CNN data/model
  prediction/      # CNN scoring
  transformer/     # Transformer pipeline & training
  daily/           # Ops scripts
  utils/           # Helpers
  DATA/            # Parquet market data
  scores/          # Model prediction scores
  results/         # Backtest outputs
```

---

## Documentation
- Pipeline details: `pipeline.md`
- Transformer docs: `transformer/README.md`
- Backtest usage: `backtest/README.md`
- Notes/known issues: `improvements.md`

---

## Contributing
Issues/PRs welcome—focus on clear configs, reproducible experiments, and keeping artifacts (scores, reports) organized per module.
