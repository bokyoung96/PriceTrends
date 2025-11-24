# PriceTrends

## 📈 Overview

PriceTrends is a research framework for **stock‑price trend prediction**. It integrates deep learning models (CNN, Transformer) with a robust backtesting engine to evaluate trading strategies based on predicted trends.

The framework consists of three main pillars:
1.  **CNN Pipeline** (`core/` & `prediction/`) – Converts OHLCV data into chart images and trains a Convolutional Neural Network.
2.  **Transformer Pipeline** (`transformer/`) – Processes raw time‑series data using a custom Transformer with Variable Selection Network.
3.  **Backtest Engine** (`backtest/`) – A flexible, event-driven backtester for validating strategies, supporting various weighting schemes, transaction costs, and benchmarking.

---

## 🛠️ Modules

| Module | Description |
| :--- | :--- |
| `core/` | Data loading, preprocessing, and CNN model definitions. |
| `prediction/` | Image generation for CNN, model evaluation, and scoring. |
| `transformer/` | End‑to‑end Transformer model, feature engineering, and training scripts. |
| `backtest/` | **[NEW]** Comprehensive backtesting engine (Portfolio, Engine, Reporting). |
| `daily/` | Scripts for daily operational tasks and orchestration. |
| `utils/` | Helper utilities for path management and visualization. |

---

## 🚀 Key Features

### 1. Transformer Pipeline
-   **Memory‑Efficient**: Uses `numpy.memmap` for window creation, enabling `stride=1` (daily rolling windows) on large datasets without RAM issues.
-   **Flexible Configuration**: `config.json` separates **mode** (`TEST`/`PRODUCTION`) from **timeframe** (`SHORT`/`MEDIUM`/`LONG`), allowing mix-and-match experiments.
-   **Progress Tracking**: Integrated `tqdm` for real-time feedback on data loading and training.

### 2. Backtest Engine
-   **Event-Driven**: Simulates daily rebalancing with realistic constraints (entry lag, transaction costs, taxes).
-   **Multi-Strategy Support**: Compare multiple strategies (e.g., CNN vs. Transformer vs. Ensemble) in a single run.
-   **Rich Reporting**: Generates detailed performance reports including:
    -   Cumulative Returns & Equity Curves
    -   Drawdown Analysis
    -   Monthly Return Heatmaps
    -   Win Rate & Sharpe Ratio
-   **Validation**: Includes logic to validate backtest assumptions against benchmarks (e.g., KOSPI 200).

---

## 📦 Quick Start

### Prerequisites
-   Python 3.8+
-   Dependencies listed in `requirements.txt`

### 1. Data Preparation
Ensure your OHLCV data (Parquet format) is located in the `DATA/` directory.

### 2. Training a Model (Transformer)
```bash
# Train a Transformer model with TEST mode and MEDIUM timeframe
python transformer/train.py
```

### 3. Running a Backtest
The `backtest/main.py` script serves as the entry point for running backtests.

```bash
# Run a comprehensive comparison of multiple models
python backtest/main.py
```

You can customize the backtest in `backtest/main.py`:
```python
tester = run_comprehensive_comparison_example(
    input_days=20,
    return_days=20,
    rebalance_frequency="M",  # Monthly rebalancing
    start_date="2012-01-01",
    # ...
)
```

---

## � Project Structure

```
PriceTrends/
├── backtest/            # Backtesting engine & reporting
│   ├── engine.py        # Core simulation logic
│   ├── portfolio.py     # Portfolio state management
│   ├── report.py        # Performance analysis & visualization
│   └── main.py          # Backtest entry point
├── core/                # Core data & CNN modules
├── prediction/          # CNN prediction & scoring
├── transformer/         # Transformer model & pipeline
│   ├── model.py         # Network architecture
│   ├── train.py         # Training script
│   └── params.py        # Configuration management
├── daily/               # Daily operation scripts
├── utils/               # Utility functions
├── DATA/                # Market data (Parquet)
├── scores/              # Model prediction scores
└── results/             # Backtest reports & artifacts
```

---

## 📝 Documentation

-   **Pipeline Details**: See `pipeline.md` for a deep dive into the data processing and training workflows.
-   **Transformer Docs**: Check `transformer/README.md` for specific details on the Transformer implementation.

---

## 🎉 Contributing

Feel free to open issues or submit pull requests to improve the framework. Happy trading!
