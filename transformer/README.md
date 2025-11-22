# Transformer Stock Price Prediction Model

This directory contains a Transformer-based deep learning model for stock price movement prediction using technical indicators and market data.

## 📁 Directory Structure

```
transformer/
├── README.md           # This file
├── README_KR.md        # Korean documentation
├── model.py           # Transformer architecture
├── features.py        # Technical indicator calculations
├── pipeline.py        # Data processing pipeline (with memmap)
├── params.py          # Configuration loader
├── config.json        # Configuration file
├── train.py          # Training script
├── evaluate.py       # Model evaluation
└── DATA/             # Cached preprocessed data (.pt files)
```

## 🏗️ Model Architecture

### Overview
The model uses a custom Transformer encoder with Variable Selection Network (VSN) for learning from multiple technical indicators.

### Components

1. **Variable Selection Network (VSN)**
   - Automatically learns importance weights for different technical indicators
   - Per-variable Gated Residual Networks (GRN) for feature-specific processing
   - Soft attention mechanism across features

2. **Transformer Encoder**
   - Multi-head self-attention to capture temporal dependencies
   - GELU activation for non-linearity
   - Layer normalization (pre-norm architecture)
   - Residual connections

3. **Output Head**
   - GRN for final feature refinement
   - Layer normalization
   - Linear classifier for binary prediction (up/down)

### Key Modules

#### `GRN` (Gated Residual Network)
- Multi-layer perceptron with gating mechanism
- ELU activation + Dropout + GLU
- Skip connections for residual learning

#### `GLU` (Gated Linear Unit)
- Learnable gating mechanism: `x ⊙ σ(gate)`
- Helps control information flow

#### `PosEncoding` (Positional Encoding)
- Learnable positional embeddings
- Allows model to understand temporal ordering

## 📊 Features (Technical Indicators)

The model supports the following technical indicators (configured via `features` parameter):

### Basic Price Features
- **`logreturn`**: Log returns (log price ratios)
- **`hlspread`**: High-Low spread normalized by close
- **`ocgap`**: Open-Close gap (overnight returns)
- **`volumez`**: Volume z-score (normalized volume)

### Trend Indicators
- **`ma`**: Moving Average deviation (default: 20-day)
- **`ema`**: Exponential Moving Average deviation (default: 20-day)
- **`magap`**: Price-to-MA ratio gap
- **`macd`**: MACD histogram normalized by price

### Momentum Indicators
- **`rsi`**: Relative Strength Index (default: 14-day, normalized to [-0.5, 0.5])

### Volatility Indicators
- **`rollingvol`**: Rolling standard deviation of returns (default: 20-day)
- **`bb`**: Bollinger Bands position (normalized to [-0.5, 0.5])

### Custom Parameters
You can customize indicator parameters:
```python
features = [
    "logreturn",
    ("ma", {"window": 50}),  # 50-day MA
    ("rsi", {"window": 21}), # 21-day RSI
    ("bb", {"window": 20, "num_std": 2.5})  # Custom Bollinger Bands
]
```

## ⚙️ Configuration

### New Structure (mode + timeframe)

Configuration is now separated into two independent dimensions:

```json
{
  "mode_configs": {
    "TEST": { ... },
    "PRODUCTION": { ... }
  },
  "timeframe_configs": {
    "SHORT": { ... },
    "MEDIUM": { ... },
    "LONG": { ... }
  }
}
```

### `mode_configs` (Environment Settings)

| Mode | Train Years | Test Years | batch_size | max_epoch | lr |
|------|-------------|------------|------------|-----------|-----|
| **TEST** | 2000-2011 | 2012-2024 | 64 | 10 | 1e-4 |
| **PRODUCTION** | 2000-2021 | 2022-2024 | 256 | 50 | 1e-5 |

### `timeframe_configs` (Prediction Horizon)

| Timeframe | lookback | stride | horizon | Purpose |
|-----------|----------|--------|---------|---------|
| **SHORT** | 20 days | 1 | 5 days | Short-term momentum |
| **MEDIUM** | 60 days | 1 | 20 days | Monthly rebalancing |
| **LONG** | 126 days | 1 | 60 days | Long-term trends |

### Key Parameters Explained

#### **lookback** (Observation Window)
- How many past days the model observes
- Example: `lookback=60` → model sees past 60 days of price/indicator data

#### **stride** (Window Sliding Interval)
- How many days to move forward when creating training samples
- `stride=1` → daily sliding (maximum data, recommended)
- `stride=20` → jump 20 days (less data, faster training)

#### **horizon** (Prediction Target)
- How many days ahead to predict
- Example: `horizon=20` → predict price change 20 days from now
- Label: 1 if `price[today + 20] > price[today]` else 0

#### **min_assets**
- Minimum number of valid stocks required per time window
- Filters out dates with too few tradable stocks
- Higher value = stricter filtering

#### **norm**: Normalization strategy
- `"asset"`: Normalize each stock independently
- `"cross"`: Normalize across stocks at each time point
- `"none"`: No normalization

## 🚀 Usage

### Training

```python
from transformer.params import TransformerParams
from transformer.train import Trainer
from transformer.pipeline import Config

params = TransformerParams()

# Combine mode and timeframe
tcfg = params.get_config(mode="TEST", timeframe="MEDIUM")

cfg = Config(
    lookback=tcfg.lookback,
    stride=tcfg.stride,
    horizon=tcfg.horizon,
    features=tuple(tcfg.features),
    min_assets=tcfg.min_assets,
    norm=tcfg.norm,
    train_years=tcfg.train_years,
    test_years=tcfg.test_years
)

trainer = Trainer(cfg, name=f"transformer_{tcfg.mode}")
trainer.train(
    epochs=tcfg.max_epoch,
    batch=tcfg.batch_size,
    lr=tcfg.lr,
    d_model=tcfg.d_model,
    nhead=tcfg.nhead,
    n_layers=tcfg.n_layers,
    d_ff=tcfg.d_ff,
    drop=tcfg.drop
)
```

### Direct Execution

```bash
# Edit train.py to set mode and timeframe
# Default: mode="TEST", timeframe="MEDIUM"
python transformer/train.py
```

### Configuration Combinations

```python
# Quick validation
params.get_config(mode="TEST", timeframe="SHORT")

# Main experiment
params.get_config(mode="TEST", timeframe="MEDIUM")

# Long-term patterns
params.get_config(mode="TEST", timeframe="LONG")

# Production deployment
params.get_config(mode="PRODUCTION", timeframe="MEDIUM")
```

## 📈 Data Pipeline

### Flow

1. **Load Raw Data** (`FrameLoader`)
   - Loads OHLCV data from parquet files in `DATA/` folder
   - Filters excluded tickers

2. **Feature Engineering** (`Featurizer`)
   - Computes technical indicators from raw OHLCV
   - Applies normalization if configured

3. **Window Creation** (`WindowMaker`) **[Memory Optimized]**
   - **Uses `numpy.memmap`** for memory-efficient processing
   - Slides window over time series with `lookback` length
   - Filters windows with insufficient assets (`min_assets`)
   - Creates labels based on future returns (`horizon`)
   - **Supports stride=1 without memory issues**

4. **Caching** (`Windows.save/load`)
   - Saves processed data to `transformer/DATA/win_lb{lookback}_hz{horizon}.pt`
   - Uses pickle protocol 4 for large datasets (>4GB)

5. **Dataset & DataLoaders** (`StockDataset`, `get_loaders`)
   - Wraps windows into PyTorch Dataset
   - **Year-based split** (train_years vs test_years)
   - Default: chronological split, not random

### Memory Efficiency

**Inspired by CNN's proven approach**, the pipeline uses `numpy.memmap`:

```python
# Old approach (memory issue):
data = []
for window in windows:
    data.append(window)  # Accumulates in RAM → 8GB+
final = np.concatenate(data)  # CRASH!

# New approach (memory safe):
mmap = open_memmap(temp_file, shape=(n_windows, lookback, n_features))
for i, window in enumerate(windows):
    mmap[i] = window  # Writes to disk directly → ~2MB RAM
final = mmap[:actual_count]
```

**Result:**
- ✅ `stride=1` works perfectly
- ✅ Maximum training data utilization
- ✅ Constant memory usage (~2-3MB)

### Data Shape

- **Input**: `(batch, lookback, n_features)`
  - `lookback` = 60 (configurable)
  - `n_features` = 11 (number of selected indicators)
  
- **Label**: `(batch,)` 
  - Binary classification: 0 (down), 1 (up)

## 🔧 Model Hyperparameters

Configured per timeframe in `config.json`:

```json
{
  "MEDIUM": {
    "model": {
      "d_model": 64,
      "nhead": 4,
      "n_layers": 3,
      "d_ff": 128,
      "drop": 0.1
    }
  }
}
```

### Tuning Guidelines

- **`d_model`**: Model dimension (64, 128, 256)
- **`nhead`**: Number of attention heads (must divide `d_model` evenly)
- **`n_layers`**: Transformer encoder depth (2-6)
- **`d_ff`**: Feedforward dimension (typically 2-4x `d_model`)
- **`drop`**: Dropout rate for regularization (0.1-0.3)

## 📊 Training Details

- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss
- **Device**: Automatically detects MPS (Apple Silicon) / CUDA / CPU
- **Checkpointing**: Saves best model based on validation loss
- **Data Split**: Year-based chronological split (no data leakage)

## 🎯 Performance Metrics

Logged during training:
- **Loss**: Cross-entropy loss
- **Accuracy**: Classification accuracy

Output saved to:
- **Models**: `models/transformer_{mode}/best.pth`
- **Scores**: `scores/price_trends_score_transformer_{mode}_lb{lookback}_hz{horizon}.parquet`

## 🐛 Troubleshooting

### RuntimeWarning: Mean of empty slice
- **Status**: ✅ Fixed
- **Solution**: Warnings suppressed in `features.py`

### OverflowError: >4GB pickle
- **Status**: ✅ Fixed
- **Solution**: Using `pickle_protocol=4`

### MemoryError: Cannot allocate
- **Status**: ✅ Fixed
- **Solution**: Implemented `numpy.memmap` (CNN approach)

### RuntimeError: No windows
- **Cause**: `min_assets` too strict or insufficient data
- **Solution**: Lower `min_assets` (e.g., 50 → 30)

## 📝 Notes

- The model predicts future returns at a fixed `horizon` (e.g., 20 days)
- Features are automatically normalized to prevent scale issues
- Missing data is handled via masking (NaN → 0 with validity mask)
- **Train/val split is chronological** (year-based), not random
- **stride=1 is recommended** for maximum data utilization

## 🔮 Recommended Workflow

### For 20-Day Rebalancing Strategy:

1. **Training**
   ```python
   mode="PRODUCTION"
   timeframe="MEDIUM"  # lookback=60, horizon=20, stride=1
   ```

2. **Prediction**
   ```python
   # Predict every 20 days
   dates = ["2024-01-15", "2024-02-05", "2024-02-26", ...]
   ```

3. **Ensemble (Optional)**
   ```python
   # Combine multiple timeframes
   short = predict(timeframe="SHORT")   # 0.2 weight
   medium = predict(timeframe="MEDIUM") # 0.5 weight
   long = predict(timeframe="LONG")     # 0.3 weight
   final = 0.2*short + 0.5*medium + 0.3*long
   ```

## 🚀 Future Enhancements

- [ ] Add regression mode for continuous return prediction
- [ ] Implement attention visualization
- [ ] Add learning rate scheduler
- [ ] Support multi-horizon prediction
- [ ] Cross-validation with multiple time splits
- [ ] Automate hyperparameter tuning


This directory contains a Transformer-based deep learning model for stock price movement prediction using technical indicators and market data.

## 📁 Directory Structure

```
transformer/
├── README.md           # This file
├── model.py           # Transformer architecture
├── features.py        # Technical indicator calculations
├── pipeline.py        # Data processing pipeline
├── train.py          # Training script
├── evaluate.py       # Model evaluation
└── DATA/             # Cached preprocessed data (.pt files)
```

## 🏗️ Model Architecture

### Overview
The model uses a custom Transformer encoder with Variable Selection Network (VSN) for learning from multiple technical indicators.

### Components

1. **Variable Selection Network (VSN)**
   - Automatically learns importance weights for different technical indicators
   - Per-variable Gated Residual Networks (GRN) for feature-specific processing
   - Soft attention mechanism across features

2. **Transformer Encoder**
   - Multi-head self-attention to capture temporal dependencies
   - GELU activation for non-linearity
   - Layer normalization (pre-norm architecture)
   - Residual connections

3. **Output Head**
   - GRN for final feature refinement
   - Layer normalization
   - Linear classifier for binary prediction (up/down)

### Key Modules

#### `GRN` (Gated Residual Network)
- Multi-layer perceptron with gating mechanism
- ELU activation + Dropout + GLU
- Skip connections for residual learning

#### `GLU` (Gated Linear Unit)
- Learnable gating mechanism: `x ⊙ σ(gate)`
- Helps control information flow

#### `PosEncoding` (Positional Encoding)
- Learnable positional embeddings
- Allows model to understand temporal ordering

## 📊 Features (Technical Indicators)

The model supports the following technical indicators (configured via `features` parameter):

### Basic Price Features
- **`logreturn`**: Log returns (log price ratios)
- **`hlspread`**: High-Low spread normalized by close
- **`ocgap`**: Open-Close gap (overnight returns)
- **`volumez`**: Volume z-score (normalized volume)

### Trend Indicators
- **`ma`**: Moving Average deviation (default: 20-day)
- **`ema`**: Exponential Moving Average deviation (default: 20-day)
- **`magap`**: Price-to-MA ratio gap
- **`macd`**: MACD histogram normalized by price

### Momentum Indicators
- **`rsi`**: Relative Strength Index (default: 14-day, normalized to [-0.5, 0.5])

### Volatility Indicators
- **`rollingvol`**: Rolling standard deviation of returns (default: 20-day)
- **`bb`**: Bollinger Bands position (normalized to [-0.5, 0.5])

### Custom Parameters
You can customize indicator parameters:
```python
features = [
    "logreturn",
    ("ma", {"window": 50}),  # 50-day MA
    ("rsi", {"window": 21}), # 21-day RSI
    ("bb", {"window": 20, "num_std": 2.5})  # Custom Bollinger Bands
]
```

## ⚙️ Configuration

### `Config` (in `pipeline.py`)

```python
@dataclass(frozen=True)
class Config:
    data_dir: Path           # Path to raw data (parquet files)
    out_path: Optional[Path] # Custom output path for cached data
    lookback: int = 60      # Number of historical days
    stride: int = 1         # Window sliding stride
    horizon: int = 20       # Prediction horizon (days ahead)
    min_assets: int = 10    # Minimum valid assets per window
    min_valid: float = 0.95 # Minimum data validity ratio
    norm: str = "asset"     # Normalization: "asset", "cross", "none"
    features: Tuple         # List of feature names
    vol_window: int = 20    # Volume normalization window
    zero_invalid: bool = False  # Zero out invalid values
    label_type: str = "classification"  # "classification" or "regression"
    threshold: float = 0.0  # Classification threshold
```

### Key Parameters Explained

- **`min_assets`**: Filters out time windows with too few valid stocks (prevents sparse data issues)
- **`lookback`**: How far back the model looks (60 days = ~3 months of trading data)
- **`horizon`**: How far ahead to predict (20 days = ~1 month)
- **`norm`**: 
  - `"asset"`: Normalize each stock independently
  - `"cross"`: Normalize across stocks at each time point
  - `"none"`: No normalization

## 🚀 Usage

### Training

```python
from transformer.train import Trainer
from transformer.pipeline import Config

cfg = Config(
    lookback=60,
    horizon=20,
    features=("logreturn", "hlspread", "ocgap", "volumez", "rsi", "macd"),
    min_assets=50
)

trainer = Trainer(cfg, name="transformer_v1")
trainer.train(epochs=10, batch=64, lr=1e-4)
```

### Direct Execution

```bash
# Train with default configuration
python transformer/train.py
```

## 📈 Data Pipeline

### Flow

1. **Load Raw Data** (`FrameLoader`)
   - Loads OHLCV data from parquet files in `DATA/` folder
   - Filters excluded tickers

2. **Feature Engineering** (`Featurizer`)
   - Computes technical indicators from raw OHLCV
   - Applies normalization if configured

3. **Window Creation** (`WindowMaker`)
   - Slides window over time series with `lookback` length
   - Filters windows with insufficient assets (`min_assets`)
   - Creates labels based on future returns (`horizon`)

4. **Caching** (`Windows.save/load`)
   - Saves processed data to `transformer/DATA/win_lb{lookback}_hz{horizon}.pt`
   - Uses pickle protocol 4 for large datasets (>4GB)

5. **Dataset & DataLoaders** (`StockDataset`, `get_loaders`)
   - Wraps windows into PyTorch Dataset
   - Splits by date (not random) for train/validation
   - Default split: 70% train, 30% validation

### Data Shape

- **Input**: `(batch, sequence_length, n_features)`
  - `sequence_length` = `lookback` (e.g., 60)
  - `n_features` = number of selected technical indicators
  
- **Label**: `(batch,)` 
  - Binary classification: 0 (down), 1 (up)

## 🔧 Model Hyperparameters

Configured in `Trainer.train()`:

```python
model = Transformer(
    n_feat=n_feat,        # Auto-detected from features
    d_model=64,           # Model dimension
    nhead=4,              # Number of attention heads
    n_layers=2,           # Transformer encoder layers
    d_ff=128,             # Feedforward dimension
    drop=0.1,             # Dropout rate
    n_class=2,            # Binary classification
    max_len=seq_len + 100 # Max sequence length
)
```

### Tuning Guidelines

- **`d_model`**: Increase for more capacity (try 128, 256)
- **`nhead`**: Must divide `d_model` evenly (try 4, 8)
- **`n_layers`**: More layers = more depth (try 2-6)
- **`d_ff`**: Typically 2-4x `d_model`
- **`drop`**: Regularization (0.1-0.3 for small datasets)

## 📊 Training Details

- **Optimizer**: Adam (default lr=1e-4)
- **Loss**: CrossEntropyLoss
- **Device**: Automatically detects MPS (Apple Silicon) / CUDA / CPU
- **Checkpointing**: Saves best model based on validation loss

## 🎯 Performance Metrics

Logged during training:
- **Loss**: Cross-entropy loss
- **Accuracy**: Classification accuracy

## 🐛 Troubleshooting

### RuntimeWarning: Mean of empty slice
- **Cause**: Some assets have insufficient early data
- **Solution**: Already suppressed via `warnings.catch_warnings()` in `features.py`

### OverflowError: >4GB pickle
- **Cause**: Dataset too large for default pickle protocol
- **Solution**: Already fixed with `pickle_protocol=4` in `Windows.save()`

### RuntimeError: No windows
- **Cause**: `min_assets` too strict or data too limited
- **Solution**: Lower `min_assets` (e.g., from 50 to 20)

## 📝 Notes

- The model predicts future returns at a fixed `horizon` (e.g., 20 days)
- Features are automatically normalized to prevent scale issues
- Missing data is handled via masking (NaN → 0 with validity mask)
- Train/val split is **chronological**, not random (prevents lookahead bias)

## 🔮 Future Enhancements

- [ ] Add regression mode for continuous return prediction
- [ ] Implement attention visualization
- [ ] Add learning rate scheduler
- [ ] Support multi-horizon prediction
- [ ] Cross-validation with multiple time splits
