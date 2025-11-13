## PriceTrends

CNN-based price-trend prediction pipeline for the KOSPI200. OHLCV data is turned into chart images and fed to convolutional models that output up/down probabilities. The repo contains everything from parquet conversion to training, evaluation, scoring, and a daily workflow.

---

## Overview

1. **Data → Parquet** (`core/loader.py`)
2. **Parquet → Chart Images** (`prediction/image.py`)
3. **Images → CNN Training** (`core/training.py`)
4. **Evaluation & Results** (`prediction/evaluate.py`, `prediction/score.py`)
5. **Daily Orchestration** (`daily/`)

> Transformer experiments under `transformer/` are *work in progress*; APIs can change without notice.

---

## Modules

### Core (`core/`)
- `loader.py`: XLSX → parquet converters and cached parquet loader (`DataConverter`, `DataLoader`)
- `params.py`: loads `config.json` and exposes `CNNConfig`
- `training.py`: CNN definition, Dataset, Trainer (saves checkpoints at `models/korea_cnn_{ws}d{pw}p_{mode}/checkpoint*.pth.tar`)

### Prediction (`prediction/`)
- `image.py`: batch chart/image generation (5/20/60-day by default)
- `evaluate.py`: loads checkpoints, evaluates ensemble members, writes `results/test_results_<mode>_iX_rY.parquet`
- `score.py`: post-processes parquet results and saves probability tables under `scores/`
- `score_w.py`: precision-weighted scoring (legacy workflow)
- `daily/`: realtime/daily scripts (uses same checkpoints)

### Utils (`utils/`)
- `read.py`: quick chart viewer for generated images
- `root.py`: project-wide path constants (`DATA_ROOT`, `RESULTS_ROOT`, etc.)

---

## Model Summary

- **Input**: grayscale chart image (`1 × H × W`)
- **Stack**: 3~4 Conv-BN-LeakyReLU blocks + MaxPool
- **Classifier**: Fully-connected head → binary logits
- **Windows**: 5, 20, 60 lookbacks (more possible via config)
- **Label**: based on forward horizon (`pw`) return (`estimation_start ~ estimation_end`)

---

## Quick Start

### 0. Install
```bash
pip install -r requirements.txt
```

### 1. Convert data
```python
from core.loader import DataConverter
DataConverter("DATA/DATA.xlsx", "DATA/").data_convert()
```

### 2. Generate images
```python
from prediction.image import run_batch
run_batch(frequencies=[5, 20, 60])
```

### 3. Train
```python
from core.params import CNNParams
from core.training import Trainer

params = CNNParams()
config = params.get_config("TEST", 20)
trainer = Trainer(ws=20, pw=config.pw, config=config)
dataloaders = trainer.get_dataloaders(train_years=config.train_years)
trainer.train_empirical_ensem_model(dataloaders)
```

### 4. Evaluate
```python
from prediction.evaluate import Evaluate
res = Evaluate(mode="TEST", pairs=[(20, 20)]).run_all()
df_20 = res[(20, 20)]
```
This writes `results/price_trends_test_i20_r20.parquet` and returns the same DataFrame.

### 5. Score
```python
from prediction.score import main as score_main
# Save probability tables under scores/
prob_maps = score_main(mode="test", pairs=[(20, 20)], include_average=False)
```

---

## Configuration (`config.json`)

```json
{
  "mode_configs": {
    "TEST": {
      "mode": "test",
      "train_years": [...],
      "ensem_size": 1,
      "batch_size": 64,
      "max_epoch": 10,
      "lr": 1e-4,
      "drop_prob": 0.3,
      "conv_channels": [32, 64, 128]
    },
    "PRODUCTION": {
      "...": "..."
    }
  },
  "window_configs": {
    "5": {
      "pw": 5,
      "filter_sizes": {
        "TEST": [[3, 2], [3, 2], [3, 2]],
        "PRODUCTION": [[3, 2], [3, 2], [3, 2], [3, 2]]
      }
    }
  },
  "evaluation_windows": {
    "TEST": [5, 20, 60],
    "PRODUCTION": [5, 20, 60]
  },
  "test_years": [2012, ..., 2024]
}
```

- `evaluation_windows` controls default `(ws, pw)` combos for `Evaluate.run_all`.
- `test_years` filters `start_date` of chart windows; `ending_date` can spill into the next calendar year if the lookback bridges it.

---

## Output Files

| File | Description |
|------|-------------|
| `results/price_trends_<mode>_iX_rY.parquet` | Raw evaluation probabilities per stock/date |
| `scores/price_trends_score_<mode>_iX_rY.parquet` | Pivoted prob_up tables |
| `scores/price_trends_score_<mode>_ensemble.parquet` | Ensemble average (if computed) |
| `models/korea_cnn_{ws}d{pw}p_{mode}/checkpoint*.pth.tar` | Model weights |

Each parquet includes `StockID`, `ending_date`, `prob_up`, `prob_down`, `prediction`, `label`.

---

## Training Notes

- Early stopping triggers after 3 validation epochs without improvement (`max_epoch` obeyed now that the stray `break` was fixed).
- Checkpoints saved per ensemble member. TEST mode defaults to `ensem_size=1`.
- If you add more windows, extend `config.json` and regenerate images.

---

## Evaluation Tips

- `Evaluate.run_single(ws, pw)` runs one combo without editing configs.
- `Evaluate(mode="PRODUCTION", pairs=[(20, 20), (60, 60)])` loops whatever tuples you supply.
- `score.py` and `score_w.py` now read from `price_trends_*` files; ensure evaluate step is run first.

---

## Daily Workflow

`daily/main_r.py` orchestrates:
1. Load latest data
2. Generate daily charts (`daily/image_r.py`)
3. Run prediction (`daily/evaluate_r.py`)
4. Save under `daily/results_d/`

Run:
```python
from daily.main_r import main
main(end_date="2025-08-25", timeframes=[5, 20, 60])
```

---

## Project Tree
```
PriceTrends/
├── core/
├── prediction/
│   ├── image.py
│   ├── evaluate.py
│   └── score.py
├── daily/
├── utils/
├── DATA/
├── Images/
├── models/
├── results/
├── scores/
└── transformer/   # WIP
```

---

## Disclaimer

For research/education only. Financial predictions carry risk. Use at your own discretion.
