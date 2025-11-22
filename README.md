# PriceTrends

## 📈 Overview

PriceTrends is a research framework for **stock‑price trend prediction**. It contains two pipelines:

1. **CNN pipeline** (under `core/` & `prediction/`) – converts OHLCV data into chart images and trains a convolutional model.
2. **Transformer pipeline** (under `transformer/`) – works directly on the raw time‑series using a custom Transformer with Variable Selection Network.

Both pipelines share the same data‑loader utilities and can be mixed‑and‑matched.

---

## 🛠️ Modules

| Module | Description |
|--------|-------------|
| `core/` | CSV/XLSX → Parquet conversion, generic data loader, CNN utilities |
| `prediction/` | Image generation, CNN evaluation & scoring |
| `transformer/` | End‑to‑end Transformer model, feature engineering, mem‑mapped window creation |
| `daily/` | Scripts for daily‑run orchestration |
| `utils/` | Helper utilities (path constants, quick image viewer) |

---

## 🚀 Transformer Pipeline Highlights

- **Memory‑efficient window creation** – `pipeline.WindowMaker.make` now uses **`numpy.memmap`** (inspired by the CNN implementation). This allows `stride=1` (daily rolling windows) without blowing up RAM.
- **Config redesign** – `config.json` separates **mode** (`TEST` / `PRODUCTION`) from **timeframe** (`SHORT`, `MEDIUM`, `LONG`). You can now combine them freely, e.g. `params.get_config(mode="TEST", timeframe="MEDIUM")`.
- **Progress bars** – Both data loading and training loops are wrapped with **`tqdm`**, giving you live feedback on window generation, epoch progress, and batch processing.
- **Cross‑platform** – Works on macOS (Apple MPS), CUDA, and CPU. No platform‑specific code.

---

## 📦 Quick Start (Transformer)

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Prepare data (parquet files under DATA/)
#    (use the existing core.loader utilities)

# 3️⃣ Train a model (example: TEST + MEDIUM configuration)
python transformer/train.py
```

The script will:
1. **Load / generate windows** – shows a tqdm bar like `Creating windows: 100%|████| 6290/6290`.
2. **Build DataLoaders** – also wrapped with tqdm (you’ll see `Loading batches…`).
3. **Train** – each epoch displays `Ep 1/10 - train` and `Ep 1/10 - validate` progress bars.

---

## 📊 Adding tqdm to Data Loading (optional)

If you want a progress bar while the `DataLoader` iterates over batches, the `Trainer.train` method already uses:

```python
pbar = tqdm(loader, desc=f"Ep {ep+1}/{epochs} - {phase}")
```

You can also wrap the **window creation** step manually (already done) or any custom preprocessing step with `tqdm`.

---

## 🛡️ Known Issues & Fixes

- **Label dtype error** – `StockDataset.__getitem__` now casts the label to `int` before creating a `torch.long` tensor, fixing the `TypeError: 'numpy.float32' object cannot be interpreted as an integer`.
- **Memory usage** – Thanks to `numpy.memmap`, you can safely set `stride=1` for daily rolling windows without OOM crashes.
- **Cross‑platform** – The code checks for `torch.backends.mps.is_available()` and falls back to CPU if MPS is not present.

---

## 📚 Documentation

- English README (this file) – explains the overall project and how to run the Transformer pipeline.
- Korean README – see `transformer/README_KR.md` for a Korean version of the Transformer documentation.

---

## 📂 Project Tree (excerpt)

```
PriceTrends/
├── core/
├── prediction/
├── transformer/
│   ├── README.md            # English docs (this file)
│   ├── README_KR.md         # Korean docs
│   ├── model.py
│   ├── pipeline.py          # memmap window creation
│   ├── params.py            # mode + timeframe config loader
│   ├── train.py
│   └── ...
├── daily/
├── utils/
└── README.md                # Top‑level project overview (this file)
```

---

## 🎉 Thanks

Feel free to open issues or submit pull requests. Happy modeling!
