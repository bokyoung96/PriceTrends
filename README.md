## PriceTrends: CNN-Based KOSPI200 Price Trend Prediction

A deep learning system that predicts price trends in the Korean KOSPI200 market using Convolutional Neural Networks (CNN) trained on chart images.

## 🎯 Project Overview

This project implements an innovative approach to financial market prediction by converting price data into chart images and training CNN models to identify price trends. The system processes Korean KOSPI200 market data and generates probability scores for upward/downward price movements.

### Key Features

- **Chart Image Generation**: Converts OHLCV data into compact chart images (32x15 to 96x180 pixels)
- **Multi-Timeframe Analysis**: Supports 5-day, 20-day, and 60-day prediction windows
- **Ensemble Learning**: Combines multiple CNN models for robust predictions
- **Production-Ready Pipeline**: Complete data processing, training, and evaluation workflow
- **Daily Rebalancing**: Real-time prediction system for daily portfolio updates

## 🏗️ Architecture

### Data Pipeline Flow

```
Excel Data → Parquet Conversion → Chart Images → CNN Training → Prediction → Results
```

### Core Modules

#### 1. **Core Module** (`core/`)
- `loader.py`: Data loading and parquet conversion utilities
- `params.py`: Configuration and hyperparameter management  
- `training.py`: CNN model architecture and training logic

#### 2. **Prediction Module** (`prediction/`)
- `image.py`: Batch chart image generation from price data
- `evaluate.py`: Model evaluation and prediction generation
- `score.py`: Results processing and ensemble averaging

#### 3. **Daily Module** (`daily/`)
- `main_r.py`: Daily rebalancing orchestrator
- `image_r.py`: Real-time chart generation for current dates
- `evaluate_r.py`: Daily prediction pipeline

#### 4. **Utils Module** (`utils/`)
- `read.py`: Chart visualization and debugging tools
- `score_w.py`: Score writing utilities

## 📊 Model Architecture

### CNN Structure
- **Input**: Chart images (1 channel, variable dimensions)
- **Convolutional Layers**: 3-4 layers with BatchNorm and LeakyReLU
- **Pooling**: MaxPool2D for dimension reduction
- **Output**: Binary classification (up/down trend)

### Timeframe Configurations
- **5-day**: 32×15 pixel images, 3 conv layers
- **20-day**: 64×60 pixel images, 3 conv layers  
- **60-day**: 96×180 pixel images, 3 conv layers

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch pandas numpy matplotlib pillow pyarrow
```

### Basic Usage

1. **Data Preparation**
```python
from core.loader import DataConverter
converter = DataConverter("DATA/DATA.xlsx", "DATA/")
converter.data_convert()
```

2. **Batch Image Generation**
```python
from prediction.image import ChartBatchProcessor
processor = ChartBatchProcessor()
processor.generate_batch_dataset()
```

3. **Model Training**
```python
from core.training import Trainer
from core.params import CNNParams

params = CNNParams()
config = params.get_config("TEST", 5)  # Test mode, 5-day window
trainer = Trainer()
trainer.train_empirical_ensem_model()
```

4. **Batch Prediction**
```python
from prediction.evaluate import ModelEvaluator
evaluator = ModelEvaluator(input_days=5, return_days=5, config=config)
predictions = evaluator.predict()
```

5. **Results Analysis**
```python
from prediction.score import ResultLoader
loader = ResultLoader()
final_scores = loader.avg_prob  # Ensemble predictions
```

## 📅 Daily Rebalancing

For real-time daily predictions:

```python
from daily.main_r import main

# Run daily pipeline for specific date
main(end_date="20250825", timeframes=[5, 20, 60])
```

This will:
1. Generate chart images for the current date
2. Run predictions using pre-trained models
3. Save results to `daily/results_d/`

## ⚙️ Configuration

### Mode Settings (`config.json`)

**Test Mode** (Development)
```json
{
  "ensem_size": 1,
  "batch_size": 64,
  "max_epoch": 10,
  "lr": 1e-4,
  "drop_prob": 0.3
}
```

**Production Mode** (Full Training)
```json
{
  "ensem_size": 5,
  "batch_size": 256,
  "max_epoch": 50,
  "lr": 1e-5,
  "drop_prob": 0.5
}
```

### Parameter Management
```python
from core.params import CNNParams
params = CNNParams()
config = params.get_config("PRODUCTION", 20)  # Production mode, 20-day window
```

## 📈 Output Format

### Prediction Results
The system generates probability scores for each stock-date combination:

| Column | Description |
|--------|-------------|
| `StockID` | Stock identifier |
| `ending_date` | Prediction date |
| `prob_up` | Probability of upward movement (0-1) |
| `prob_down` | Probability of downward movement (0-1) |
| `prediction` | Binary prediction (0/1) |
| `label` | Actual outcome (0/1) |

### Daily Predictions
```python
# Excel output: daily/results_d/pred_YYYYMMDD.xlsx
# JSON summary: daily/results_d/summary_YYYYMMDD.json
```

## 🔧 Advanced Usage

### Custom Timeframes
```python
# Train custom window size
from core.params import CNNParams
params = CNNParams()
config = params.get_config("PRODUCTION", 10)  # 10-day window
```

### Model Evaluation
```python
# Load specific model results
from prediction.score import ResultLoader
loader = ResultLoader()
results = loader.load_results(5, 5)  # 5-day input, 5-day return
prob_up = results['prob_up']  # Pivot table of up probabilities
```

### Chart Visualization
```python
from utils.read import ChartViewer
viewer = ChartViewer(intervals=5)
viewer.display_charts("A005930", [270, 6090])  # Samsung Electronics
```

## 📁 Project Structure

```
PriceTrends/
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── loader.py           # Data loading and conversion
│   ├── params.py           # Configuration management
│   └── training.py         # CNN model training
│
├── prediction/             # Batch prediction system
│   ├── __init__.py
│   ├── image.py           # Batch chart generation
│   ├── evaluate.py        # Model evaluation
│   └── score.py           # Results processing
│
├── daily/                  # Daily rebalancing system
│   ├── __init__.py
│   ├── main_r.py          # Daily pipeline orchestrator
│   ├── image_r.py         # Real-time chart generation
│   ├── evaluate_r.py      # Daily prediction logic
│   ├── Images_r/          # Daily chart images
│   └── results_d/         # Daily predictions
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── read.py            # Visualization tools
│   └── score_w.py         # Score utilities
│
├── DATA/                   # Market data (parquet files)
├── Images/                 # Generated chart images
├── models/                 # Trained model checkpoints
├── config.json            # Model hyperparameters
└── pipeline.md            # Detailed documentation
```

## 🎯 Performance Metrics

The system evaluates performance using:
- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: For up/down trend classification
- **Ensemble Stability**: Cross-model prediction consistency
- **Daily Tracking**: Real-time prediction accuracy

## 🔬 Technical Details

### Data Processing
- **Input**: OHLCV data from KOSPI200 constituents
- **Preprocessing**: Invalid ticker filtering, data validation
- **Storage**: Efficient parquet format for fast I/O
- **Output**: Normalized chart images with volume indicators

### Model Training
- **Framework**: PyTorch
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, BatchNorm, early stopping
- **Checkpointing**: Best model saving based on validation loss

### Prediction Pipeline
- **Ensemble Method**: Average of multiple model outputs
- **Probability Calibration**: Softmax normalization
- **Time Alignment**: Synchronized predictions across timeframes
- **Batch Processing**: Efficient GPU utilization

## 📚 References

This project demonstrates the application of computer vision techniques to financial time series analysis, leveraging the pattern recognition capabilities of CNNs for market trend prediction.

## 🤝 Contributing

For questions or contributions, please refer to the detailed pipeline documentation in `pipeline.md`.

---

**Note**: This system is designed for research and educational purposes. Financial predictions carry inherent risks and should not be used as the sole basis for investment decisions.