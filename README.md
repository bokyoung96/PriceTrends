## PriceTrends: CNN-Based KOSPI200 Price Trend Prediction

A deep learning system that predicts price trends in the Korean KOSPI200 market using Convolutional Neural Networks (CNN) trained on chart images.

## 🎯 Project Overview

This project implements an innovative approach to financial market prediction by converting price data into chart images and training CNN models to identify price trends. The system processes Korean KOSPI200 market data and generates probability scores for upward/downward price movements.

### Key Features

- **Chart Image Generation**: Converts OHLCV data into compact chart images (32x15 to 96x180 pixels)
- **Multi-Timeframe Analysis**: Supports 5-day, 20-day, and 60-day prediction windows
- **Ensemble Learning**: Combines multiple CNN models for robust predictions
- **Production-Ready Pipeline**: Complete data processing, training, and evaluation workflow

## 🏗️ Architecture

### Data Pipeline Flow

```
Excel Data → Parquet Conversion → Chart Images → CNN Training → Prediction → Results
```

### Core Components

1. **Data Processing** (`loader.py`)
   - Converts Excel market data to efficient Parquet format
   - Handles data validation and filtering

2. **Image Generation** (`image.py`)
   - Transforms price data into chart images
   - Includes OHLCV visualization with moving averages

3. **Model Training** (`training.py`)
   - CNN architecture optimized for chart pattern recognition
   - Ensemble training with multiple model instances

4. **Evaluation** (`evaluate.py`)
   - Generates prediction probabilities
   - Calculates model performance metrics

5. **Results Processing** (`score.py`)
   - Converts predictions to pivot table format
   - Provides ensemble averaging across timeframes

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
from loader import DataConverter
converter = DataConverter("DATA.xlsx", "DATA/")
converter.data_convert()
```

2. **Image Generation**
```python
from image import ChartBatchProcessor
processor = ChartBatchProcessor()
processor.generate_batch_dataset()
```

3. **Model Training**
```python
from training import Trainer
trainer = Trainer()
trainer.train_empirical_ensem_model()
```

4. **Prediction**
```python
from evaluate import ModelEvaluator
evaluator = ModelEvaluator(input_days=5, return_days=5, config=config)
predictions = evaluator.predict()
```

5. **Results Analysis**
```python
from score import ResultLoader
loader = ResultLoader()
final_scores = loader.avg_prob  # Ensemble predictions
```

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
from params import CNNParams
params = CNNParams()
config = params.get_config("TEST", 5)  # Test mode, 5-day window
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

### Final Ensemble Scores
```python
# Pivot table format: Date × Stock × Probability
final_scores = loader.avg_prob
# Shape: (dates, stocks) with values 0-1
```

## 🔧 Advanced Usage

### Custom Timeframes
```python
# Train custom window size
config = params.get_config("PRODUCTION", 10)  # 10-day window
```

### Model Evaluation
```python
# Load specific model results
results = loader.load_results(5, 5)  # 5-day input, 5-day return
prob_up = results['prob_up']  # Pivot table of up probabilities
```

### Chart Visualization
```python
from read import ChartViewer
viewer = ChartViewer(intervals=5)
viewer.display_charts("A005930", [270, 6090])  # Samsung Electronics
```

## 📁 Project Structure

```
PriceTrends/
├── params.py           # Configuration management
├── config.json         # Model hyperparameters
├── loader.py           # Data loading and conversion
├── image.py            # Chart image generation
├── training.py         # CNN model training
├── evaluate.py         # Model evaluation and prediction
├── score.py            # Results processing and analysis
├── read.py             # Chart visualization tools
├── pipeline.md         # Detailed pipeline documentation
├── DATA/               # Processed market data
├── Images/             # Generated chart images
├── models/             # Trained model checkpoints
└── results/            # Prediction outputs
```

## 🎯 Performance Metrics

The system evaluates performance using:
- **Accuracy**: Overall prediction correctness
- **Precision/Recall**: For up/down trend classification
- **Ensemble Stability**: Cross-model prediction consistency

## 🔬 Technical Details

### Data Processing
- **Input**: OHLCV data from KOSPI200 constituents
- **Preprocessing**: Invalid ticker filtering, data validation
- **Output**: Normalized chart images with volume indicators

### Model Training
- **Framework**: PyTorch
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, BatchNorm, early stopping

### Prediction Pipeline
- **Ensemble Method**: Average of multiple model outputs
- **Probability Calibration**: Softmax normalization
- **Time Alignment**: Synchronized predictions across timeframes

## 📚 References

This project demonstrates the application of computer vision techniques to financial time series analysis, leveraging the pattern recognition capabilities of CNNs for market trend prediction.

## 🤝 Contributing

For questions or contributions, please refer to the detailed pipeline documentation in `pipeline.md`.

---

**Note**: This system is designed for research and educational purposes. Financial predictions carry inherent risks and should not be used as the sole basis for investment decisions.
