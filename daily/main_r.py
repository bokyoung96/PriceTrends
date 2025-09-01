import sys
import json
import pandas as pd
from typing import Optional, List, Dict
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.loader import DataLoader
from daily.image_r import GenerateImages_r
from daily.evaluate_r import RealTimePredictor


@dataclass
class Config:
    target_date: Optional[str] = None
    tickers: Optional[List[str]] = None
    intervals: List[int] = field(default_factory=lambda: [5, 20, 60])
    max_date_discrepancy: int = 1
    top_n: int = 10
    skip_existing_images: bool = False
    save_results: bool = True
    results_dir: str = "results_d"


class ImageProcessor:
    def __init__(self, loader: DataLoader, config: Config):
        self.loader = loader
        self.config = config
        self.images_dir = Path(__file__).parent.parent / "Images_r"
        
    def process(self, target_date: str):
        if (self.config.skip_existing_images and 
            all(self.images_dir.joinpath(str(i)).glob(f'*_{target_date}_{i}d.png') 
                for i in self.config.intervals if self.images_dir.joinpath(str(i)).exists())):
            print("Using existing charts")
            return
            
        print("Generating charts...")
        
        data_dict = {dtype: self.loader.to_pandas(self.loader.load(dtype)) 
                    for dtype in ["open", "low", "high", "close", "volume"]}
        
        GenerateImages_r(
            o_data=data_dict["open"],
            l_data=data_dict["low"], 
            h_data=data_dict["high"],
            c_data=data_dict["close"],
            v_data=data_dict["volume"],
            max_date_discrepancy_days=self.config.max_date_discrepancy
        ).generate_realtime_images(target_date, tickers=self.config.tickers)


class PredictionResult:
    def __init__(self, predictor: RealTimePredictor, config: Config):
        self.config = config
        self.target_date = config.target_date
        self.predictor = predictor
        
        ensemble = predictor.ensemble_avg or {}
        self.prob_up = self._extract_data(ensemble.get('prob_up'))
        self.prob_down = self._extract_data(ensemble.get('prob_down'))
        self.prediction = self._extract_data(ensemble.get('prediction'))
        self.model_count = ensemble.get('model_count', 0)
        
    def _extract_data(self, data):
        if data is None:
            return pd.Series()
        return data.iloc[0] if isinstance(data, pd.DataFrame) else data
    
    @property
    def top_stocks(self) -> pd.Series:
        return self.prob_up.nlargest(self.config.top_n)
    
    @property 
    def bottom_stocks(self) -> pd.Series:
        return self.prob_up.nsmallest(self.config.top_n)
    
    def get_ticker(self, ticker: str) -> Dict[str, float]:
        return ({k: getattr(self, k)[ticker] for k in ['prob_up', 'prob_down', 'prediction']} 
                if ticker in self.prob_up.index else {})
    
    def save(self):
        if not self.config.save_results:
            return
            
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)
        
        final_results = {}
        
        if hasattr(self.predictor, 'all_results') and self.predictor.all_results:
            for model_name, result_df in self.predictor.all_results.items():
                if result_df is not None and not result_df.empty and 'prob_up' in result_df.columns:
                    interval = model_name.replace('I', '')
                    prob_up_series = result_df.set_index('StockID')['prob_up']
                    final_results[f'{interval}d'] = prob_up_series
        
        if not self.prob_up.empty:
            final_results['avg'] = self.prob_up
        
        if final_results:
            combined_df = pd.DataFrame(final_results)
            
            excel_path = results_dir / f'pred_{self.target_date}.xlsx'
            combined_df.to_excel(excel_path, index=True, sheet_name='Predictions')
            print(f"All predictions saved: {excel_path}")
            
            print(f"Columns: {list(combined_df.columns)} | Stocks: {len(combined_df)}")
        
        summary = {
            'date': self.target_date,
            'intervals': list(final_results.keys()) if final_results else [],
            'stock_count': len(final_results.get('avg', [])) if final_results else 0
        }
        
        with open(results_dir / f'summary_{self.target_date}.json', 'w') as f:
            json.dump(summary, f, indent=2)


class StockPredictor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.loader = DataLoader(str(Path(__file__).parent.parent / "DATA"))
        self.image_processor = ImageProcessor(self.loader, self.config)
        
    def predict(self, target_date: Optional[str] = None) -> PredictionResult:
        self.config.target_date = target_date or self.config.target_date or self.loader.get_latest_date()
        print(f"Prediction date: {self.config.target_date}")
        
        self.image_processor.process(self.config.target_date)
        
        print("AI model predicting...")
        predictor = RealTimePredictor(target_date=self.config.target_date, intervals=self.config.intervals).execute()
        
        result = PredictionResult(predictor, self.config)
        
        if self.config.save_results:
            result.save()
        self._print_results(result)
        
        return result
    
    def _print_results(self, result: PredictionResult):
        if result.prob_up.empty:
            print("No prediction results")
            return
            
        print(f"\nPrediction Results - Models: {result.model_count}, Stocks: {len(result.prob_up)}")
        
        for label, stocks in [("TOP", result.top_stocks), ("BOTTOM", result.bottom_stocks)]:
            print(f"\n{label} {self.config.top_n}")
            for ticker, prob in stocks.items():
                print(f"  {ticker:8s} : {prob:6.2%}")


def predict(date: Optional[str] = None, tickers: Optional[List[str]] = None) -> PredictionResult:
    config = Config(target_date=date, tickers=tickers)
    predictor = StockPredictor(config)
    return predictor.predict()


def quick_check(ticker: str, date: Optional[str] = None) -> float:
    result = predict(date=date, tickers=[ticker])
    info = result.get_ticker(ticker)
    return info.get('prob_up', 0.0)


def batch_predict(dates: List[str]) -> pd.DataFrame:
    all_results = {}
    
    for date in dates:
        print(f"\n{'='*40}")
        print(f"Date: {date}")
        result = predict(date=date)
        if not result.prob_up.empty:
            all_results[date] = result.prob_up
            
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


if __name__ == "__main__":
    result = predict('20250228')