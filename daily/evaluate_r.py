import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, Optional, List, Union

from prediction.evaluate import ModelEvaluator
from core.params import CNNParams


class RealTimeDataset(Dataset):
    def __init__(self, images_dir: str, intervals: int, target_date: str) -> None:
        self.intervals = intervals
        self.target_date = target_date
        
        interval_dir = os.path.join(images_dir, str(intervals))
        if not os.path.exists(interval_dir):
            raise FileNotFoundError(f"Images directory not found: {interval_dir}")
        
        self.image_files = []
        for filename in os.listdir(interval_dir):
            if filename.endswith(f'{target_date}_{intervals}d.png'):
                ticker = filename.split('_')[0]
                self.image_files.append({
                    'ticker': ticker,
                    'filepath': os.path.join(interval_dir, filename),
                    'filename': filename
                })
        
        if not self.image_files:
            raise ValueError(f"No images found for date {target_date} and interval {intervals}d")
        
        sample_image = Image.open(self.image_files[0]['filepath'])
        self.image_width = sample_image.width
        self.image_height = sample_image.height
        
        print(f"Found {len(self.image_files)} images for {intervals}d interval")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        item = self.image_files[idx]
        
        image = Image.open(item['filepath'])
        image_array = np.array(image, dtype=np.float32)
        image_tensor = torch.FloatTensor(image_array).unsqueeze(0)
        
        return {
            'image': image_tensor,
            'label': torch.tensor(0, dtype=torch.long),
            'StockID': item['ticker'],
            'ending_date': self.target_date
        }


class RealTimeEvaluator(ModelEvaluator):
    def get_realtime_dataloader(self, images_dir: str, target_date: str):
        try:
            dataset = RealTimeDataset(images_dir, self.input_days, target_date)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading dataset for {self.model_name}: {e}")
            return None

        if len(dataset) == 0:
            return None

        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, 
                               shuffle=False, num_workers=0)
        return dataloader

    def predict_realtime(self, images_dir: str, target_date: str) -> Optional[pd.DataFrame]:
        dataloader = self.get_realtime_dataloader(images_dir, target_date)
        if dataloader is None:
            return None

        original_get_test_dataloader = self.get_test_dataloader
        self.get_test_dataloader = lambda: dataloader
        
        result_df = self.predict()
        
        self.get_test_dataloader = original_get_test_dataloader
        return result_df


class RealTimeScore:
    def __init__(self, target_date: str):
        self.target_date = target_date
        self.pivot_dict = {}

    def create_pivot_dict(self, all_results: Dict[str, pd.DataFrame]) -> Dict:
        pivot_dict = {}
        
        for model_name, df in all_results.items():
            if df is None or df.empty:
                continue
                
            df = df.copy()
            df['ending_date'] = pd.to_datetime(self.target_date, format='%Y%m%d')
            
            model_pivots = {}
            for col in ['prediction', 'prob_up', 'prob_down']:
                if col in df.columns:
                    pivot = df.pivot(index='ending_date', columns='StockID', values=col)
                    model_pivots[col] = pivot
            
            pivot_dict[model_name] = model_pivots
        
        self.pivot_dict = pivot_dict
        return pivot_dict

    def create_ensemble_average(self) -> Dict[str, Union[pd.DataFrame, int]]:
        if not self.pivot_dict:
            return {}
        
        common_stocks = None
        for model_name, pivots in self.pivot_dict.items():
            if 'prob_up' in pivots:
                model_stocks = set(pivots['prob_up'].columns)
                if common_stocks is None:
                    common_stocks = model_stocks
                else:
                    common_stocks = common_stocks.intersection(model_stocks)
        
        if not common_stocks:
            return {}
        
        common_stocks = sorted(list(common_stocks))
        print(f"Common stocks across all models: {len(common_stocks)}")
        
        prob_up_dfs = []
        prob_down_dfs = []
        
        for model_name, pivots in self.pivot_dict.items():
            if 'prob_up' in pivots and 'prob_down' in pivots:
                aligned_up = pivots['prob_up'][common_stocks]
                aligned_down = pivots['prob_down'][common_stocks]
                
                prob_up_dfs.append(aligned_up)
                prob_down_dfs.append(aligned_down)
        
        if not prob_up_dfs:
            return {}
        
        avg_prob_up = sum(prob_up_dfs) / len(prob_up_dfs)
        avg_prob_down = sum(prob_down_dfs) / len(prob_down_dfs)
        avg_prediction = (avg_prob_up > 0.5).astype(int)
        
        ensemble_result = {
            'prob_up': avg_prob_up,
            'prob_down': avg_prob_down,
            'prediction': avg_prediction,
            'model_count': len(prob_up_dfs)
        }
        
        print(f"Created ensemble average from {len(prob_up_dfs)} models")
        return ensemble_result


class RealTimeViewer:
    def __init__(self, model_pivots: Dict[str, pd.DataFrame]):
        self._pivots = model_pivots

    def __getitem__(self, col: str) -> pd.DataFrame:
        if col not in self._pivots:
            raise KeyError(f"'{col}' not found. Available data: {list(self._pivots.keys())}")
        
        df = self._pivots[col]
        
        if isinstance(df, pd.DataFrame) and not df.empty:
            series = df.iloc[0].sort_values(ascending=True)
            return series.to_frame(col)
        return df
    
    def __repr__(self) -> str:
        return f"<RealTimeViewer(keys={list(self._pivots.keys())})>"


class RealTimePredictor:
    def __init__(self, target_date: str, images_dir: Optional[str] = None, 
                 intervals: Optional[List[int]] = None):
        self.target_date = target_date
        self.images_dir = images_dir or os.path.join(os.path.dirname(__file__), 'Images_r')
        self.intervals = intervals or [5, 20, 60]
        
        self.all_results: Dict[str, pd.DataFrame] = {}
        self.pivot_dict: Dict[str, Dict[str, pd.DataFrame]] = {}
        self.ensemble_avg: Dict[str, Union[pd.DataFrame, int]] = {}

    def execute(self) -> 'RealTimePredictor':
        print(f"Real-time prediction for {self.target_date}")
        
        params = CNNParams()
        
        for input_days in self.intervals:
            if input_days not in params.window_sizes:
                continue

            config = params.get_config('TEST', input_days)
            evaluator = RealTimeEvaluator(input_days=input_days, return_days=input_days, config=config)
            results_df = evaluator.predict_realtime(self.images_dir, self.target_date)

            if results_df is not None:
                model_name = f"I{input_days}"
                print(f"Model {model_name}: {len(results_df)} predictions")
                self.all_results[model_name] = results_df

        scorer = RealTimeScore(self.target_date)
        self.pivot_dict = scorer.create_pivot_dict(self.all_results)
        self.ensemble_avg = scorer.create_ensemble_average()
        
        return self

    def __repr__(self) -> str:
        model_count = self.ensemble_avg.get('model_count', 0)
        status = f"{model_count} models in ensemble" if model_count > 0 else "Not executed or no results"
        return f"<RealTimePredictor(target_date='{self.target_date}', {status})>"

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name in ('prob_up', 'prob_down', 'prediction'):
            if name in self.ensemble_avg:
                df = self.ensemble_avg[name]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    series = df.iloc[0].sort_values(ascending=True)
                    return series.to_frame(name)
                return df
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'. Did you forget to call .execute()?")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, model_name: str) -> RealTimeViewer:
        if model_name not in self.pivot_dict:
            raise KeyError(f"Model '{model_name}' not found. Available models: {list(self.pivot_dict.keys())}")
        return RealTimeViewer(self.pivot_dict[model_name])


if __name__ == "__main__":
    dates = ["20250822"]
    all_prob_ups = {}
    
    for date in dates:
        print(f"\nProcessing date: {date}")
        predictor = RealTimePredictor(target_date=date).execute()
        
        if predictor.ensemble_avg and 'prob_up' in predictor.ensemble_avg:
            prob_up_series = predictor.ensemble_avg['prob_up'].iloc[0]
            all_prob_ups[date] = prob_up_series
            print(f"Got {len(prob_up_series)} stocks for {date}")
    
    if all_prob_ups:
        df = pd.DataFrame(all_prob_ups)
