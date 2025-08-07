import os
import pandas as pd
from typing import Optional, List
from tqdm import tqdm

from W_price_trends.loader import DataLoader
from W_price_trends.image import ChartConfig, MarketData, ChartGenerator


class RealTimeProcessor:
    def __init__(self, chart_generator: ChartGenerator, max_date_discrepancy_days: int = 1) -> None:
        self.generator = chart_generator
        self.max_date_discrepancy = pd.Timedelta(days=max_date_discrepancy_days)

    def generate_realtime_charts(self, target_date: str, tickers: Optional[List[str]] = None) -> None:
        target_timestamp = pd.Timestamp(target_date)
        all_tickers = self.generator.market_data.close.columns
        selected_tickers = tickers if tickers is not None else all_tickers
        
        frequencies = [5, 20, 60]
        
        print(f"Generating real-time charts for {len(selected_tickers)} tickers on {target_date}")
        
        for freq in frequencies:
            config = ChartConfig(
                intervals=freq,
                image_height=32 if freq == 5 else 64 if freq == 20 else 96,
                include_ma=True,
                include_volume=True,
                img_save_dir=os.path.join(os.path.dirname(__file__), 'Images_r')
            )
            
            temp_generator = ChartGenerator(self.generator.market_data, config)
            
            save_dir = os.path.join(config.img_save_dir, str(freq))
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\nProcessing {freq}-day charts...")
            
            successful_charts = 0
            failed_charts = 0
            
            for ticker in tqdm(selected_tickers, desc=f"{freq}d charts"):
                try:
                    ticker_dates = temp_generator.ma_data[ticker].dropna().index
                    
                    actual_target = target_timestamp
                    if target_timestamp not in ticker_dates:
                        available_before = ticker_dates[ticker_dates <= target_timestamp]
                        if len(available_before) == 0:
                            raise ValueError(f"No data available on or before {target_timestamp}")
                        
                        latest_available_date = available_before[-1]
                        date_diff = target_timestamp - latest_available_date
                        
                        if date_diff > self.max_date_discrepancy:
                            raise ValueError(f"Data is too old. Last available: {latest_available_date.strftime('%Y-%m-%d')}, Gap: {date_diff.days} days")
                        
                        actual_target = latest_available_date
                    
                    target_idx = ticker_dates.get_loc(actual_target)
                    start_idx = target_idx - freq + 1
                    
                    if start_idx < 0:
                        raise ValueError(f"Not enough data: need {freq} days before {actual_target}")
                    
                    start_date = ticker_dates[start_idx]
                    end_date = ticker_dates[target_idx]
                    
                    estimation_start = end_date
                    estimation_end = end_date
                    
                    image, _ = temp_generator.generate_chart_image(
                        ticker, start_date, end_date, estimation_start, estimation_end)
                    
                    filename = f'{ticker}_{target_date}_{freq}d.png'
                    save_path = os.path.join(save_dir, filename)
                    image.save(save_path)
                    
                    successful_charts += 1
                    
                except Exception as e:
                    if "Data is too old" not in str(e) and "No data available" not in str(e) and "Not enough data" not in str(e):
                         print(f"Failed chart for {ticker}: {e}")
                    failed_charts += 1
            
            print(f"{freq}d charts: {successful_charts} success, {failed_charts} failed")

    def save_single_chart(self, ticker: str, target_date: str, intervals: int) -> None:
        target_timestamp = pd.Timestamp(target_date)
        
        config = ChartConfig(
            intervals=intervals,
            image_height=32 if intervals == 5 else 64 if intervals == 20 else 96,
            include_ma=True,
            include_volume=True,
            img_save_dir=os.path.join(os.path.dirname(__file__), 'Images_r')
        )
        
        temp_generator = ChartGenerator(self.generator.market_data, config)
        
        try:
            ticker_dates = temp_generator.ma_data[ticker].dropna().index
            
            actual_target = target_timestamp
            if target_timestamp not in ticker_dates:
                available_before = ticker_dates[ticker_dates <= target_timestamp]
                if len(available_before) == 0:
                    raise ValueError(f"No MA data available before {target_timestamp}")

                latest_available_date = available_before[-1]
                date_diff = target_timestamp - latest_available_date
                
                if date_diff > self.max_date_discrepancy:
                    raise ValueError(f"Data is too old. Last available: {latest_available_date.strftime('%Y-%m-%d')}, Gap: {date_diff.days} days")

                actual_target = latest_available_date
            
            target_idx = ticker_dates.get_loc(actual_target)
            start_idx = target_idx - intervals + 1
            
            if start_idx < 0:
                raise ValueError(f"Not enough data: need {intervals} days before {actual_target}")
            
            start_date = ticker_dates[start_idx]
            end_date = ticker_dates[target_idx]
            
            estimation_start = end_date
            estimation_end = end_date
            
            image, _ = temp_generator.generate_chart_image(
                ticker, start_date, end_date, estimation_start, estimation_end)
            
            save_dir = os.path.join(config.img_save_dir, str(intervals))
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f'{ticker}_{target_date}_{intervals}d.png'
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            
            print(f"Saved: {save_path}")
            
        except Exception as e:
            print(f"Failed to save {ticker} {intervals}d chart: {e}")
            raise


class GenerateImages_r:
    def __init__(self, 
                 o_data: pd.DataFrame, 
                 l_data: pd.DataFrame, 
                 h_data: pd.DataFrame, 
                 c_data: pd.DataFrame, 
                 v_data: pd.DataFrame,
                 max_date_discrepancy_days: int = 1) -> None:
        
        market_data = MarketData(
            open=o_data,
            low=l_data, 
            high=h_data,
            close=c_data,
            volume=v_data
        )
        
        config = ChartConfig()
        self.generator = ChartGenerator(market_data, config)
        self.realtime_processor = RealTimeProcessor(self.generator, max_date_discrepancy_days)

    def generate_realtime_images(self, target_date: str, tickers: Optional[List[str]] = None) -> None:
        self.realtime_processor.generate_realtime_charts(target_date, tickers=tickers)

    def save_single_image(self, ticker: str, target_date: str, intervals: int) -> None:
        self.realtime_processor.save_single_chart(ticker, target_date, intervals)


def run(target_date: str = "20250630", tickers: Optional[List[str]] = None, max_date_discrepancy_days: int = 1):
    print(f"=== Running Real-Time Chart Generation for {target_date} ===")
    
    loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "DATA"))
    print("Available datasets:", loader.available())
    
    open_data = loader.load("open")
    low_data = loader.load("low") 
    high_data = loader.load("high")
    close_data = loader.load("close")
    volume_data = loader.load("volume")
    
    generator = GenerateImages_r(
        o_data=open_data,
        l_data=low_data,
        h_data=high_data,
        c_data=close_data,
        v_data=volume_data,
        max_date_discrepancy_days=max_date_discrepancy_days
    )
    
    print(f"Generating charts for date: {target_date}")
    if tickers:
        print(f"Selected tickers: {tickers}")
    else:
        print("Processing all available tickers")
    
    generator.generate_realtime_images(target_date, tickers=tickers)
    
    print("\n=== Real-time generation completed ===")
    print("Charts saved to Images_r/5, Images_r/20, Images_r/60 directories")


if __name__ == "__main__":
    run(target_date="20250731", max_date_discrepancy_days=1) 