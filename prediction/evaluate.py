import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from typing import Dict, Optional

from core.training import KoreanEquityDataset, CNNModel
from core.params import CNNParams
from core.trainer import DeviceSelector


class AccuracyResult:
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = os.path.join(os.path.dirname(__file__), results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        self.accuracy_summary = {}

    def print_accuracy(self, labels: pd.Series, predictions: pd.Series, model_name: str) -> float:
        accuracy = (labels == predictions).mean()
        print(f"\n--- {model_name} Out-of-Sample Classification Accuracy ---")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(classification_report(labels, predictions,
              target_names=['Down', 'Up'], zero_division=0))

        cm = confusion_matrix(labels, predictions)
        print("Confusion Matrix:")
        print(cm)
        return accuracy

    def save_res(self, model_name: str, df: pd.DataFrame) -> None:
        if df.empty:
            print(f"No results to save for {model_name}.")
            return

        print("\n" + "="*50)
        print(f"RESULTS FOR {model_name}")
        print("="*50)

        accuracy = self.print_accuracy(
            df['label'], df['prediction'], model_name)
        self.accuracy_summary[model_name] = accuracy

        result_file = f"test_results_{model_name.lower().replace('/', '_')}.parquet"
        df.to_parquet(os.path.join(self.results_dir, result_file), index=False)
        print(f"Results saved to {result_file}")

    def print_summary(self) -> None:
        if not self.accuracy_summary:
            print("\nNo results to summarize.")
            return

        print("\n" + "="*30 + " FINAL SUMMARY " + "="*30)
        for model_name, accuracy in self.accuracy_summary.items():
            print(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")


class ModelEvaluator:
    def __init__(self, input_days: int, return_days: int, config: Dict) -> None:
        self.input_days = input_days
        self.return_days = return_days
        self.config = config
        selector = DeviceSelector()
        self.device = selector.resolve()
        self.model_name = f"I{input_days}/R{return_days}"
        self.exp_name = f"korea_cnn_{input_days}d{return_days}p_{config['mode']}"
        self.model_dir = os.path.join(
            os.path.dirname(__file__), '..', 'models', self.exp_name)
        print(selector.summary(self.model_name))

    def get_test_dataloader(self) -> Optional[DataLoader]:
        try:
            test_dataset = KoreanEquityDataset(
                self.input_days, self.config['test_years'])
        except FileNotFoundError:
            print(
                f"No data found for years {self.config['test_years']}. Skipping {self.model_name}.")
            return None

        if len(test_dataset) == 0:
            print(f"No data available for {self.model_name}. Skipping.")
            return None

        test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        print(f"Test size for {self.model_name}: {len(test_dataset)}")
        return test_loader

    def predict(self) -> Optional[pd.DataFrame]:
        test_loader = self.get_test_dataloader()
        if test_loader is None:
            return None

        ds = test_loader.dataset
        paddings = [(int(fs[0] / 2), int(fs[1] / 2))
                    for fs in self.config['filter_sizes']]

        all_outputs, all_labels, all_stock_ids, all_ending_dates = [], [], [], []

        for model_num in range(self.config['ensem_size']):
            model_path = os.path.join(
                self.model_dir, f"checkpoint{model_num}.pth.tar")
            if not os.path.exists(model_path):
                print(f"Model checkpoint not found at {model_path}, skipping.")
                continue

            print(
                f"Evaluating ensemble member {model_num + 1}/{self.config['ensem_size']} for {self.model_name}")

            model = CNNModel(
                layer_number=len(self.config['conv_channels']),
                input_size=(ds.image_height, ds.image_width),
                inplanes=self.config['conv_channels'][0],
                conv_layer_chanls=self.config['conv_channels'],
                drop_prob=self.config['drop_prob'],
                filter_size_list=self.config['filter_sizes'],
                stride_list=[(1, 1)] * len(self.config['conv_channels']),
                padding_list=paddings,
                dilation_list=[(1, 1)] * len(self.config['conv_channels']),
                max_pooling_list=[(2, 1)] * len(self.config['conv_channels']),
            ).to(self.device)

            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            model_outputs = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Predicting {self.model_name}"):
                    inputs = batch['image'].to(self.device)
                    outputs = model(inputs)
                    model_outputs.append(outputs.cpu())

                    if model_num == 0:
                        all_labels.append(batch['label'])
                        all_stock_ids.extend(batch['StockID'])
                        all_ending_dates.extend(batch['ending_date'])

            all_outputs.append(torch.cat(model_outputs))

        if not all_outputs:
            print(f"No models evaluated for {self.model_name}.")
            return None

        ensemble_outputs = torch.stack(all_outputs).mean(dim=0)
        probabilities = torch.softmax(ensemble_outputs, dim=1)
        predictions = (probabilities[:, 1] > 0.5).int()
        all_labels_tensor = torch.cat(all_labels)

        return pd.DataFrame({
            'StockID': all_stock_ids,
            'ending_date': all_ending_dates,
            'label': all_labels_tensor.numpy(),
            'prediction': predictions.cpu().numpy(),
            'prob_down': probabilities[:, 0].cpu().numpy(),
            'prob_up': probabilities[:, 1].cpu().numpy(),
        })


def main():
    RUN_MODE = 'PRODUCTION'

    EVALUATION_CONFIGS = [
        (5, 5),    # I5/R5: 5-day input, 5-day return prediction
        (20, 20),  # I20/R20: 20-day input, 20-day return prediction
        (60, 60),  # I60/R60: 60-day input, 60-day return prediction
    ]

    params = CNNParams()
    accuracy_result = AccuracyResult()

    for input_days, return_days in EVALUATION_CONFIGS:
        if input_days not in params.window_sizes:
            print(
                f"Warning: Input window {input_days} not in config. Skipping I{input_days}/R{return_days}.")
            continue

        print(f"\n{'='*50}")
        print(f"EVALUATING MODEL: I{input_days}/R{return_days}")
        print(f"Input: {input_days} days, Return horizon: {return_days} days")
        print(f"Running in {RUN_MODE} mode")
        print("="*50)

        config = params.get_config(RUN_MODE, input_days)
        config['test_years'] = params.get_test_years()

        evaluator = ModelEvaluator(
            input_days=input_days, return_days=return_days, config=config)
        results_df = evaluator.predict()

        if results_df is not None:
            model_name = f"I{input_days}/R{return_days}"
            accuracy_result.save_res(model_name, results_df)

    accuracy_result.print_summary()


if __name__ == "__main__":
    main()
