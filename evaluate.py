import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from typing import List, Dict, Optional

from training import KoreanEquityDataset, CNNModel


class ResultAggregator:
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = os.path.join(os.path.dirname(__file__), results_dir)
        os.makedirs(self.results_dir, exist_ok=True)

    def print_metrics(self, labels: pd.Series, predictions: pd.Series, model_name: str) -> None:
        print(f"\n--- Evaluation Metrics for {model_name} ---")
        print(classification_report(labels, predictions,
              target_names=['Down', 'Up'], zero_division=0))
        print("Confusion Matrix:")
        cm = confusion_matrix(labels, predictions)
        print(cm)

    def process_and_save_results(self, results_by_ws: Dict[int, pd.DataFrame], window_sizes: List[int]) -> None:
        if not results_by_ws:
            print("No results to process.")
            return

        print("\n" + "="*25 + " INDIVIDUAL MODEL EVALUATION " + "="*25)
        for ws, df in results_by_ws.items():
            self.print_metrics(df['label'], df['prediction'], f"{ws}d Model")

            individual_file = f"test_results_{ws}d.csv"
            df.to_csv(os.path.join(self.results_dir,
                      individual_file), index=False)
            print(f"Individual {ws}d results saved to {individual_file}")

        if len(results_by_ws) < 2:
            print("\nInsufficient models for ensemble evaluation. Skipping.")
            return

        print("\n" + "="*25 + " ENSEMBLE MODEL EVALUATION " + "="*25)

        sorted_ws = sorted(results_by_ws.keys())
        base_df = results_by_ws[sorted_ws[0]]

        for ws in sorted_ws[1:]:
            merge_df = results_by_ws[ws]
            base_df = pd.merge(
                base_df.rename(columns=lambda c: f"{c}_{sorted_ws[0]}d" if c not in [
                               'StockID', 'ending_date', 'label'] else c),
                merge_df.rename(columns=lambda c: f"{c}_{ws}d" if c not in [
                                'StockID', 'ending_date', 'label'] else c),
                on=['StockID', 'ending_date', 'label']
            )

        confidence_cols = [
            col for col in base_df.columns if 'confidence_up_' in col]
        base_df['ensemble_confidence_up'] = base_df[confidence_cols].mean(
            axis=1)
        base_df['ensemble_prediction'] = (
            base_df['ensemble_confidence_up'] > 0.5).astype(int)

        self.print_metrics(base_df['label'], base_df['ensemble_prediction'],
                           f"Ensemble ({'+'.join(map(str, sorted_ws))}d) Model")

        window_str = '_'.join([f"{ws}d" for ws in sorted_ws])
        ensemble_file = f"test_results_ensemble_{window_str}.csv"
        base_df.to_csv(os.path.join(
            self.results_dir, ensemble_file), index=False)
        print(f"\nEnsemble results saved to {ensemble_file}")


class Evaluator:
    def __init__(self, ws: int, pw: int, config: Dict) -> None:
        self.ws = ws
        self.pw = pw
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.exp_name = f"korea_cnn_{ws}d{pw}p_{config['mode']}"
        self.model_dir = os.path.join(
            os.path.dirname(__file__), 'models', self.exp_name)
        print(f"Using device: {self.device} for model {self.exp_name}")

    def get_test_dataloader(self) -> Optional[DataLoader]:
        try:
            test_dataset = KoreanEquityDataset(
                self.ws, self.config['test_years'])
        except FileNotFoundError:
            print(
                f"No data found for years {self.config['test_years']}. Skipping evaluation for ws={self.ws}.")
            return None

        if len(test_dataset) == 0:
            print(
                f"No data available for years {self.config['test_years']} in the dataset (ws={self.ws}). Skipping.")
            return None

        test_loader = DataLoader(
            test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0)
        print(f"Test size for ws={self.ws}: {len(test_dataset)}")
        return test_loader

    def generate_predictions(self) -> Optional[pd.DataFrame]:
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
                f"\n--- Evaluating Ensemble Member {model_num + 1}/{self.config['ensem_size']} for ws={self.ws} ---")

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

            model_outputs_per_run = []

            desc = f"Predicting with model {model_num+1} (ws={self.ws})"

            with torch.no_grad():
                for batch in tqdm(test_loader, desc=desc):
                    inputs = batch['image'].to(self.device)
                    outputs = model(inputs)
                    model_outputs_per_run.append(outputs.cpu())

                    if model_num == 0:
                        all_labels.append(batch['label'])
                        all_stock_ids.extend(batch['StockID'])
                        all_ending_dates.extend(batch['ending_date'])

            all_outputs.append(torch.cat(model_outputs_per_run))

        if not all_outputs:
            print(f"No models were evaluated for ws={self.ws}. Exiting.")
            return None

        ensemble_outputs = torch.stack(all_outputs).mean(dim=0)
        _, preds = torch.max(ensemble_outputs, 1)
        all_labels_tensor = torch.cat(all_labels)

        return pd.DataFrame({
            'StockID': all_stock_ids,
            'ending_date': all_ending_dates,
            'label': all_labels_tensor.numpy(),
            'prediction': preds.numpy(),
            'confidence_down': torch.softmax(ensemble_outputs, dim=1)[:, 0].numpy(),
            'confidence_up': torch.softmax(ensemble_outputs, dim=1)[:, 1].numpy(),
        })


def main():
    RUN_MODE = 'TEST'
    TEST_YEARS = list(range(2012, 2025))
    EVALUATE_WINDOWS = [5, 20]

    MODE_CONFIGS = {
        'TEST': {
            'mode': 'test',
            'test_years': TEST_YEARS,
            'ensem_size': 1,
            'batch_size': 64,
            'drop_prob': 0.3,
            'conv_channels': [32, 64, 128],
        },
        'PRODUCTION': {
            'mode': 'production',
            'test_years': TEST_YEARS,
            'ensem_size': 5,
            'batch_size': 256,
            'drop_prob': 0.5,
            'conv_channels': [32, 64, 128, 256],
        }
    }

    WINDOW_CONFIGS = {
        5: {
            'pw': 5,
            'filter_sizes': {
                'TEST': [(3, 2), (3, 2), (3, 2)],
                'PRODUCTION': [(3, 2), (3, 2), (3, 2), (3, 2)]
            }
        },
        20: {
            'pw': 20,
            'filter_sizes': {
                'TEST': [(5, 3), (3, 3), (3, 3)],
                'PRODUCTION': [(5, 3), (3, 3), (3, 3), (3, 3)]
            }
        },
        60: {
            'pw': 60,
            'filter_sizes': {
                'TEST': [(5, 3), (5, 3), (3, 3)],
                'PRODUCTION': [(5, 3), (5, 3), (3, 3), (3, 3)]
            }
        }
    }

    all_results = {}

    for ws in EVALUATE_WINDOWS:
        if ws not in WINDOW_CONFIGS:
            print(
                f"Warning: Window size {ws} not found in WINDOW_CONFIGS. Skipping.")
            continue

        window_config = WINDOW_CONFIGS[ws]
        pw = window_config['pw']
        print(f"\n{'='*25} EVALUATING MODEL: {ws}d{pw}p {'='*25}")
        print(f"--- Running in {RUN_MODE} MODE ---")

        config = MODE_CONFIGS[RUN_MODE].copy()
        config['filter_sizes'] = window_config['filter_sizes'][RUN_MODE]

        evaluator = Evaluator(ws=ws, pw=pw, config=config)
        results_df = evaluator.generate_predictions()
        if results_df is not None:
            all_results[ws] = results_df

    aggregator = ResultAggregator()
    aggregator.process_and_save_results(all_results, EVALUATE_WINDOWS)


if __name__ == "__main__":
    main()
