import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from core.params import CNNConfig, CNNParams
from core.training import KoreanEquityDataset, CNNModel
from utils.root import MODELS_ROOT, RESULTS_ROOT

logger = logging.getLogger(__name__)


class EvaluationLogger:
    def __init__(self, log: logging.Logger | None = None) -> None:
        self.log = log or logger

    def report(self, model_name: str, labels: pd.Series, predictions: pd.Series) -> float:
        accuracy = (labels == predictions).mean()
        self.log.info("--- %s Out-of-Sample Classification Accuracy ---", model_name)
        self.log.info("Accuracy: %.4f (%.2f%%)", accuracy, accuracy * 100)
        report = classification_report(
            labels,
            predictions,
            target_names=['Down', 'Up'],
            zero_division=0,
        )
        self.log.info("Classification Report:\n%s", report)

        cm = confusion_matrix(labels, predictions)
        self.log.info("Confusion Matrix:\n%s", cm)
        return float(accuracy)


class ResultWriter:
    def __init__(self, results_dir: Path = RESULTS_ROOT, log: logging.Logger | None = None) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or logger

    def save(self, model_name: str, df: pd.DataFrame) -> Path:
        result_file = f"price_trends_results_{model_name.lower().replace('/', '_')}.parquet"
        result_path = self.results_dir / result_file
        df.to_parquet(result_path, index=False)
        self.log.info("Results saved to %s", result_path)
        return result_path


class GetResult:
    def __init__(self, results_dir: Path = RESULTS_ROOT, mode: str = 'TEST') -> None:
        self.reporter = EvaluationLogger()
        self.writer = ResultWriter(results_dir)
        self.accuracy_summary: Dict[str, float] = {}
        self.mode = mode.lower()

    def save_res(self, model_name: str, df: pd.DataFrame) -> None:
        if df.empty:
            logger.warning("No results to save for %s", model_name)
            return

        logger.info("%s RESULTS FOR %s %s", "=" * 10, model_name, "=" * 10)

        accuracy = self.reporter.report(
            model_name=model_name,
            labels=df['label'],
            predictions=df['prediction'],
        )
        self.accuracy_summary[model_name] = accuracy
        enriched_name = f"{self.mode}_{model_name.lower().replace('/', '_')}"
        self.writer.save(enriched_name, df)

    def print_summary(self) -> None:
        if not self.accuracy_summary:
            logger.info("No results to summarize.")
            return

        logger.info("%s FINAL SUMMARY %s", "=" * 15, "=" * 15)
        for model_name, accuracy in self.accuracy_summary.items():
            logger.info("%s: %.4f (%.2f%%)", model_name, accuracy, accuracy * 100)


class BatchResultCollector:
    def __init__(self) -> None:
        self.model_outputs: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.stock_ids: List[str] = []
        self.ending_dates: List[str] = []

    def record_metadata(self, batch: Dict) -> None:
        self.labels.append(batch['label'])
        self.stock_ids.extend(batch['StockID'])
        self.ending_dates.extend(batch['ending_date'])

    def add_model_outputs(self, outputs: List[torch.Tensor]) -> None:
        if outputs:
            self.model_outputs.append(torch.cat(outputs))

    def finalize(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]]:
        if not self.model_outputs:
            return None
        ensemble_outputs = torch.stack(self.model_outputs).mean(dim=0)
        labels_tensor = torch.cat(self.labels) if self.labels else torch.tensor([], dtype=torch.long)
        return ensemble_outputs, labels_tensor, self.stock_ids, self.ending_dates


class ModelEvaluator:
    def __init__(self, input_days: int, return_days: int, config: CNNConfig, models_root: Path = MODELS_ROOT) -> None:
        self.input_days = input_days
        self.return_days = return_days
        self.config = config
        selector = DeviceSelector()
        self.device = selector.resolve()
        self.model_name = f"I{input_days}/R{return_days}"
        self.exp_name = f"korea_cnn_{input_days}d{return_days}p_{config.mode}"
        self.model_dir = Path(models_root) / self.exp_name
        logger.info(selector.summary(self.model_name))

    def get_dataloader(self) -> Optional[DataLoader]:
        if not self.config.test_years:
            logger.warning("No test years configured for %s.", self.model_name)
            return None
        try:
            dataset = KoreanEquityDataset(
                self.input_days, self.config.test_years)
        except FileNotFoundError:
            logger.warning(
                "No data found for years %s. Skipping %s.",
                self.config.test_years,
                self.model_name,
            )
            return None

        if len(dataset) == 0:
            logger.warning("No data available for %s. Skipping.", self.model_name)
            return None

        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        logger.info("Test size for %s: %d", self.model_name, len(dataset))
        return loader

    def predict(self) -> Optional[pd.DataFrame]:
        loader = self.get_dataloader()
        if loader is None:
            return None

        ds = loader.dataset
        paddings = [(int(fs[0] / 2), int(fs[1] / 2))
                    for fs in self.config.filter_sizes]

        collector = BatchResultCollector()

        for model_num in range(self.config.ensem_size):
            model_path = self.model_dir / f"checkpoint{model_num}.pth.tar"
            if not model_path.exists():
                logger.warning("Model checkpoint not found at %s, skipping.", model_path)
                continue

            logger.info(
                "Evaluating ensemble member %d/%d for %s",
                model_num + 1,
                self.config.ensem_size,
                self.model_name,
            )

            model = CNNModel(
                layer_number=len(self.config.conv_channels),
                input_size=(ds.image_height, ds.image_width),
                inplanes=self.config.conv_channels[0],
                conv_layer_chanls=self.config.conv_channels,
                drop_prob=self.config.drop_prob,
                filter_size_list=self.config.filter_sizes,
                stride_list=[(1, 1)] * len(self.config.conv_channels),
                padding_list=paddings,
                dilation_list=[(1, 1)] * len(self.config.conv_channels),
                max_pooling_list=[(2, 1)] * len(self.config.conv_channels),
            ).to(self.device)

            checkpoint = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            model_outputs = []

            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Predicting {self.model_name}"):
                    inputs = batch['image'].to(self.device)
                    outputs = model(inputs)
                    model_outputs.append(outputs.cpu())

                    if model_num == 0:
                        collector.record_metadata(batch)

            collector.add_model_outputs(model_outputs)

        finalized = collector.finalize()
        if finalized is None:
            logger.warning("No models evaluated for %s.", self.model_name)
            return None

        ensemble_outputs, all_labels_tensor, all_stock_ids, all_ending_dates = finalized
        probabilities = torch.softmax(ensemble_outputs, dim=1)
        predictions = (probabilities[:, 1] > 0.5).int()

        return pd.DataFrame({
            'StockID': all_stock_ids,
            'ending_date': all_ending_dates,
            'label': all_labels_tensor.numpy(),
            'prediction': predictions.cpu().numpy(),
            'prob_down': probabilities[:, 0].cpu().numpy(),
            'prob_up': probabilities[:, 1].cpu().numpy(),
        })


class Evaluate:
    def __init__(
        self,
        mode: str = 'TEST',
        params: Optional[CNNParams] = None,
        pairs: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.mode = mode
        self.params = params or CNNParams()
        self.result_handler = GetResult(mode=mode)
        self.pairs = list(pairs) if pairs is not None else None

    def run_single(self, input_days: int, return_days: int) -> Optional[pd.DataFrame]:
        if input_days not in self.params.window_sizes:
            logger.warning(
                "Input window %d not in config. Skipping I%d/R%d.",
                input_days,
                input_days,
                return_days,
            )
            return

        logger.info("%s", "=" * 50)
        logger.info("EVALUATING MODEL: I%d/R%d", input_days, return_days)
        logger.info("Input: %d days, Return horizon: %d days", input_days, return_days)
        logger.info("Running in %s mode", self.mode)
        logger.info("%s", "=" * 50)

        config = self.params.get_config(self.mode, input_days)
        config.with_test_years(self.params.get_test_years())

        evaluator = ModelEvaluator(
            input_days=input_days,
            return_days=return_days,
            config=config,
        )
        results_df = evaluator.predict()

        if results_df is not None:
            model_name = f"I{input_days}/R{return_days}"
            self.result_handler.save_res(model_name, results_df)
        return results_df

    def run_all(self) -> Dict[Tuple[int, int], pd.DataFrame]:
        evaluation_pairs = self.pairs or self.params.get_evaluation_pairs(self.mode)
        results: Dict[Tuple[int, int], pd.DataFrame] = {}
        for input_days, return_days in evaluation_pairs:
            df = self.run_single(input_days, return_days)
            if df is not None:
                results[(input_days, return_days)] = df

        self.result_handler.print_summary()
        return results


def main(
        mode: str = 'TEST',
        pairs: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[Tuple[int, int], pd.DataFrame]:
    evaluator = Evaluate(mode=mode, pairs=pairs)
    return evaluator.run_all()


if __name__ == "__main__":
    res = main(mode='TEST',
               pairs=[(5, 5)])
