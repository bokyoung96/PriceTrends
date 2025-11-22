import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.device import DeviceSelector
from core.models.model1 import CNNModel
from core.models.model2 import FusionDataset, ForeignFeatureMLP, FusionModel
from core.params import CNNConfig, CNNParams
from prediction.evaluations.evaluate1 import BatchResultCollector, GetResult
from utils.root import MODELS_ROOT


logger = logging.getLogger(__name__)


class FusionModelEvaluator:
    def __init__(
        self,
        input_days: int,
        return_days: int,
        config: CNNConfig,
        models_root: Path = MODELS_ROOT,
    ) -> None:
        self.input_days = input_days
        self.return_days = return_days
        self.config = config

        selector = DeviceSelector()
        self.device = selector.resolve()
        self.model_name = f"I{input_days}/R{return_days}"
        self.exp_name = f"korea_cnn_{input_days}d{return_days}p_{config.mode}_fusion"
        self.model_dir = Path(models_root) / self.exp_name
        logger.info(selector.summary(self.model_name))

    def get_dataloader(self) -> Optional[DataLoader]:
        if not self.config.test_years:
            logger.warning("No test years configured for %s.", self.model_name)
            return None
        try:
            dataset = FusionDataset(
                ws=self.input_days,
                years=self.config.test_years,
                foreign_windows=(5, 20, 60),
                norm_years=self.config.train_years,
            )
        except FileNotFoundError:
            logger.warning(
                "No fusion data found for years %s. Skipping %s.",
                self.config.test_years,
                self.model_name,
            )
            return None

        # Log image vs fusion-matched sample counts for diagnostics
        try:
            image_samples = len(dataset.image_dataset.metadata)
            fusion_samples = len(dataset.merged)
            logger.info(
                "Image samples: %d, Fusion matched samples: %d",
                image_samples,
                fusion_samples,
            )
        except Exception:
            # Best-effort logging; do not break evaluation if attributes change
            pass

        if len(dataset) == 0:
            logger.warning("No fusion data available for %s. Skipping.", self.model_name)
            return None

        pin_memory = self.device.type == "cuda"
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )
        logger.info("Fusion test size for %s: %d", self.model_name, len(dataset))
        return loader

    def _build_model(self, dataset: FusionDataset) -> FusionModel:
        feature_dim = dataset.feature_dim

        paddings = [(int(fs[0] / 2), int(fs[1] / 2)) for fs in self.config.filter_sizes]

        cnn_encoder = CNNModel(
            layer_number=len(self.config.conv_channels),
            input_size=(dataset.image_dataset.image_height, dataset.image_dataset.image_width),
            inplanes=self.config.conv_channels[0],
            conv_layer_chanls=self.config.conv_channels,
            drop_prob=self.config.drop_prob,
            filter_size_list=self.config.filter_sizes,
            stride_list=[(1, 1)] * len(self.config.conv_channels),
            padding_list=paddings,
            dilation_list=[(1, 1)] * len(self.config.conv_channels),
            max_pooling_list=[(2, 1)] * len(self.config.conv_channels),
        )

        foreign_encoder = ForeignFeatureMLP(
            input_dim=feature_dim,
            embedding_dim=128,
            hidden_dim=128,
        )

        fusion_model = FusionModel(
            image_encoder=cnn_encoder,
            foreign_encoder=foreign_encoder,
            fusion_hidden_dims=(256, 128),
            dropout=self.config.drop_prob,
        ).to(self.device)

        return fusion_model

    def _resolve_model_paths(self) -> List[Path]:
        """Resolve available ensemble checkpoints with graceful fallback."""
        paths: List[Path] = []

        for model_num in range(self.config.ensem_size):
            candidate = self.model_dir / f"fusion_checkpoint{model_num}.pth.tar"
            if candidate.exists():
                paths.append(candidate)

        if not paths:
            fallback = self.model_dir / "fusion_checkpoint.pth.tar"
            if fallback.exists():
                logger.info(
                    "No numbered fusion checkpoints found for %s; using fallback %s",
                    self.model_name,
                    fallback,
                )
                paths.append(fallback)

        return paths

    def predict(self) -> Optional[pd.DataFrame]:
        loader = self.get_dataloader()
        if loader is None:
            return None

        dataset = loader.dataset
        if not isinstance(dataset, FusionDataset):
            logger.error("Expected FusionDataset for evaluation, got %s", type(dataset))
            return None

        model_paths = self._resolve_model_paths()
        if not model_paths:
            logger.warning("No fusion model checkpoints found in %s. Skipping.", self.model_dir)
            return None

        collector = BatchResultCollector()
        any_metadata_recorded = False

        for model_idx, model_path in enumerate(model_paths):
            logger.info(
                "Evaluating fusion ensemble member %d/%d for %s (%s)",
                model_idx + 1,
                len(model_paths),
                self.model_name,
                model_path.name,
            )

            model = self._build_model(dataset)
            checkpoint = torch.load(str(model_path), map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            model_outputs: List[torch.Tensor] = []

            with torch.no_grad():
                for batch in tqdm(
                    loader,
                    desc=f"Predicting Fusion {self.model_name} [{model_idx + 1}/{len(model_paths)}]",
                ):
                    images = batch["image"].to(self.device)
                    foreign_features = batch["foreign_features"].to(self.device)
                    outputs = model(images, foreign_features)
                    model_outputs.append(outputs.cpu())

                    if not any_metadata_recorded:
                        collector.record_metadata(batch)

            collector.add_model_outputs(model_outputs)
            any_metadata_recorded = True

        finalized = collector.finalize()
        if finalized is None:
            logger.warning("No outputs collected for fusion %s.", self.model_name)
            return None

        ensemble_outputs, all_labels_tensor, all_stock_ids, all_ending_dates = finalized
        probabilities = torch.softmax(ensemble_outputs, dim=1)
        predictions = (probabilities[:, 1] > 0.5).int()

        return pd.DataFrame(
            {
                "StockID": all_stock_ids,
                "ending_date": all_ending_dates,
                "label": all_labels_tensor.numpy(),
                "prediction": predictions.cpu().numpy(),
                "prob_down": probabilities[:, 0].cpu().numpy(),
                "prob_up": probabilities[:, 1].cpu().numpy(),
            }
        )


class EvaluateFusion:
    def __init__(
        self,
        mode: str = "TEST",
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
                "Input window %d not in config. Skipping Fusion I%d/R%d.",
                input_days,
                input_days,
                return_days,
            )
            return None

        logger.info("%s", "=" * 50)
        logger.info("EVALUATING FUSION MODEL: I%d/R%d", input_days, return_days)
        logger.info("Input: %d days, Return horizon: %d days", input_days, return_days)
        logger.info("Running in %s mode", self.mode)
        logger.info("%s", "=" * 50)

        config = self.params.get_config(self.mode, input_days)
        config.with_test_years(self.params.get_test_years())

        evaluator = FusionModelEvaluator(
            input_days=input_days,
            return_days=return_days,
            config=config,
        )

        results_df = evaluator.predict()

        if results_df is not None:
            model_name = f"I{input_days}/R{return_days}_fusion"
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
    mode: str = "TEST",
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], pd.DataFrame]:
    evaluator = EvaluateFusion(mode=mode, pairs=pairs)
    return evaluator.run_all()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    res = main(mode="TEST")
