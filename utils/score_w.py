print("Script start...")
import os
import pandas as pd
from functools import reduce
from sklearn.metrics import precision_score

from prediction.evaluate import ModelEvaluator
from core.params import CNNParams


class WeightedResultLoader:
    def __init__(self, validation_year: int, run_mode: str = 'TEST'):
        self.params = CNNParams()
        self.configs = [
            (5, 5),
            (20, 20),
            (60, 60),
        ]
        self.validation_year = validation_year
        self.test_years = self.params.get_test_years()
        self.run_mode = run_mode
        self.weights = {}
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def calculate_weights(self) -> None:
        print("\n" + "="*60)
        print(
            f"STEP 1: CALCULATING WEIGHTS USING VALIDATION YEAR: {self.validation_year}")
        print("="*60)

        precisions = {}
        for input_days, return_days in self.configs:
            model_name = f"I{input_days}/R{return_days}"
            print(f"\n--- Evaluating {model_name} on validation data ---")

            config = self.params.get_config(self.run_mode, input_days)
            config['test_years'] = [self.validation_year]

            evaluator = ModelEvaluator(
                input_days=input_days, return_days=return_days, config=config)

            results_df = evaluator.predict()

            if results_df is None or results_df.empty:
                print(
                    f"Could not generate predictions for {model_name}. Assigning precision of 0.")
                precisions[model_name] = 0
                continue

            precision = precision_score(
                results_df['label'],
                results_df['prediction'],
                pos_label=0,
                zero_division=0
            )
            precisions[model_name] = precision
            print(
                f"'{model_name}' Down-prediction Precision on {self.validation_year} data: {precision:.4f}")

        total_precision = sum(precisions.values())
        if total_precision == 0:
            print("\nWarning: All models have 0 precision. Using equal weights.")
            num_models = len(self.configs)
            self.weights = {model_name: 1 /
                            num_models for model_name in precisions.keys()}
        else:
            self.weights = {
                model_name: p / total_precision for model_name, p in precisions.items()}

        print("\n" + "="*60)
        print("CALCULATED ENSEMBLE WEIGHTS")
        for model_name, weight in self.weights.items():
            print(f"{model_name}: {weight:.4f}")
        print("="*60)

    def score_test_data(self) -> None:
        if not self.weights:
            print("Weights not calculated. Aborting.")
            return

        print("\n" + "="*60)
        print(f"STEP 2: SCORING TEST DATA FOR YEARS: {self.test_years}")
        print("="*60)

        model_predictions = []
        for input_days, return_days in self.configs:
            model_name = f"I{input_days}/R{return_days}"
            print(f"\n--- Predicting for {model_name} on test data ---")

            config = self.params.get_config(self.run_mode, input_days)
            config['test_years'] = self.test_years

            evaluator = ModelEvaluator(
                input_days=input_days, return_days=return_days, config=config)

            results_df = evaluator.predict()

            if results_df is None or results_df.empty:
                print(
                    f"Could not generate predictions for {model_name} on test set. Skipping.")
                continue

            df_renamed = results_df[['StockID', 'ending_date', 'label', 'prob_up']].rename(
                columns={'prob_up': f'prob_up_{model_name.replace("/", "_")}'})
            model_predictions.append(df_renamed)

        if not model_predictions:
            print("No predictions were generated. Aborting.")
            return

        print("\nMerging model predictions...")
        final_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=['StockID', 'ending_date', 'label'], how='inner'),
            model_predictions
        )
        print(
            f"Merged {len(final_df)} common data points across all models.")

        final_df['weighted_prob_up'] = 0.0
        for model_name, weight in self.weights.items():
            col_name = f'prob_up_{model_name.replace("/", "_")}'
            final_df['weighted_prob_up'] += final_df[col_name] * weight

        self._save_and_summarize_results(final_df)

    def _save_and_summarize_results(self, final_df: pd.DataFrame) -> None:
        output_path = os.path.join(
            self.results_dir, 'test_results_weighted.parquet')
        final_df.to_parquet(output_path, index=False)

        print("\n" + "="*60)
        print("STEP 3: FINAL SCORING COMPLETE")
        print(
            f"Results with weighted probability scores saved to {output_path}")
        print("="*60)

        print("\nFinal Results Head:")
        print(final_df.head())

    def run(self) -> None:
        self.calculate_weights()
        self.score_test_data()


def main():
    VALIDATION_YEAR = 2011
    RUN_MODE = 'TEST'
    scorer = WeightedResultLoader(validation_year=VALIDATION_YEAR,
                                 run_mode=RUN_MODE)
    scorer.run()


if __name__ == "__main__":
    main() 