import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from functools import cached_property
import matplotlib.pyplot as plt
import seaborn as sns


class ResultLoader:
    def __init__(self, results_dir: str = 'results/results_production'):
        self.results_dir = Path(__file__).parent / results_dir

    def raw_results(self, i: int, r: int) -> pd.DataFrame:
        file_path = self.results_dir / f"test_results_i{i}_r{r}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return pd.read_parquet(file_path)

    def load_results(self, i: int, r: int) -> Dict[str, pd.DataFrame]:
        df = self.raw_results(i, r).reset_index(drop=False)
        df['ending_date'] = pd.to_datetime(
            df['ending_date'].astype(str), format='%Y%m%d')

        pivot_dict = {}
        for col in ['label', 'prediction', 'prob_down', 'prob_up']:
            pivot_df = df.pivot(index='ending_date',
                                columns='StockID', values=col)
            pivot_dict[col] = pivot_df
        return pivot_dict

    def available_models(self) -> List[Tuple[int, int]]:
        models = []
        for file_path in self.results_dir.glob("test_results_i*_r*.parquet"):
            name = file_path.stem
            parts = name.replace("test_results_", "").split("_")
            i = int(parts[0][1:])
            r = int(parts[1][1:])
            models.append((i, r))
        return sorted(models)

    @cached_property
    def avg_prob(self) -> pd.DataFrame:
        prob_ups = []
        models = self.available_models()
        for i, r in tqdm(models, desc="Loading models"):
            try:
                data = self.load_results(i, r)
                prob_ups.append(data['prob_up'])
            except FileNotFoundError:
                print(f"Model I{i}/R{r} not found, skipping...")
                continue

        if len(prob_ups) < 2:
            raise ValueError("Need at least 2 models for average")

        common_dates = prob_ups[0].index
        common_ids = prob_ups[0].columns

        for prob_up in prob_ups[1:]:
            common_dates = common_dates.intersection(prob_up.index)
            common_ids = common_ids.intersection(prob_up.columns)

        print(
            f"Common dates: {len(common_dates)}, Common stocks: {len(common_ids)}")

        aligned_probs = []
        for prob_up in prob_ups:
            aligned = prob_up.loc[common_dates, common_ids]
            aligned_probs.append(aligned)

        avgs = sum(aligned_probs) / len(aligned_probs)
        return avgs

    def __call__(self, i: int, r: int = None) -> Dict[str, pd.DataFrame]:
        if r is None:
            r = i
        return self.load_results(i, r)

    @cached_property
    def prob_up_5d(self) -> pd.DataFrame:
        return self(5)['prob_up']

    @cached_property
    def prob_up_20d(self) -> pd.DataFrame:
        return self(20)['prob_up']

    @cached_property
    def prob_up_60d(self) -> pd.DataFrame:
        return self(60)['prob_up']

    def analyze_and_save_results(self, plot: bool = True):
        dir_name = self.results_dir.name
        env_suffix = dir_name.replace('results_', '')

        model_is = [5, 20, 60]
        model_props = {
            f'Model i={i}': getattr(self, f'prob_up_{i}d') for i in model_is
        }

        output_dir = self.results_dir.parent

        for i in model_is:
            try:
                prob_df = getattr(self, f'prob_up_{i}d')
                output_path = output_dir / f"price_trends_avg_{env_suffix}_{i}.parquet"
                prob_df.to_parquet(output_path)
                print(f"Saved: {output_path}")
            except FileNotFoundError:
                print(f"Model i={i} not found, skipping save.")
            except Exception as e:
                print(f"Error saving model i={i}: {e}")

        try:
            avg_prob_df = self.avg_prob
            output_path = output_dir / f"price_trends_avg_{env_suffix}_ensemble.parquet"
            avg_prob_df.to_parquet(output_path)
            print(f"Saved ensemble average: {output_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"Could not save ensemble average: {e}")

        if not plot:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        titles = list(model_props.keys()) + ['Ensemble Average']

        for idx, (title, prob_df_property) in enumerate(model_props.items()):
            try:
                prob_up = prob_df_property.values.flatten()
                prob_up = prob_up[~pd.isna(prob_up)]

                axes[idx].hist(prob_up, bins=50, alpha=0.7, edgecolor='black')
                axes[idx].set_title(title)
                axes[idx].set_xlabel('Prob Up')
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
                print(f"{title} - Mean: {prob_up.mean():.4f}, Std: {prob_up.std():.4f}")

            except FileNotFoundError:
                print(f"{title} not found")
                axes[idx].text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center', transform=axes[idx].transAxes)

        try:
            avg_prob = self.avg_prob.values.flatten()
            avg_prob = avg_prob[~pd.isna(avg_prob)]

            axes[3].hist(avg_prob, bins=50, alpha=0.7, edgecolor='black', color='red')
            axes[3].set_title(titles[3])
            axes[3].set_xlabel('Prob Up')
            axes[3].set_ylabel('Frequency')
            axes[3].grid(True, alpha=0.3)
            print(f"{titles[3]} - Mean: {avg_prob.mean():.4f}, Std: {avg_prob.std():.4f}")

        except (ValueError, FileNotFoundError):
            print("Ensemble average not available for plotting.")
            axes[3].text(0.5, 0.5, 'Ensemble Average\nNot Found', ha='center', va='center', transform=axes[3].transAxes)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("--- Processing Test Results ---")
    test_loader = ResultLoader(results_dir='results/results_test')
    test_loader.analyze_and_save_results(plot=True)

    print("\n--- Processing Production Results ---")
    prod_loader = ResultLoader(results_dir='results/results_production')
    prod_loader.analyze_and_save_results(plot=False)