from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "DATA"
IMAGES_ROOT = PROJECT_ROOT / "Images"
MODELS_ROOT = PROJECT_ROOT / "models"
RESULTS_ROOT = PROJECT_ROOT / "results"
SCORES_ROOT = PROJECT_ROOT / "scores"

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "IMAGES_ROOT",
    "MODELS_ROOT",
    "RESULTS_ROOT",
    "SCORES_ROOT",
]
