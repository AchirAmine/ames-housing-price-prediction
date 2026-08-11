from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TRAIN_FILE = DATA_DIR / "train.csv"
TARGET_COL = "SalePrice"
RANDOM_SEED = 42
N_FOLDS = 5
