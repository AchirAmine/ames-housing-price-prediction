from pathlib import Path
import pandas as pd
import numpy as np
from src.ames.config import TRAIN_FILE, TARGET_COL

def load_raw_data(path: Path = TRAIN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)

def remove_known_outliers(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["GrLivArea"] > 4000) & (df[TARGET_COL] < 300000)
    return df[~mask].copy()

def get_feature_target_split(df: pd.DataFrame, target_col: str = TARGET_COL):
    X = df.drop(columns=[target_col, "Id"], errors="ignore")
    y = np.log1p(df[target_col])
    return X, y
