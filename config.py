from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = Path(os.getenv("ENERGY_RESULTS_ROOT", BASE_DIR / "results"))
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

RESAMPLED_DATA_ROOT = Path(
    os.getenv("ENERGY_RESAMPLED_DATA_ROOT", BASE_DIR / "data" / "resampled")
)
HOME_DATA_ROOT = Path(os.getenv("ENERGY_HOME_DATA_ROOT", BASE_DIR / "data" / "home"))
ELE_DATA_ROOT = Path(os.getenv("ENERGY_ELE_DATA_ROOT", BASE_DIR / "data" / "ele"))

DATA_SPLITS = {
    "Home": float(os.getenv("ENERGY_TRAIN_SPLIT_HOME", 0.5)),
    "default": float(os.getenv("ENERGY_TRAIN_SPLIT_DEFAULT", 0.8)),
}

SARIMA_LOOKBACK = int(os.getenv("ENERGY_SARIMA_LOOKBACK", 60 * 12))
MINMAX_FEATURE_RANGE = (
    float(os.getenv("ENERGY_SCALER_MIN", 0.0)),
    float(os.getenv("ENERGY_SCALER_MAX", 1.0)),
)

LSTM_TOTAL_PORTION = float(os.getenv("ENERGY_LSTM_TOTAL_PORTION", 0.8))
LSTM_VAL_RATIO_WITHIN_TRAIN = float(os.getenv("ENERGY_LSTM_VAL_RATIO", 0.3))

DEFAULT_RESULTS_DATASET = os.getenv("ENERGY_RESULTS_DATASET", "1s")
EXPECTED_RESULT_FILES = [
    "SVR.obj",
    "RFR.obj",
    "SAMFOR.obj",
    "LSTM.obj",
    "FireflyAlgorithm.obj",
    "Mod_FireflyAlgorithm.obj",
]


def train_split_for(datatype_opt: str) -> float:
    """Return training split fraction for the given datatype."""
    return DATA_SPLITS.get(datatype_opt, DATA_SPLITS["default"])


def lstm_val_fraction() -> float:
    """Fraction of the dataset allocated to validation for LSTM training."""
    return LSTM_TOTAL_PORTION * LSTM_VAL_RATIO_WITHIN_TRAIN


def lstm_train_fraction() -> float:
    """Fraction of the dataset allocated to LSTM training after validation split."""
    return LSTM_TOTAL_PORTION - lstm_val_fraction()
