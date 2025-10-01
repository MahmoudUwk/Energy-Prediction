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


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _parse_list(env_name: str, default, cast=str):
    value = os.getenv(env_name)
    if value is None:
        return tuple(default)

    parsed = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            parsed.append(cast(item))
        except ValueError:
            continue

    return tuple(parsed) if parsed else tuple(default)


SAMFOR_OPTION = int(os.getenv("ENERGY_SAMFOR_OPTION", 2))
SAMFOR_DATATYPE = os.getenv("ENERGY_SAMFOR_DATATYPE", "ele")
SAMFOR_SEQUENCE_LENGTH = int(os.getenv("ENERGY_SAMFOR_SEQUENCE", 7))

SAMFOR_MODELS = tuple(
    name.strip()
    for name in os.getenv("ENERGY_SAMFOR_MODELS", "RFR,SVR").split(",")
    if name.strip()
)

SAMFOR_RANDOM_FOREST_PARAMS = {
    "random_state": int(os.getenv("ENERGY_RFR_RANDOM_STATE", 0)),
    "n_estimators": int(os.getenv("ENERGY_RFR_N_ESTIMATORS", 100)),
}

SAMFOR_SVR_PARAMS = {
    "C": float(os.getenv("ENERGY_SVR_C", 10)),
    "epsilon": float(os.getenv("ENERGY_SVR_EPSILON", 0.01)),
    "kernel": os.getenv("ENERGY_SVR_KERNEL", "rbf"),
    "gamma": os.getenv("ENERGY_SVR_GAMMA", "scale"),
}

SAMFOR_PERSIST_MODELS = frozenset(
    name.strip()
    for name in os.getenv("ENERGY_SAMFOR_PERSIST_MODELS", "SVR").split(",")
    if name.strip()
)

SAMFOR_SAVE_RESULTS = _env_flag("ENERGY_SAMFOR_SAVE_RESULTS", True)


SAMFOR_SAMFOR_PARAMS = {
    "option": SAMFOR_OPTION,
    "datatype": SAMFOR_DATATYPE,
    "sequence_length": SAMFOR_SEQUENCE_LENGTH,
    "algorithms": _parse_list("ENERGY_SAMFOR_ALGORITHMS", ("SAMFOR",), str),
    "lssvr_params": {
        "C": float(os.getenv("ENERGY_LSSVR_C", 1.0)),
        "gamma": float(os.getenv("ENERGY_LSSVR_GAMMA", 0.001)),
        "kernel": os.getenv("ENERGY_LSSVR_KERNEL", "rbf"),
    },
    "svr_params": {
        "C": float(os.getenv("ENERGY_SAMFOR_SVR_C", 1.0)),
        "epsilon": float(os.getenv("ENERGY_SAMFOR_SVR_EPSILON", 0.001)),
        "kernel": os.getenv("ENERGY_SAMFOR_SVR_KERNEL", "rbf"),
        "gamma": os.getenv("ENERGY_SAMFOR_SVR_GAMMA", "scale"),
    },
    "plot_results": _env_flag("ENERGY_SAMFOR_PLOT", True),
    "persist_models": frozenset(
        name.strip()
        for name in os.getenv("ENERGY_SAMFOR_PERSIST", "SAMFOR").split(",")
        if name.strip()
    ),
}


LSTM_TRAINING_CONFIG = {
    "option": int(os.getenv("ENERGY_LSTM_OPTION", 3)),
    "algorithm": os.getenv("ENERGY_LSTM_ALGORITHM", "LSTM"),
    "data_types": _parse_list("ENERGY_LSTM_DATA_TYPES", ("Home",), str),
    "sequence_lengths": _parse_list("ENERGY_LSTM_SEQUENCE_LENGTHS", (23,), int),
    "units": _parse_list("ENERGY_LSTM_UNITS", (72,), int),
    "num_layers": _parse_list("ENERGY_LSTM_NUM_LAYERS", (1,), int),
    "num_features": _parse_list("ENERGY_LSTM_NUM_FEATURES", (1,), int),
    "epochs": int(os.getenv("ENERGY_LSTM_EPOCHS", 2000)),
    "learning_rate": float(os.getenv("ENERGY_LSTM_LEARNING_RATE", 1e-4)),
    "batch_size_power": int(os.getenv("ENERGY_LSTM_BATCH_SIZE_POWER", 11)),
    "verbose": int(os.getenv("ENERGY_LSTM_VERBOSE", 2)),
    "validation_split": float(os.getenv("ENERGY_LSTM_VALIDATION_SPLIT", 0.0)),
    "patience": int(os.getenv("ENERGY_LSTM_PATIENCE", 50)),
    "use_callbacks": _env_flag("ENERGY_LSTM_USE_CALLBACKS", True),
    "plot_model": _env_flag("ENERGY_LSTM_PLOT_MODEL", False),
    "model_plot_filename": os.getenv("ENERGY_LSTM_MODEL_PLOT", "model_1.png"),
    "persist_models": _env_flag("ENERGY_LSTM_PERSIST", True),
    "save_results": _env_flag("ENERGY_LSTM_SAVE_RESULTS", True),
}


LSTM_SEARCH_CONFIG = {
    "option": int(os.getenv("ENERGY_LSTM_SEARCH_OPTION", 3)),
    "datatype_options": _parse_list("ENERGY_LSTM_SEARCH_DATA_TYPES", ("5T", "Home"), str),
    "run_search": _env_flag("ENERGY_LSTM_SEARCH_RUN", True),
    "population_size": int(os.getenv("ENERGY_LSTM_SEARCH_POPULATION", 5)),
    "num_epochs": int(os.getenv("ENERGY_LSTM_SEARCH_EPOCHS", 2500)),
    "iterations": int(os.getenv("ENERGY_LSTM_SEARCH_ITERATIONS", 15)),
    "algorithms": _parse_list(
        "ENERGY_LSTM_SEARCH_ALGORITHMS", ("Mod_FireflyAlgorithm", "FireflyAlgorithm"), str
    ),
    "plot_convergence": _env_flag("ENERGY_LSTM_SEARCH_PLOT", True),
    "batch_size_power": int(os.getenv("ENERGY_LSTM_SEARCH_BATCH_POWER", 12)),
    "patience": int(os.getenv("ENERGY_LSTM_SEARCH_PATIENCE", 30)),
    "persist_models": _env_flag("ENERGY_LSTM_SEARCH_PERSIST", True),
}


def train_split_for(datatype_opt: str) -> float:
    """Return training split fraction for the given datatype."""
    return DATA_SPLITS.get(datatype_opt, DATA_SPLITS["default"])


def lstm_val_fraction() -> float:
    """Fraction of the dataset allocated to validation for LSTM training."""
    return LSTM_TOTAL_PORTION * LSTM_VAL_RATIO_WITHIN_TRAIN


def lstm_train_fraction() -> float:
    """Fraction of the dataset allocated to LSTM training after validation split."""
    return LSTM_TOTAL_PORTION - lstm_val_fraction()
