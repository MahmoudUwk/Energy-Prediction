from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULTS_ROOT = BASE_DIR / "results"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

RESAMPLED_DATA_ROOT = BASE_DIR / "dataset" / "resampled data"

DATA_SPLITS = {
    "default": 0.8,
}

SARIMA_LOOKBACK = 60 * 12
MINMAX_FEATURE_RANGE = (0.0, 1.0)

LSTM_TOTAL_PORTION = 0.8
LSTM_VAL_RATIO_WITHIN_TRAIN = 0.3

DEFAULT_RESULTS_DATASET = "1s"
EXPECTED_RESULT_FILES = [
    "SVR.obj",
    "RFR.obj",
    "SAMFOR.obj",
    "LSTM.obj",
    "FireflyAlgorithm.obj",
    "Mod_FireflyAlgorithm.obj",
]


SAMFOR_OPTION = 2
SAMFOR_DATATYPE = DEFAULT_RESULTS_DATASET
SAMFOR_SEQUENCE_LENGTH = 7

SAMFOR_MODELS = ("RFR", "SVR")

SAMFOR_RANDOM_FOREST_PARAMS = {
    "random_state": 0,
    "n_estimators": 100,
}

SAMFOR_SVR_PARAMS = {
    "C": 10.0,
    "epsilon": 0.01,
    "kernel": "rbf",
    "gamma": "scale",
}

SAMFOR_PERSIST_MODELS = frozenset({"SVR"})
SAMFOR_SAVE_RESULTS = True

SAMFOR_SAMFOR_PARAMS = {
    "option": SAMFOR_OPTION,
    "datatype": SAMFOR_DATATYPE,
    "sequence_length": SAMFOR_SEQUENCE_LENGTH,
    "algorithms": ("SAMFOR",),
    "lssvr_params": {
        "C": 1.0,
        "gamma": 0.001,
        "kernel": "rbf",
    },
    "svr_params": SAMFOR_SVR_PARAMS,
    "plot_results": True,
    "persist_models": frozenset({"SAMFOR"}),
}


LSTM_TRAINING_CONFIG = {
    "option": 3,
    "algorithm": "LSTM",
    "data_types": (DEFAULT_RESULTS_DATASET,),
    "sequence_lengths": (23,),
    "units": (72,),
    "num_layers": (1,),
    "num_features": (1,),
    "epochs": 2000,
    "learning_rate": 1e-4,
    "batch_size_power": 11,
    "verbose": 2,
    "validation_split": 0.0,
    "patience": 50,
    "use_callbacks": True,
    "plot_model": False,
    "model_plot_filename": "model_1.png",
    "persist_models": True,
    "save_results": True,
}


LSTM_SEARCH_CONFIG = {
    "option": 3,
    "datatype_options": (DEFAULT_RESULTS_DATASET,),
    "run_search": True,
    "population_size": 5,
    "num_epochs": 2500,
    "iterations": 10,
    "algorithms": ("Mod_FireflyAlgorithm", "FireflyAlgorithm"),
    "plot_convergence": True,
    "batch_size_power": 12,
    "patience": 20,
    "persist_models": True,
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
