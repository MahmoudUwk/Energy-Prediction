from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from config import (
    SAMFOR_DATATYPE,
    SAMFOR_MODELS,
    SAMFOR_OPTION,
    SAMFOR_PERSIST_MODELS,
    SAMFOR_RANDOM_FOREST_PARAMS,
    SAMFOR_SAVE_RESULTS,
    SAMFOR_SEQUENCE_LENGTH,
    SAMFOR_SVR_PARAMS,
)
from preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    get_SAMFOR_data,
    inverse_transf,
    log_results,
    save_object,
)


def _time_call(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return RMSE(y_true, y_pred), MAE(y_true, y_pred), MAPE(y_true, y_pred)


def _persist_predictions(base_path: Path, name: str, y_true: np.ndarray, y_pred: np.ndarray):
    filename = base_path / f"{name}.obj"
    save_object({"y_test": y_true, "y_test_pred": y_pred}, filename)


def _train_and_evaluate(
    model,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    scaler,
    save_path: Path,
    datatype_opt: str,
    seq_length: int,
):
    print(f"Training {name}...")
    _, train_elapsed = _time_call(model.fit, X_train, y_train)
    y_pred_scaled, test_elapsed = _time_call(model.predict, X_test)

    y_true = inverse_transf(y_test_scaled, scaler)
    y_pred = inverse_transf(y_pred_scaled, scaler)

    rmse, mae, mape = _compute_metrics(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    row = [name, rmse, mae, mape, seq_length, train_elapsed / 60, test_elapsed]
    if SAMFOR_SAVE_RESULTS:
        log_results(row, datatype_opt, str(save_path))
    if name in SAMFOR_PERSIST_MODELS:
        _persist_predictions(save_path, name, y_true, y_pred)


def _available_models():
    return {
        "RFR": lambda: RandomForestRegressor(**SAMFOR_RANDOM_FOREST_PARAMS),
        "SVR": lambda: SVR(**SAMFOR_SVR_PARAMS),
    }


def main():
    seq = SAMFOR_SEQUENCE_LENGTH
    datatype_opt = SAMFOR_DATATYPE
    option = SAMFOR_OPTION

    X_train, y_train, X_test, y_test, save_path_str, _, scaler = get_SAMFOR_data(
        option, datatype_opt, seq
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    save_path = Path(save_path_str)
    models = _available_models()

    for name in SAMFOR_MODELS:
        if name not in models:
            print(f"Skipping unsupported model '{name}'.")
            continue
        model = models[name]()
        _train_and_evaluate(
            model,
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            save_path,
            datatype_opt,
            seq,
        )


if __name__ == "__main__":
    main()