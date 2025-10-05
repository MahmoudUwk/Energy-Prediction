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
from tools.preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    compute_metrics,
    get_SAMFOR_data,
    inverse_transf,
    log_results,
    persist_model_results,
    save_object,
    time_call,
)


# Utility functions moved to preprocess_data2.py


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
    _, train_elapsed = time_call(model.fit, X_train, y_train)
    y_pred_scaled, test_elapsed = time_call(model.predict, X_test)

    y_true = inverse_transf(y_test_scaled, scaler)
    y_pred = inverse_transf(y_pred_scaled, scaler)

    rmse, mae, mape = compute_metrics(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

    persist_model_results(
        name,
        save_path,
        y_true,
        y_pred,
        rmse,
        mae,
        mape,
        seq_length,
        train_elapsed / 60,
        test_elapsed,
        datatype_opt,
        plot_results=False,  # SVR script doesn't seem to use plots
        persist_model=True,
        persist_models_list=SAMFOR_PERSIST_MODELS,
    )


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