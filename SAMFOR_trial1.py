from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

from config import SAMFOR_SAMFOR_PARAMS
from lssvr import LSSVR
from preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    compute_metrics,
    get_SAMFOR_data,
    inverse_transf,
    log_results,
    persist_model_results,
    plot_test,
    save_object,
    time_call,
)


def _train_model(use_lssvr: bool, X_train, y_train, lssvr_params, svr_params):
    if use_lssvr:
        model = LSSVR(**lssvr_params)
        name = "SAMFOR_SARIMA_LSSVR"
    else:
        model = SVR(**svr_params)
        name = "SAMFOR"
    print(f"Training {name}...")
    _, elapsed = time_call(model.fit, X_train, np.squeeze(y_train))
    return model, name, elapsed / 60


def _evaluate_model(model, X_test, y_test_scaled, scaler):
    """Evaluate model and return predictions with metrics."""
    y_pred_scaled, test_elapsed = time_call(model.predict, X_test)
    y_pred = inverse_transf(np.squeeze(y_pred_scaled).reshape(-1, 1), scaler)
    y_true = np.squeeze(inverse_transf(y_test_scaled, scaler))
    rmse, mae, mape = compute_metrics(y_true, y_pred)
    print(f"rmse: {rmse:.4f} || mape: {mape:.2f} || mae: {mae:.4f}")
    return y_true, y_pred, rmse, mae, mape, test_elapsed


def main():
    params = SAMFOR_SAMFOR_PARAMS
    option = params["option"]
    datatype_opt = params["datatype"]
    seq_length = params["sequence_length"]

    (
        X_train,
        y_train,
        X_test,
        y_test,
        save_path_str,
        test_time_axis,
        scaler,
    ) = get_SAMFOR_data(option, datatype_opt, seq_length)

    print(X_train.shape, X_test.shape)
    save_path = Path(save_path_str)

    use_lssvr = any(name.upper().startswith("SAMFOR_SARIMA") for name in params.get("algorithms", ("SAMFOR",)))

    model, alg_name, train_time = _train_model(
        use_lssvr,
        X_train,
        y_train,
        params["lssvr_params"],
        params["svr_params"],
    )

    y_true, y_pred, rmse, mae, mape, test_time = _evaluate_model(model, X_test, y_test, scaler)
    persist_model_results(
        alg_name,
        save_path,
        y_true,
        y_pred,
        rmse,
        mae,
        mape,
        seq_length,
        train_time,
        test_time,
        params["datatype"],
        params.get("plot_results", False),
        params.get("persist_models", True),
        params.get("persist_models", []),
    )


if __name__ == "__main__":
    main()