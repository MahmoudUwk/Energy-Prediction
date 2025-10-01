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
    get_SAMFOR_data,
    inverse_transf,
    log_results,
    plot_test,
    save_object,
)


def _time_call(fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def _train_model(use_lssvr: bool, X_train, y_train, lssvr_params, svr_params):
    if use_lssvr:
        model = LSSVR(**lssvr_params)
        name = "SAMFOR_SARIMA_LSSVR"
    else:
        model = SVR(**svr_params)
        name = "SAMFOR"
    print(f"Training {name}...")
    _, elapsed = _time_call(model.fit, X_train, np.squeeze(y_train))
    return model, name, elapsed / 60


def _evaluate(model, X_test, y_test_scaled, scaler):
    y_pred_scaled, test_elapsed = _time_call(model.predict, X_test)
    y_pred = inverse_transf(np.squeeze(y_pred_scaled).reshape(-1, 1), scaler)
    y_true = np.squeeze(inverse_transf(y_test_scaled, scaler))
    rmse = RMSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    print(f"rmse: {rmse:.4f} || mape: {mape:.2f} || mae: {mae:.4f}")
    return y_true, y_pred, rmse, mae, mape, test_elapsed


def _persist_results(
    name: str,
    save_path: Path,
    y_true,
    y_pred,
    rmse,
    mae,
    mape,
    seq,
    train_time,
    test_time,
    persist_model: bool,
):
    row = [name, rmse, mae, mape, seq, train_time, test_time]
    log_results(row, SAMFOR_SAMFOR_PARAMS["datatype"], str(save_path))
    if SAMFOR_SAMFOR_PARAMS["plot_results"]:
        name_sav = save_path / f"{name}_datatype_opt{SAMFOR_SAMFOR_PARAMS['datatype']}.png"
        plot_test(None, y_true, y_pred, str(name_sav), name)
    if persist_model and name in SAMFOR_SAMFOR_PARAMS["persist_models"]:
        filename = save_path / f"{name}.obj"
        save_object({"y_test": y_true, "y_test_pred": y_pred}, filename)


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

    y_true, y_pred, rmse, mae, mape, test_time = _evaluate(model, X_test, y_test, scaler)
    _persist_results(
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
        params.get("persist_models", True),
    )


if __name__ == "__main__":
    main()