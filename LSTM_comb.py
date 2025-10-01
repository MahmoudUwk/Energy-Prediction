from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import plot_model

from config import LSTM_TRAINING_CONFIG
from preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    expand_dims,
    get_SAMFOR_data,
    inverse_transf,
    log_results_LSTM,
    save_object,
)


def _ensure_three_dim(arr: np.ndarray | list | None) -> np.ndarray:
    if arr is None:
        return np.array([])
    if isinstance(arr, list):
        arr = np.array(arr)
    if getattr(arr, "size", 0) == 0:
        return np.array([])
    if arr.ndim < 3:
        arr = expand_dims(arr)
    return arr


def _prepare_targets(y: np.ndarray | list) -> np.ndarray:
    arr = np.array(y)
    if arr.size == 0:
        return np.array([])
    return expand_dims(expand_dims(arr))


def _subset_features(arr: np.ndarray, n_feat: int) -> np.ndarray:
    if arr.size == 0 or arr.ndim < 3 or n_feat <= 0:
        return arr
    max_feat = min(n_feat, arr.shape[2])
    return arr[:, :, :max_feat]


def _build_callbacks(cfg):
    if not cfg["use_callbacks"]:
        return None
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
        )
    ]


def _build_model(units, num_layers, input_dim, output_dim, learning_rate, plot_enabled, plot_filename):
    model = Sequential(name="LSTM_HP")
    return_sequences = num_layers > 1
    model.add(LSTM(units=units, input_shape=input_dim, return_sequences=return_sequences))
    for layer_idx in range(1, num_layers):
        model.add(LSTM(units=units, return_sequences=layer_idx < num_layers - 1))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    if plot_enabled:
        plot_model(model, to_file=plot_filename, show_shapes=True)
    return model


def _evaluate(model, X_test, y_test_scaled, scaler):
    y_true = inverse_transf(y_test_scaled, scaler)
    y_pred = inverse_transf(model.predict(X_test, verbose=0), scaler)
    rmse = RMSE(y_true, y_pred)
    mae = MAE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    return y_true, y_pred, rmse, mae, mape


def _best_epoch(history):
    val_loss = history.history.get("val_loss")
    if val_loss:
        return int(np.argmin(val_loss))
    return int(np.argmin(history.history["loss"]))


def _persist_results(
    save_path: Path,
    datatype_opt: str,
    alg_name: str,
    seq: int,
    num_layers: int,
    units: int,
    best_epoch: int,
    used_features: int,
    train_time: float,
    test_time: float,
    rmse: float,
    mae: float,
    mape: float,
    y_true,
    y_pred,
):
    if LSTM_TRAINING_CONFIG["save_results"]:
        row = [
            alg_name,
            rmse,
            mae,
            mape,
            seq,
            num_layers,
            units,
            best_epoch,
            datatype_opt,
            train_time,
            test_time,
            used_features,
        ]
        log_results_LSTM(row, datatype_opt, str(save_path))

    if LSTM_TRAINING_CONFIG["persist_models"]:
        filename = save_path / f"{alg_name}.obj"
        save_object({"y_test": y_true, "y_test_pred": y_pred}, filename)


def main():
    cfg = LSTM_TRAINING_CONFIG
    option = cfg["option"]
    alg_name = cfg["algorithm"]
    batch_size = 2 ** cfg["batch_size_power"]

    for datatype_opt in cfg["data_types"]:
        for n_feat in cfg["num_features"]:
            for seq in cfg["sequence_lengths"]:
                (
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    save_path_str,
                    _,
                    scaler,
                ) = get_SAMFOR_data(option, datatype_opt, seq)

                y_train = _prepare_targets(y_train)
                y_val = _prepare_targets(y_val) if len(y_val) else np.array([])
                X_train = _subset_features(_ensure_three_dim(X_train), n_feat)
                X_val = _subset_features(_ensure_three_dim(X_val), n_feat)
                X_test = _subset_features(_ensure_three_dim(X_test), n_feat)

                used_features = X_train.shape[2] if X_train.size else 0
                input_dim = (X_train.shape[1], used_features)
                output_dim = y_train.shape[-1] if y_train.size else 1

                callbacks = _build_callbacks(cfg)
                save_path = Path(save_path_str)

                for units in cfg["units"]:
                    for num_layers in cfg["num_layers"]:
                        print(
                            f"Training {alg_name}: data={datatype_opt}, seq={seq}, "
                            f"units={units}, layers={num_layers}, n_feat={used_features}"
                        )

                        model = _build_model(
                            units,
                            num_layers,
                            input_dim,
                            output_dim,
                            cfg["learning_rate"],
                            cfg["plot_model"],
                            cfg["model_plot_filename"],
                        )

                        fit_kwargs = {
                            "epochs": cfg["epochs"],
                            "batch_size": batch_size,
                            "verbose": cfg["verbose"],
                            "shuffle": True,
                        }
                        if callbacks:
                            fit_kwargs["callbacks"] = callbacks

                        if X_val.size and y_val.size:
                            fit_kwargs["validation_data"] = (X_val, y_val)
                        elif cfg["validation_split"] > 0:
                            fit_kwargs["validation_split"] = cfg["validation_split"]

                        start_train = time.time()
                        history = model.fit(X_train, y_train, **fit_kwargs)
                        train_time = (time.time() - start_train) / 60

                        start_test = time.time()
                        y_true, y_pred, rmse, mae, mape = _evaluate(model, X_test, y_test, scaler)
                        test_time = time.time() - start_test

                        best_epoch = _best_epoch(history) if callbacks else int(np.argmin(history.history["loss"]))

                        print(
                            f"Metrics -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, "
                            f"MAPE: {mape:.2f}%"
                        )

                        _persist_results(
                            save_path,
                            datatype_opt,
                            alg_name,
                            seq,
                            num_layers,
                            units,
                            best_epoch,
                            used_features,
                            train_time,
                            test_time,
                            rmse,
                            mae,
                            mape,
                            y_true,
                            y_pred,
                        )


if __name__ == "__main__":
    main()