"""Data preprocessing utilities for energy prediction experiments."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import (
    RESULTS_ROOT,
    RESAMPLED_DATA_ROOT,
    SARIMA_LOOKBACK,
    MINMAX_FEATURE_RANGE,
    train_split_for,
    lstm_train_fraction,
    lstm_val_fraction,
)


# ----------------------------------------------------------------------
# Error metrics & tensor helpers
# ----------------------------------------------------------------------
def RMSE(test, pred):
    return np.sqrt(np.mean((np.squeeze(test) - np.squeeze(pred)) ** 2))


def MAE(test, pred):
    return np.mean(np.abs(np.squeeze(pred) - np.squeeze(test)))


def MAPE(test, pred):
    ind = np.where(test != 0)[0].flatten()
    return 100 * np.mean(
        np.abs(np.squeeze(pred[ind]) - np.squeeze(test[ind]))
        / np.abs(np.squeeze(test[ind]))
    )


def inverse_transf(X, scaler):
    return np.array((X * (scaler.data_max_[0] - scaler.data_min_[0])) + scaler.data_min_[0])


def expand_dims(X):
    return np.expand_dims(X, axis=len(X.shape))


def expand_dims_first(x):
    return np.expand_dims(x, axis=0)


# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def feature_creation(data):
    df = data.copy()
    df["Minute"] = data.index.minute.astype(float)
    df["DOW"] = data.index.dayofweek.astype(float)
    df["H"] = data.index.hour.astype(float)
    return df


# ----------------------------------------------------------------------
# Sliding-window generators
# ----------------------------------------------------------------------
def sliding_windows(data, seq_length, k_step):
    x = np.zeros((len(data) - seq_length - k_step + 1, seq_length))
    y = np.zeros((len(data) - seq_length - k_step + 1, k_step))
    for ind in range(len(x)):
        x[ind, :] = data[ind : ind + seq_length]
        y[ind, :] = data[ind + seq_length : ind + seq_length + k_step]
    return x, y


def sliding_windows2d(data, seq_length, k_step, num_feat):
    x = np.zeros((len(data) - seq_length - k_step + 1, seq_length * num_feat))
    y = np.zeros((len(data) - seq_length - k_step + 1, k_step))
    for ind in range(len(x)):
        x[ind, :] = np.reshape(data[ind : ind + seq_length, :], -1)
        y[ind] = data[ind + seq_length : ind + seq_length + k_step, 0]
    return x, y


def sliding_windows2d_lstm(data, seq_length):
    x = np.zeros((len(data) - seq_length, seq_length, data.shape[1]))
    y = np.zeros((len(data) - seq_length, 1))
    for ind in range(len(x)):
        x[ind, :, :] = data[ind : ind + seq_length, :]
        y[ind] = data[ind + seq_length, 0]
    return x, y


# ----------------------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------------------
def save_object(obj, filename):
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def loadDatasetObj(fname):
    with open(fname, "rb") as file_id:
        return pickle.load(file_id)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def plot_test(test_time, y_test, y_test_pred, path, alg_name):
    from matplotlib import pyplot as plt

    scale_mv = 1000
    plt.figure(figsize=(10, 7), dpi=180)
    plt.plot(test_time, scale_mv * y_test, color="red", linewidth=2.0, alpha=0.6)
    plt.plot(test_time, scale_mv * y_test_pred, color="blue", linewidth=0.8)
    plt.legend(["Actual", "Predicted"])
    plt.xlabel("Timestamp")
    plt.ylabel("mW")
    plt.title(f"Energy Prediction using {alg_name}")
    plt.xticks(rotation=25)
    plt.show()
    plt.savefig(path)


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def _append_row(save_path: str, filename: str, columns: list[str], row):
    file_path = Path(save_path) / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.exists():
        pd.DataFrame(columns=columns).to_csv(file_path, index=False)
    df = pd.read_csv(file_path)
    df.loc[len(df)] = row
    df.to_csv(file_path, index=False, header=True)


def log_results(row, datatype_opt, save_path):
    save_name = f"results_{datatype_opt}.csv"
    cols = ["Algorithm", "RMSE", "MAE", "MAPE(%)", "seq", "train_time(min)", "test_time(s)"]
    _append_row(save_path, save_name, cols, row)


def log_results_LSTM(row, datatype_opt, save_path):
    save_name = f"results_LSTM_feat_3_{datatype_opt}.csv"
    cols = [
        "Algorithm",
        "RMSE",
        "MAE",
        "MAPE(%)",
        "seq",
        "num_layers",
        "units",
        "best epoch",
        "data_type",
        "train_time(min)",
        "test_time(s)",
        "n_feat",
    ]
    _append_row(save_path, save_name, cols, row)


def get_Hzdata(datatype_opt, data_root: Optional[Path] = None, save_root: Optional[Path] = None):
    data_dir = Path(data_root) if data_root is not None else RESAMPLED_DATA_ROOT
    save_dir = Path(save_root) if save_root is not None else RESULTS_ROOT

    save_path = save_dir / datatype_opt
    save_path.mkdir(parents=True, exist_ok=True)

    data_path = data_dir / f"{datatype_opt}.csv"
    df = pd.read_csv(data_path)
    df.set_index(pd.to_datetime(df["timestamp"]), inplace=True, drop=True, append=False)
    if "timestamp" in df.columns:
        df.drop(columns=["timestamp"], inplace=True)
    df = df[["P", "Q", "V"]]
    df = feature_creation(df)
    return df, str(save_path)


def _load_dataset(datatype_opt: str):
    return get_Hzdata(datatype_opt)


# ----------------------------------------------------------------------
# Primary data factory
# ----------------------------------------------------------------------
def get_SAMFOR_data(option, datatype_opt, seq_length, get_sav_path=0):
    df, save_root_str = _load_dataset(datatype_opt)
    save_root = Path(save_root_str)

    if get_sav_path == 1:
        return save_root_str

    len_data = len(df)
    train_ratio = train_split_for(datatype_opt)
    train_len = int(train_ratio * len_data)
    sarima_len = min(SARIMA_LOOKBACK, train_len) if option in (0, 1) else 0
    train_len_lssvr = max(train_len - sarima_len, 0)
    test_len = len_data - train_len

    df_array = df.to_numpy()
    scaler = MinMaxScaler(feature_range=MINMAX_FEATURE_RANGE)
    if train_len > 0:
        scaler.fit(df_array[:train_len])
    else:
        scaler.fit(df_array)
    df_normalized = pd.DataFrame(scaler.transform(df_array), index=df.index, columns=df.columns)

    multivariate = df_normalized.shape[1] > 1
    k_step = 1

    if option == 0:
        return (
            df_normalized.iloc[:sarima_len, :],
            train_len_lssvr,
            test_len,
            save_root_str,
            scaler,
        )

    if option == 2:
        train_clf = df_normalized.iloc[:train_len].to_numpy()
        testset = df_normalized.iloc[train_len:].to_numpy()
        test_time = df_normalized.index[train_len:-seq_length]
        if multivariate:
            X_clf, y_clf = sliding_windows2d(train_clf, seq_length, k_step, train_clf.shape[1])
            X_test, y_test = sliding_windows2d(testset, seq_length, k_step, testset.shape[1])
        else:
            X_clf, y_clf = sliding_windows(train_clf.squeeze(), seq_length, k_step)
            X_test, y_test = sliding_windows(testset.squeeze(), seq_length, k_step)
        return (
            X_clf,
            np.squeeze(y_clf),
            X_test,
            np.squeeze(y_test),
            save_root_str,
            test_time,
            scaler,
        )

    if option == 3:
        train_portion = lstm_train_fraction()
        val_portion = lstm_val_fraction()
        train_len_lstm = int(train_portion * len_data)
        val_len_lstm = int(val_portion * len_data)
        train_x = df_normalized.iloc[:train_len_lstm].to_numpy()
        val_x = df_normalized.iloc[train_len_lstm : train_len_lstm + val_len_lstm].to_numpy()
        test_x = df_normalized.iloc[train_len_lstm + val_len_lstm :].to_numpy()
        test_time = df_normalized.index[train_len_lstm + val_len_lstm : -seq_length]
        X_train, y_train = sliding_windows2d_lstm(train_x, seq_length)
        if val_len_lstm:
            X_val, y_val = sliding_windows2d_lstm(val_x, seq_length)
            y_val = np.squeeze(y_val)
        else:
            X_val, y_val = [], []
        X_test, y_test = sliding_windows2d_lstm(test_x, seq_length)
        return (
            X_train,
            np.squeeze(y_train),
            X_val,
            y_val,
            X_test,
            np.squeeze(y_test),
            save_root_str,
            test_time,
            scaler,
        )

    if option == 1:
        if train_len_lssvr <= 0:
            raise ValueError("LSSVR training window must be positive when option=1")
        sarima_path = save_root / "SARIMA_prediction_P_.csv"
        sarima_pred = pd.read_csv(sarima_path, header=None).to_numpy()
        train_lssvr = df_normalized.iloc[sarima_len : sarima_len + train_len_lssvr].to_numpy()
        testset = df_normalized.iloc[train_len:].to_numpy()
        test_time = df_normalized.index[train_len:-seq_length]
        if multivariate:
            X_LSSVR, y_LSSVR = sliding_windows2d(train_lssvr, seq_length, k_step, train_lssvr.shape[1])
            X_test, y_test = sliding_windows2d(testset, seq_length, k_step, testset.shape[1])
        else:
            X_LSSVR, y_LSSVR = sliding_windows(train_lssvr.squeeze(), seq_length, k_step)
            X_test, y_test = sliding_windows(testset.squeeze(), seq_length, k_step)
        sarima_train = sarima_pred[seq_length : seq_length + X_LSSVR.shape[0]]
        sarima_test = sarima_pred[
            train_len_lssvr + seq_length - 1 : train_len_lssvr + seq_length - 1 + X_test.shape[0]
        ]
        X_LSSVR = np.concatenate((X_LSSVR, sarima_train), axis=1)
        X_test = np.concatenate((X_test, sarima_test), axis=1)
        return (
            X_LSSVR,
            y_LSSVR,
            X_test,
            y_test,
            save_root_str,
            test_time,
            scaler,
        )

    raise ValueError(f"Unsupported option: {option}")