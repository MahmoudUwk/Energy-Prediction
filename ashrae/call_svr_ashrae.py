from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR

# Add project root to path for imports when running as script
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from preprocessing_ashrae_disjoint import (
    preprocess_ashrae_disjoint_splits,
    get_ashrae_lstm_data_disjoint,
)
from ashrae_config import ASHRAE_TRAINING_CONFIG, ASHRAE_RESULTS_ROOT
from save_ashrae_results import save_ashrae_svr_results
from tools.preprocess_data2 import RMSLE


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": float(
            100
            * np.mean(np.abs((y_true[y_true != 0] - y_pred[y_true != 0]) / y_true[y_true != 0]))
        )
        if np.any(y_true != 0)
        else float("inf"),
        "R2": r2_score(y_true, y_pred),
        "RMSLE": RMSLE(y_true, y_pred),
    }


def main() -> bool:
    print("=" * 80)
    print("ASHRAE SVR - Disjoint Building Splits")
    print("=" * 80)

    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = preprocess_ashrae_disjoint_splits(
        target_samples=ASHRAE_TRAINING_CONFIG["max_samples"],
        train_fraction=ASHRAE_TRAINING_CONFIG["train_fraction"],
        val_fraction=ASHRAE_TRAINING_CONFIG["val_fraction"],
        test_fraction=ASHRAE_TRAINING_CONFIG["test_fraction"],
    )

    # Build sequences (per building) then flatten to last timestep features
    seq_len = ASHRAE_TRAINING_CONFIG["sequence_length"]
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_len
    )

    X_tr_flat = X_tr_lstm[:, -1, :]
    X_va_flat = X_va_lstm[:, -1, :]
    X_te_flat = X_te_lstm[:, -1, :]

    # Combine train+val to train SVR
    X_train_all = np.vstack([X_tr_flat, X_va_flat])
    y_train_all = np.hstack([y_tr_lstm, y_va_lstm])

    print(f"Train (flat): {X_train_all.shape} | Test (flat): {X_te_flat.shape}")

    # Train SVR on scaled targets
    model = SVR(C=10.0, epsilon=0.01, kernel="rbf", gamma="scale")

    start_train = time.time()
    model.fit(X_train_all, y_train_all)
    train_time = (time.time() - start_train) / 60.0

    start_pred = time.time()
    y_pred_scaled = model.predict(X_te_flat)
    test_time = time.time() - start_pred

    # Inverse transform to original scale
    y_true = target_scaler.inverse_transform(y_te_lstm.reshape(-1, 1)).flatten()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = calc_metrics(y_true, y_pred)

    print("\n📊 Metrics (original scale):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  Train time (min): {train_time:.2f}")
    print(f"  Test time (s):   {test_time:.2f}")

    # Save outputs using centralized saver
    saved_files = save_ashrae_svr_results(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        train_time_min=train_time,
        test_time_s=test_time,
        seq_length=seq_len,
    )

    print(f"\nSaved results to: {saved_files['metrics'].parent}")
    for save_type, path in saved_files.items():
        print(f"  {save_type}: {path.name}")
    return True


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)


