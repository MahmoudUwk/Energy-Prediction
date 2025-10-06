from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tools.preprocess_data2 import inverse_transf, time_call


def train_and_evaluate_rfr(
    X_train: np.ndarray,
    y_train_scaled: np.ndarray,
    X_val: np.ndarray,
    y_val_scaled: np.ndarray,
    X_test: np.ndarray,
    y_test_scaled: np.ndarray,
    target_scaler,
    params: Dict,
):
    """Train RandomForestRegressor with given params and evaluate.

    Returns original-scale predictions and metrics on validation and test sets.
    """
    model = RandomForestRegressor(**params)

    start_train = time.time()
    model.fit(X_train, y_train_scaled)
    train_time_min = (time.time() - start_train) / 60.0

    # Validation
    y_val_pred_scaled = model.predict(X_val)
    y_val_true = inverse_transf(y_val_scaled, target_scaler).flatten()
    y_val_pred = inverse_transf(y_val_pred_scaled, target_scaler).flatten()

    val_rmse = float(np.sqrt(mean_squared_error(y_val_true, y_val_pred)))
    val_mae = float(mean_absolute_error(y_val_true, y_val_pred))
    val_r2 = float(r2_score(y_val_true, y_val_pred))
    val_mape = float(100 * np.mean(np.abs((y_val_true - y_val_pred) / np.clip(y_val_true, 1e-9, None))))

    # Test
    start_test = time.time()
    y_test_pred_scaled = model.predict(X_test)
    test_time_s = time.time() - start_test
    y_test_true = inverse_transf(y_test_scaled, target_scaler).flatten()
    y_test_pred = inverse_transf(y_test_pred_scaled, target_scaler).flatten()

    test_rmse = float(np.sqrt(mean_squared_error(y_test_true, y_test_pred)))
    test_mae = float(mean_absolute_error(y_test_true, y_test_pred))
    test_r2 = float(r2_score(y_test_true, y_test_pred))
    test_mape = float(100 * np.mean(np.abs((y_test_true - y_test_pred) / np.clip(y_test_true, 1e-9, None))))

    return {
        "model": model,
        "train_time_min": train_time_min,
        "val_metrics": {"RMSE": val_rmse, "MAE": val_mae, "R2": val_r2, "MAPE": val_mape},
        "test_metrics": {"RMSE": test_rmse, "MAE": test_mae, "R2": test_r2, "MAPE": test_mape},
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }



