from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():

    print("=" * 80)
    print("RFR ASHRAE DATASET TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 80)

    from .preprocessing_ashrae_disjoint import (
        preprocess_ashrae_disjoint_splits,
        get_ashrae_lstm_data_disjoint,
    )
    from models.RFR_energy_data import train_and_evaluate_rfr
    from .save_ashrae_results import save_ashrae_rfr_results
    from config import SAMFOR_SAMFOR_PARAMS

    # Load ASHRAE data
    print("Loading ASHRAE dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    # Get LSTM sequences then flatten to 2D for RFR
    seq_length = SAMFOR_SAMFOR_PARAMS["sequence_length"]
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_length
    )
    # Feature augmentation: last-step features + target lags/statistics
    def build_aug_features(X_lstm: np.ndarray) -> np.ndarray:
        last_step = X_lstm[:, -1, :]
        target_hist = X_lstm[:, :, 0]
        lag1 = target_hist[:, -1]
        lag2 = target_hist[:, -2] if X_lstm.shape[1] >= 2 else lag1
        lag3 = target_hist[:, -3] if X_lstm.shape[1] >= 3 else lag2
        roll_mean = target_hist.mean(axis=1)
        roll_std = target_hist.std(axis=1)
        extra = np.stack([lag1, lag2, lag3, roll_mean, roll_std], axis=1)
        return np.concatenate([last_step, extra], axis=1)

    X_train_flat = build_aug_features(X_tr_lstm)
    X_val_flat = build_aug_features(X_va_lstm)
    X_test_flat = build_aug_features(X_te_lstm)

    print(f"Train: {X_train_flat.shape}, Val: {X_val_flat.shape}, Test: {X_test_flat.shape}")

    # Hyperparameter grid for RFR
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20, 40],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5, None],
        "bootstrap": [True, False],
        "random_state": [0],
        "n_jobs": [5],
    }

    base_rfr = RandomForestRegressor()
    start_tune = time.time()
    grid_search = RandomizedSearchCV(
        estimator=base_rfr,
        param_distributions=param_grid,
        n_iter=12,
        scoring="neg_mean_squared_error",
        cv=2,
        n_jobs=5,
        verbose=1,
        random_state=42,
    )
    grid_search.fit(X_train_flat, y_tr_lstm)
    tune_time = (time.time() - start_tune) / 60.0

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV MSE: {-grid_search.best_score_:.4f}")

    # Final training on train+val with best params
    best_params = grid_search.best_params_
    X_train_all = np.vstack([X_train_flat, X_val_flat])
    y_train_all = np.hstack([y_tr_lstm, y_va_lstm])

    start_final = time.time()
    final_model = RandomForestRegressor(**best_params)
    final_model.fit(X_train_all, y_train_all)
    final_train_time_min = (time.time() - start_final) / 60.0

    # Evaluate on test
    y_test_pred_scaled = final_model.predict(X_test_flat)
    y_test_true = scaler.inverse_transform(y_te_lstm.reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    test_rmse = float(np.sqrt(np.mean((y_test_true - y_test_pred) ** 2)))
    test_mae = float(np.mean(np.abs(y_test_true - y_test_pred)))
    test_r2 = float(1 - np.sum((y_test_true - y_test_pred) ** 2) / np.sum((y_test_true - y_test_true.mean()) ** 2))
    _mask_test = np.abs(y_test_true) > 1.0
    test_mape = float(
        100.0
        * np.mean(
            np.abs((y_test_true[_mask_test] - y_test_pred[_mask_test]) / y_test_true[_mask_test])
        )
    ) if np.any(_mask_test) else float("inf")

    result = {
        "val_metrics": {},
        "test_metrics": {"RMSE": test_rmse, "MAE": test_mae, "R2": test_r2, "MAPE": test_mape},
        "train_time_min": final_train_time_min,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
    }

    metrics = result["test_metrics"]

    saved = save_ashrae_rfr_results(
        metrics=metrics,
        y_true=result["y_test_true"],
        y_pred=result["y_test_pred"],
        train_time_min=tune_time + result["train_time_min"],
        test_time_s=(0.0),
        seq_length=seq_length,
    )

    print("=" * 80)
    print("RFR HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)
    print(f"Best Parameters: {best_params}")
    print(f"Validation: {result['val_metrics']}")
    print(f"Test: {metrics}")
    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()


