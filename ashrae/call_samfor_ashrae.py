from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR


def main():
    # Add project root to path for imports when running as script
    if __name__ == "__main__":
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    print("=" * 80)
    print("SAMFOR ASHRAE DATASET TRAINING WITH HYPERPARAMETER TUNING")
    print("=" * 80)

    # Import ASHRAE preprocessing and models after path setup
    from preprocessing_ashrae_disjoint import preprocess_ashrae_disjoint_splits, get_ashrae_lstm_data_disjoint
    from models.SAMFOR_trial1 import run_samfor
    from config import SAMFOR_SAMFOR_PARAMS
    from save_ashrae_results import save_ashrae_samfor_results

    # Load ASHRAE data
    print("Loading ASHRAE dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    # Get LSTM sequences then flatten for SAMFOR (2D data)
    seq_length = SAMFOR_SAMFOR_PARAMS["sequence_length"]
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_length
    )

    # SAMFOR expects 2D data (samples, features) not 3D (samples, timesteps, features)
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

    # SAMFOR needs scaled targets for consistency with _evaluate_model
    y_train_scaled = y_tr_lstm  # Use scaled targets for training
    y_val_scaled = y_va_lstm    # Use scaled targets for validation
    y_test_scaled = y_te_lstm   # Use scaled targets for evaluation

    print(f"Data shapes for SAMFOR:")
    print(f"  X_train: {X_train_flat.shape}")
    print(f"  X_val: {X_val_flat.shape}")
    print(f"  X_test: {X_test_flat.shape}")
    print(f"  y_train: {y_train_scaled.shape}")
    print(f"  y_val: {y_val_scaled.shape}")
    print(f"  y_test: {y_test_scaled.shape}")

    # Define reduced hyperparameter space for SVR (to limit total fits)
    param_distributions = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'epsilon': [0.001, 0.01, 0.1, 0.2],
        'gamma': ['scale', 'auto', 0.01],
    }

    print("\n🔍 Starting hyperparameter tuning with RandomizedSearchCV...")
    print(f"Parameter distributions: {param_distributions}")
    
    # Create base SVR model
    base_svr = SVR(kernel='rbf')
    
    # Perform randomized search with limited trials
    start_tune = time.time()
    grid_search = RandomizedSearchCV(
        estimator=base_svr,
        param_distributions=param_distributions,
        n_iter=8,
        scoring='neg_mean_squared_error',  # Use MSE for regression
        cv=2,  # fewer folds to reduce fits
        n_jobs=5,
        verbose=1,
        random_state=42,
    )
    
    # Fit on training data
    grid_search.fit(X_train_flat, y_train_scaled)
    tune_time = (time.time() - start_tune) / 60.0
    
    print(f"\n✅ Hyperparameter tuning completed in {tune_time:.2f} minutes")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score (MSE): {-grid_search.best_score_:.6f}")
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on validation set
    print("\n📊 Evaluating best model on validation set...")
    y_val_pred_scaled = best_model.predict(X_val_flat)
    
    # Convert to original scale for metrics
    y_val_true_orig = scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
    y_val_pred_orig = scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate validation metrics (robust MAPE ignoring near-zero denominators)
    val_rmse = np.sqrt(mean_squared_error(y_val_true_orig, y_val_pred_orig))
    val_mae = mean_absolute_error(y_val_true_orig, y_val_pred_orig)
    val_r2 = r2_score(y_val_true_orig, y_val_pred_orig)
    _mask_val = np.abs(y_val_true_orig) > 1.0
    val_mape = (
        100.0
        * np.mean(
            np.abs(
                (y_val_true_orig[_mask_val] - y_val_pred_orig[_mask_val])
                / y_val_true_orig[_mask_val]
            )
        )
        if np.any(_mask_val)
        else float("inf")
    )
    
    print(f"Validation Metrics:")
    print(f"  RMSE: {val_rmse:.4f} kWh")
    print(f"  MAE: {val_mae:.4f} kWh")
    print(f"  R²: {val_r2:.4f}")
    print(f"  MAPE: {val_mape:.2f}%")
    
    # Final training on train+val with best params
    X_train_all = np.vstack([X_train_flat, X_val_flat])
    y_train_all = np.hstack([y_train_scaled, y_val_scaled])
    final_model = SVR(kernel='rbf', **grid_search.best_params_)
    start_final = time.time()
    final_model.fit(X_train_all, y_train_all)
    final_train_time_min = (time.time() - start_final) / 60.0

    # Evaluate on test set (original scale)
    print("\n📊 Evaluating final model on test set (train+val)...")
    start_test = time.time()
    y_test_pred_scaled = final_model.predict(X_test_flat)
    test_time = time.time() - start_test
    y_test_true_orig = scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    y_test_pred_orig = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    test_rmse = np.sqrt(mean_squared_error(y_test_true_orig, y_test_pred_orig))
    test_mae = mean_absolute_error(y_test_true_orig, y_test_pred_orig)
    test_r2 = r2_score(y_test_true_orig, y_test_pred_orig)
    _mask_test = np.abs(y_test_true_orig) > 1.0
    test_mape = (
        100.0
        * np.mean(
            np.abs(
                (y_test_true_orig[_mask_test] - y_test_pred_orig[_mask_test])
                / y_test_true_orig[_mask_test]
            )
        )
        if np.any(_mask_test)
        else float("inf")
    )
    
    print(f"Test Metrics:")
    print(f"  RMSE: {test_rmse:.4f} kWh")
    print(f"  MAE: {test_mae:.4f} kWh")
    print(f"  R²: {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    
    # Save results using centralized ASHRAE saver
    metrics = {
        "RMSE": float(test_rmse),
        "MAE": float(test_mae),
        "R2": float(test_r2),
        "MAPE": float(test_mape)
    }
    
    saved_files = save_ashrae_samfor_results(
        metrics=metrics,
        y_true=y_test_true_orig,
        y_pred=y_test_pred_orig,
        train_time_min=tune_time + final_train_time_min,
        test_time_s=test_time,
        algorithm="SAMFOR_Tuned",
        seq_length=seq_length,
        datatype="1s",
    )

    print("=" * 80)
    print("SAMFOR HYPERPARAMETER TUNING COMPLETED")
    print("=" * 80)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Validation Metrics: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}, R²={val_r2:.4f}, MAPE={val_mape:.2f}%")
    print(f"Test Metrics: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, R²={test_r2:.4f}, MAPE={test_mape:.2f}%")
    print(f"Tuning time: {tune_time:.2f} min")
    print(f"Test time: {test_time:.4f} s")
    print(f"Results saved to: {saved_files}")


if __name__ == "__main__":
    main()


