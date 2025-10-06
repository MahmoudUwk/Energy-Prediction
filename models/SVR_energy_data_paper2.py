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

    # Persist unscaled arrays for plotting/scatter later
    save_object(
        {
            "y_test": np.asarray(y_true).flatten(),
            "y_test_pred": np.asarray(y_pred).flatten(),
            "seq_length": seq_length,
            "algorithm": name,
            "datatype": datatype_opt,
            "train_time_min": train_elapsed / 60,
            "test_time_s": test_elapsed,
        },
        save_path / f"{name}.obj",
    )
    
    return {
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape},
        "y_true": y_true,
        "y_pred": y_pred,
        "train_time": train_elapsed / 60,
        "test_time": test_elapsed,
        "algorithm": name
    }


def _available_models():
    return {
        "RFR": lambda: RandomForestRegressor(**SAMFOR_RANDOM_FOREST_PARAMS),
        "SVR": lambda: SVR(**SAMFOR_SVR_PARAMS),
    }


def run_svr(X_train, y_train, X_test, y_test, scaler, seq_length, save_path_str="results/ashrae"):
    """
    Run SVR models with pre-loaded data.
    
    Args:
        X_train: Training features (2D array)
        y_train: Training targets (1D array)
        X_test: Test features (2D array) 
        y_test: Test targets (1D array)
        scaler: Scaler object for inverse transformation
        seq_length: Sequence length (for compatibility)
        save_path_str: Path to save results
        
    Returns:
        dict: Results for each model including metrics and predictions
    """
    print(f"Training SVR models on data shapes: X_train{X_train.shape}, X_test{X_test.shape}")
    save_path = Path(save_path_str)
    models = _available_models()
    results = {}

    for name in SAMFOR_MODELS:
        if name not in models:
            print(f"Skipping unsupported model '{name}'.")
            continue
        model = models[name]()
        result = _train_and_evaluate(
            model,
            name,
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            save_path,
            SAMFOR_DATATYPE,
            seq_length,
        )
        results[name] = result
    
    return results


def main():
    """Legacy main function - now redirects to caller scripts."""
    raise NotImplementedError(
        "SVR main() should be called with pre-loaded data. "
        "Use ashrae/call_svr_ashrae.py for ASHRAE dataset."
    )


if __name__ == "__main__":
    main()