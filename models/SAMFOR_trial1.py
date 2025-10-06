from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVR

from config import SAMFOR_SAMFOR_PARAMS
from sklearn.svm import LinearSVR

# Import ASHRAE results saver for consistent saving
import sys
from pathlib import Path
ashrae_path = Path(__file__).parent.parent / "ashrae"
if str(ashrae_path) not in sys.path:
    sys.path.insert(0, str(ashrae_path))

from save_ashrae_results import save_ashrae_samfor_results

# Remove ASHRAE-specific imports - dataset loading should be handled by callers

from tools.preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    compute_metrics,
    get_SAMFOR_data,
    inverse_transf,
    log_results,
    persist_model_results,
    plot_test,
    time_call,
)


def _train_model(use_lssvr: bool, X_train, y_train, lssvr_params, svr_params):
    if use_lssvr:
        # LinearSVR uses only C parameter, ignore gamma and kernel
        linear_svr_params = {"C": lssvr_params.get("C", 1.0)}
        model = LinearSVR(**linear_svr_params)
        name = "SAMFOR_SARIMA_LinearSVR"
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
    rmse, mae, mape, rmsle = compute_metrics(y_true, y_pred)
    print(f"rmse: {rmse:.4f} || mape: {mape:.2f} || mae: {mae:.4f}")
    return y_true, y_pred, rmse, mae, mape, test_elapsed


def run_samfor(X_train, y_train, X_test, y_test, scaler, seq_length, save_path_str="results/ashrae"):
    """
    Run SAMFOR model with pre-loaded data.
    
    Args:
        X_train: Training features (2D array)
        y_train: Training targets (1D array)
        X_test: Test features (2D array) 
        y_test: Test targets (1D array)
        scaler: Scaler object for inverse transformation
        seq_length: Sequence length (for compatibility)
        save_path_str: Path to save results
        
    Returns:
        dict: Results including metrics and predictions
    """
    params = SAMFOR_SAMFOR_PARAMS
    
    print(f"Training SAMFOR on data shapes: X_train{X_train.shape}, X_test{X_test.shape}")
    save_path = Path(save_path_str)

    use_lssvr = any(name.upper().startswith("SAMFOR_SARIMA") for name in params.get("algorithms", ("SAMFOR",)))
    model, alg_name, train_time = _train_model(
        use_lssvr,
        X_train,
        y_train,
        params["lssvr_params"],
        params["svr_params"],
    )

    # Evaluate model - expects scaled targets
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

    # Save results using centralized ASHRAE saver
    saved_files = save_ashrae_samfor_results(
        metrics={"RMSE": rmse, "MAE": mae, "MAPE": mape},
        y_true=y_true,
        y_pred=y_pred,
        train_time_min=train_time,
        test_time_s=test_time,
        algorithm=alg_name,
        seq_length=seq_length,
        datatype=params["datatype"],
    )
    
    return {
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE": mape},
        "y_true": y_true,
        "y_pred": y_pred,
        "train_time": train_time,
        "test_time": test_time,
        "algorithm": alg_name,
        "saved_files": saved_files
    }


def main():
    """Legacy main function - now redirects to caller scripts."""
    raise NotImplementedError(
        "SAMFOR main() should be called with pre-loaded data. "
        "Use ashrae/call_samfor_ashrae.py for ASHRAE dataset."
    )


if __name__ == "__main__":
    main()