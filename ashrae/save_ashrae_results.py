"""Centralized ASHRAE results saving utility.

This module provides a consistent interface for saving ASHRAE model results,
ensuring all models save metrics, predictions, and metadata in a standardized format.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd

from tools.preprocess_data2 import save_object


class ASHRAEResultsSaver:
    """Centralized saver for ASHRAE model results."""

    def __init__(self, results_root: Path, model_name: str, algorithm: str = None):
        """
        Initialize the results saver.

        Args:
            results_root: Root directory for ASHRAE results
            model_name: Name of the model (e.g., 'svr', 'samfor', 'lstm')
            algorithm: Specific algorithm name (e.g., 'SVR_ASHRAE', 'SAMFOR', 'LSTM_ModFF')
        """
        self.results_root = results_root
        self.model_name = model_name
        self.algorithm = algorithm or model_name

        # Create directory structure
        self.results_dir = results_root / model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Artifact directory for large objects
        self.artifacts_dir = self.results_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)

    def save_metrics(self, metrics: Dict[str, float]) -> Path:
        """Save model metrics to CSV."""
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.results_dir / "metrics.csv"

        if metrics_path.exists():
            # Append to existing file
            existing_df = pd.read_csv(metrics_path)
            combined_df = pd.concat([existing_df, metrics_df], ignore_index=True)
            combined_df.to_csv(metrics_path, index=False)
        else:
            # Create new file
            metrics_df.to_csv(metrics_path, index=False)

        return metrics_path

    def save_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        algorithm: str = None,
        **kwargs
    ) -> Path:
        """
        Save unscaled predictions and test data with a unified envelope.

        Args:
            y_true: Ground truth values (unscaled)
            y_pred: Predicted values (unscaled)
            algorithm: Override algorithm name
            **kwargs: Additional metadata (timing, params, etc.)
        """
        alg_name = algorithm or self.algorithm
        payload = {
            "y_test": y_true,
            "y_test_pred": y_pred,
            **kwargs,
        }
        return self.save_artifact(payload=payload, algorithm=alg_name)

    def save_artifact(
        self,
        payload: Dict[str, Any],
        name: str | None = None,
        algorithm: str | None = None,
        **kwargs,
    ) -> Path:
        """Save a generic artifact with a unified envelope schema.

        The saved object includes: algorithm, model_name, timestamp, and payload.
        """
        alg_name = algorithm or self.algorithm
        artifact_data: Dict[str, Any] = {
            "algorithm": alg_name,
            "model_name": self.model_name,
            "timestamp": time.time(),
            "payload": payload,
            **kwargs,
        }
        filename = f"{name or alg_name}.obj"
        artifact_path = self.artifacts_dir / filename
        save_object(artifact_data, artifact_path)
        return artifact_path

    def save_model_info(self, model_info: Dict[str, Any]) -> Path:
        """Save model information and parameters."""
        info_path = self.results_dir / "model_info.json"

        if info_path.exists():
            # Load existing info and merge
            with open(info_path, 'r') as f:
                existing_info = json.load(f)
            existing_info.update(model_info)
            model_info = existing_info

        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)

        return info_path

    def save_training_log(self, log_data: Dict[str, Any]) -> Path:
        """Save training/validation logs and convergence data."""
        log_path = self.results_dir / "training_log.json"

        if log_path.exists():
            # Load existing log and merge
            with open(log_path, 'r') as f:
                existing_log = json.load(f)
            existing_log.update(log_data)
            log_data = existing_log

        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)

        return log_path

    def save_comparison_data(self, comparison_data: Dict[str, Any]) -> Path:
        """Save data for model comparison across different runs."""
        comp_path = self.results_dir / "comparison_data.json"

        if comp_path.exists():
            # Load existing data and merge
            with open(comp_path, 'r') as f:
                existing_data = json.load(f)
            existing_data.update(comparison_data)
            comparison_data = existing_data

        with open(comp_path, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)

        return comp_path

    def save_all(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_info: Optional[Dict[str, Any]] = None,
        training_log: Optional[Dict[str, Any]] = None,
        comparison_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Path]:
        """
        Save all results in one call.

        Returns:
            Dict mapping save type to file path
        """
        saved_files = {}

        # Save metrics
        saved_files["metrics"] = self.save_metrics(metrics)

        # Save predictions
        saved_files["predictions"] = self.save_predictions(
            y_true, y_pred, **kwargs
        )

        # Save model info if provided
        if model_info:
            saved_files["model_info"] = self.save_model_info(model_info)

        # Save training log if provided
        if training_log:
            saved_files["training_log"] = self.save_training_log(training_log)

        # Save comparison data if provided
        if comparison_data:
            saved_files["comparison_data"] = self.save_comparison_data(comparison_data)

        return saved_files


def get_ashrae_results_saver(model_name: str, algorithm: str = None) -> ASHRAEResultsSaver:
    """Factory function to get ASHRAE results saver.

    Supports execution as a script or as a package by attempting absolute import
    first and falling back to relative import.
    """
    try:
        from ashrae_config import ASHRAE_RESULTS_ROOT  # type: ignore
    except Exception:
        from .ashrae_config import ASHRAE_RESULTS_ROOT  # type: ignore
    return ASHRAEResultsSaver(ASHRAE_RESULTS_ROOT, model_name, algorithm)


def save_ashrae_search_artifact(
    algorithm: str,
    payload: Dict[str, Any],
    name: str | None = None,
    **kwargs,
) -> Path:
    """Convenience: save LSTM search artifacts under lstm_search model."""
    saver = get_ashrae_results_saver("lstm_search", algorithm)
    return saver.save_artifact(payload=payload, name=name, **kwargs)


# Convenience functions for common use cases
def save_ashrae_svr_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time_min: float,
    test_time_s: float,
    seq_length: int = 23,
    **kwargs
) -> Dict[str, Path]:
    """Save SVR results with standard metadata."""
    saver = get_ashrae_results_saver("svr", "SVR_ASHRAE")

    model_info = {
        "algorithm": "SVR_ASHRAE",
        "model_type": "Support Vector Regression",
        "train_time_min": train_time_min,
        "test_time_s": test_time_s,
        "sequence_length": seq_length,
    }

    return saver.save_all(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        model_info=model_info,
        **kwargs
    )


def save_ashrae_rfr_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time_min: float,
    test_time_s: float,
    seq_length: int = 23,
    **kwargs
) -> Dict[str, Path]:
    """Save Random Forest Regression (RFR) results with standard metadata."""
    saver = get_ashrae_results_saver("rfr", "RFR_ASHRAE")

    model_info = {
        "algorithm": "RFR_ASHRAE",
        "model_type": "RandomForestRegressor",
        "train_time_min": train_time_min,
        "test_time_s": test_time_s,
        "sequence_length": seq_length,
    }

    return saver.save_all(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        model_info=model_info,
        **kwargs
    )


def save_ashrae_samfor_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time_min: float,
    test_time_s: float,
    algorithm: str,
    seq_length: int = 23,
    **kwargs
) -> Dict[str, Path]:
    """Save SAMFOR results with standard metadata."""
    saver = get_ashrae_results_saver("samfor", algorithm)

    model_info = {
        "algorithm": algorithm,
        "model_type": "SAMFOR",
        "train_time_min": train_time_min,
        "test_time_s": test_time_s,
        "sequence_length": seq_length,
    }

    return saver.save_all(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        model_info=model_info,
        **kwargs
    )


def save_ashrae_lstm_results(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    best_params: Dict[str, Any],
    best_epoch: int,
    algorithm: str,
    datatype: str,
    train_time_min: float = 0,
    test_time_s: float = 0,
    **kwargs
) -> Dict[str, Path]:
    """Save LSTM results with standard metadata."""
    saver = get_ashrae_results_saver("lstm", algorithm)

    model_info = {
        "algorithm": algorithm,
        "model_type": "LSTM",
        "best_params": best_params,
        "best_epoch": best_epoch,
        "datatype": datatype,
        "train_time_min": train_time_min,
        "test_time_s": test_time_s,
    }

    return saver.save_all(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        model_info=model_info,
        **kwargs
    )


if __name__ == "__main__":
    # Test the saver
    saver = get_ashrae_results_saver("test_model", "TestAlgorithm")

    # Test metrics
    test_metrics = {"RMSE": 1.23, "MAE": 0.89, "R2": 0.95}
    metrics_path = saver.save_metrics(test_metrics)
    print(f"Metrics saved to: {metrics_path}")

    # Test predictions
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    pred_path = saver.save_predictions(y_true, y_pred, train_time_min=5.5, test_time_s=0.2)
    print(f"Predictions saved to: {pred_path}")

    print(f"Results directory: {saver.results_dir}")

