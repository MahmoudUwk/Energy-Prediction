from __future__ import annotations

from pathlib import Path
import os
import sys
from datetime import datetime
import csv
from typing import List


def run_lstm_search_ashrae(algorithms: List[str], output_suffix: str = ""):
    # Force unbuffered stdout to ensure progress prints appear immediately
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    algo_names = "_".join([a.replace("_", "") for a in algorithms])
    print(f"[CALLER] Starting LSTM hyperparameter search ({algo_names}){output_suffix}", flush=True)
    print(f"[CALLER] Algorithms: {algorithms}", flush=True)

    # Import search module to allow wrapping of internals
    import models.LSTM_hyperpara_search as hp

    # Import ASHRAE results saver for progress logging
    from .save_ashrae_results import save_ashrae_lstm_results

    # Create unique progress path based on algorithms (under unified results root)
    from .ashrae_config import ASHRAE_RESULTS_ROOT
    progress_filename = f"search_progress_{algo_names}{output_suffix}.csv"
    progress_path = ASHRAE_RESULTS_ROOT / progress_filename
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CSV header if it doesn't exist
    if not progress_path.exists():
        with progress_path.open("w", newline="") as f:
            cols = ["timestamp", "phase", "datatype", "algorithm", "units", "layers", "seq", "learning_rate", "save_path", "artifact"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()

    def append_progress(row: dict) -> None:
        cols = [
            "timestamp",
            "phase",
            "datatype",
            "algorithm",
            "units",
            "layers",
            "seq",
            "learning_rate",
            "save_path",
            "artifact",
        ]
        row_out = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            **{k: row.get(k, "") for k in cols if k != "timestamp"},
        }
        try:
            with progress_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writerow(row_out)
        except Exception as e:
            print(f"[ERROR] Failed to write to progress CSV: {e}", flush=True)

    # Legacy wrappers for artifacts/training are no longer used and have been removed

    # Import ASHRAE preprocessing and config
    from .preprocessing_ashrae_disjoint import (
        preprocess_ashrae_disjoint_splits,
        get_ashrae_lstm_data_disjoint,
    )
    from .ashrae_config import ASHRAE_TRAINING_CONFIG
    from config import LSTM_SEARCH_CONFIG
    
    # Override algorithms in config
    LSTM_SEARCH_CONFIG_CUSTOM = LSTM_SEARCH_CONFIG.copy()
    LSTM_SEARCH_CONFIG_CUSTOM["algorithms"] = tuple(algorithms)
    
    # Temporarily override the module-level config
    hp.LSTM_SEARCH_CONFIG = LSTM_SEARCH_CONFIG_CUSTOM

    print("[CALLER] Loading ASHRAE dataset...", flush=True)
    # Load ASHRAE data (RAW, not windowed yet - windowing happens per-iteration)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    print(f"[CALLER] Raw data shapes:", flush=True)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}", flush=True)
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}", flush=True)
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}", flush=True)

    print("[CALLER] Launching LSTM hyperparameter search...", flush=True)
    print("         (Data will be windowed dynamically based on seq parameter)", flush=True)
    # Run LSTM hyperparameter search - pass RAW data, windowing function, and scaler
    results = hp.run_lstm_search(
        X_train_raw=X_train,
        y_train_raw=y_train,
        X_val_raw=X_val,
        y_val_raw=y_val,
        X_test_raw=X_test,
        y_test_raw=y_test,
        scaler=scaler,
        windowing_func=get_ashrae_lstm_data_disjoint,
        output_suffix=output_suffix,
    )

    print("=" * 80, flush=True)
    print(f"LSTM HYPERPARAMETER SEARCH COMPLETED ({algo_names})", flush=True)
    print("=" * 80, flush=True)
    print(f"Best parameters: {results['best_params']}", flush=True)
    print(f"Best score: {results['best_score']:.4f}", flush=True)
    print(f"Search results: {len(results['search_results'])} iterations completed", flush=True)
    print(f"Progress CSV: {progress_path}", flush=True)
    
    return results

def run_lstm_search_single(algorithm: str, output_suffix: str = ""):
    """Convenience wrapper to run search for a single algorithm."""
    return run_lstm_search_ashrae([algorithm], output_suffix=output_suffix)


