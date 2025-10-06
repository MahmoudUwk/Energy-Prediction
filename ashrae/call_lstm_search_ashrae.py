from __future__ import annotations

from pathlib import Path
import sys
import os
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

    # Ensure project root on sys.path so config/tools imports work
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import search module to allow wrapping of internals
    import models.LSTM_hyperpara_search as hp

    # Import ASHRAE results saver for progress logging
    from ashrae.save_ashrae_results import save_ashrae_lstm_results

    # Create unique progress path based on algorithms
    progress_filename = f"search_progress_{algo_names}{output_suffix}.csv"
    progress_path = root / "results" / "ashrae" / progress_filename
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

    # Wrap _save_artifacts to log search-phase best params
    _orig_save_artifacts = hp._save_artifacts

    def _save_artifacts_wrapped(task, best_params, save_path, prefix, config):
        append_progress(
            {
                "phase": "search",
                "datatype": tuple(getattr(config, "datatype_options", ("1s",)))[0]
                if isinstance(config, dict) and "datatype_options" in config
                else "1s",
                "algorithm": prefix,
                "units": best_params.get("units"),
                "layers": best_params.get("num_layers"),
                "seq": best_params.get("seq"),
                "learning_rate": best_params.get("learning_rate"),
                "save_path": str(save_path),
                "artifact": f"Best_param{prefix}.obj",
            }
        )
        return _orig_save_artifacts(task, best_params, save_path, prefix, config)

    hp._save_artifacts = _save_artifacts_wrapped

    # Wrap _train_with_best_params to log training phase
    _orig_train_best = hp._train_with_best_params
    try:
        from tools.preprocess_data2 import get_SAMFOR_data
    except Exception:
        get_SAMFOR_data = None

    def _train_with_best_params_wrapped(config, datatype_opt, best_params, algorithm_name):
        append_progress(
            {
                "phase": "train_start",
                "datatype": datatype_opt,
                "algorithm": algorithm_name,
                "units": best_params.get("units"),
                "layers": best_params.get("num_layers"),
                "seq": best_params.get("seq"),
                "learning_rate": best_params.get("learning_rate"),
            }
        )

        result = _orig_train_best(config, datatype_opt, best_params, algorithm_name)

        # Log training completion with centralized saver (but don't override the main save)
        # The main LSTM script already saves via save_ashrae_lstm_results

        return result

    hp._train_with_best_params = _train_with_best_params_wrapped

    # Import ASHRAE preprocessing and config after path setup
    from ashrae.preprocessing_ashrae_disjoint import preprocess_ashrae_disjoint_splits, get_ashrae_lstm_data_disjoint
    from ashrae.ashrae_config import ASHRAE_TRAINING_CONFIG
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
        windowing_func=get_ashrae_lstm_data_disjoint
    )

    print("=" * 80, flush=True)
    print(f"LSTM HYPERPARAMETER SEARCH COMPLETED ({algo_names})", flush=True)
    print("=" * 80, flush=True)
    print(f"Best parameters: {results['best_params']}", flush=True)
    print(f"Best score: {results['best_score']:.4f}", flush=True)
    print(f"Search results: {len(results['search_results'])} iterations completed", flush=True)
    print(f"Progress CSV: {progress_path}", flush=True)
    
    return results


if __name__ == "__main__":
    # Default: run both algorithms
    run_lstm_search_ashrae(["Mod_FireflyAlgorithm", "FireflyAlgorithm"])


