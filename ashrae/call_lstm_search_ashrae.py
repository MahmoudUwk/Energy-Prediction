from __future__ import annotations

from pathlib import Path
import sys
import os
from datetime import datetime
import csv

# Immediate import-time banner to confirm execution start
print("[CALLER] ashrae.call_lstm_search_ashrae loaded", flush=True)


def main():
    # Force unbuffered stdout to ensure progress prints appear immediately
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("[CALLER] Starting LSTM hyperparameter search (ashrae.call_lstm_search_ashrae)", flush=True)

    # Ensure project root on sys.path so config/tools imports work
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import search module to allow wrapping of internals
    import models.LSTM_hyperpara_search as hp

    # Import ASHRAE results saver for progress logging
    from .save_ashrae_results import save_ashrae_lstm_results

    progress_path = (Path("results") / "ashrae" / "search_progress.csv").resolve()
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

    print("[CALLER] Launching models.LSTM_hyperpara_search.main()", flush=True)
    hp.main()


if __name__ == "__main__":
    main()


