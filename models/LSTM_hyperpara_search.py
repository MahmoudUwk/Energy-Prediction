from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential
from keras.optimizers import Adam
from niapy.algorithms.basic import FireflyAlgorithm
from niapy.algorithms.modified import Mod_FireflyAlgorithm
from niapy.problems import Problem
from niapy.task import OptimizationType, Task

from config import LSTM_SEARCH_CONFIG
from tools.preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    save_object,
)

def R2(y_true, y_pred):
    """Calculate R-squared metric."""
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

# from niapy.algorithms.basic import BeesAlgorithm

class ProgressPrinter(Callback):
    def __init__(self, step: int = 20, prefix: str = "", total_epochs: int | None = None):
        super().__init__()
        self.step = step
        self.prefix = prefix
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.step == 0:
            msg = f"{self.prefix} epoch {epoch + 1}"
            if self.total_epochs:
                msg += f"/{self.total_epochs}"
            if "loss" in logs:
                msg += f" loss {logs['loss']:.4f}"
            if "val_loss" in logs:
                msg += f" val_loss {logs['val_loss']:.4f}"
            print(msg, flush=True)

def _hyperparameters_from_vector(x: np.ndarray) -> Dict[str, Any]:
    return {
        "units": int(x[0] * 116 + 10),
        "num_layers": int(x[1] * 4) + 1,  # max 4 layers
        "seq": int(x[2] * 30 + 1),
        "learning_rate": x[3] * 2e-2 + 0.5e-3,
    }


def _build_model(input_dim, output_dim, units, num_layers, learning_rate):
    model = Sequential(name="LSTM_HP")
    # Use Input layer for modern Keras approach (removes warning)
    model.add(Input(shape=input_dim))
    model.add(LSTM(units=units, return_sequences=num_layers > 1))
    for idx in range(1, num_layers):
        model.add(LSTM(units=units, return_sequences=idx < num_layers - 1))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def run_lstm_search(
    X_train_raw,
    y_train_raw,
    X_val_raw,
    y_val_raw,
    X_test_raw,
    y_test_raw,
    scaler,
    windowing_func,
    output_suffix: str = "",
):
    """Run LSTM hyperparameter search with RAW (unwindowed) data."""

    cfg = LSTM_SEARCH_CONFIG

    print("Running LSTM search on RAW data shapes:", flush=True)
    print(f"  X_train_raw: {X_train_raw.shape}, y_train_raw: {y_train_raw.shape}", flush=True)
    print(f"  X_val_raw: {X_val_raw.shape}, y_val_raw: {y_val_raw.shape}", flush=True)
    print(f"  X_test_raw: {X_test_raw.shape}, y_test_raw: {y_test_raw.shape}", flush=True)
    print("  (Windowing will occur per-iteration based on seq parameter)", flush=True)

    all_results: list[dict[str, Any]] = []
    best_overall_score = float("inf")
    best_overall_params: dict[str, Any] | None = None

    from pathlib import Path

    try:
        from ashrae.ashrae_config import ASHRAE_RESULTS_ROOT
    except ImportError:
        ASHRAE_RESULTS_ROOT = Path("results/ashrae")

    results_dir = ASHRAE_RESULTS_ROOT / "lstm_search"
    results_dir.mkdir(parents=True, exist_ok=True)

    for algorithm_name in cfg["algorithms"]:
        print(f"\n{'='*80}", flush=True)
        print(f"Running {algorithm_name} optimization...", flush=True)
        print("=" * 80, flush=True)

        suffix_tag = output_suffix if output_suffix else ""
        results_csv = results_dir / f"search_iterations_{algorithm_name}{suffix_tag}.csv"
        print(f"  Results will be saved to: {results_csv}", flush=True)

        problem = _create_lstm_problem(
            X_train_raw,
            y_train_raw,
            X_val_raw,
            y_val_raw,
            X_test_raw,
            y_test_raw,
            windowing_func,
            scaler,
            cfg,
            results_csv_path=results_csv,
        )

        algorithm = _select_algorithm(algorithm_name, cfg["population_size"])
        task = Task(problem, max_iters=cfg["iterations"], optimization_type=OptimizationType.MINIMIZATION)

        best_solution, best_fitness = algorithm.run(task)
        best_params = _hyperparameters_from_vector(best_solution)
        best_eval = problem.get_best_evaluation()

        artifact_path: Path | None = None
        if best_eval is not None:
            artifact_filename = f"best_{algorithm_name}{suffix_tag}.obj"
            artifact_path = results_dir / artifact_filename
            try:
                save_object(best_eval, artifact_path)
                print(f"  ✓ Best artifact saved to {artifact_path}", flush=True)
            except Exception as exc:
                print(f"  ✗ Failed to save best artifact: {exc}", flush=True)
                artifact_path = None

        print(f"\n{algorithm_name} optimization complete!", flush=True)
        print(f"  Best params: {best_params}", flush=True)
        print(f"  Best fitness: {best_fitness:.6f}", flush=True)

        all_results.append(
            {
                "algorithm": algorithm_name,
                "best_params": best_params,
                "best_fitness": best_fitness,
                "convergence": task.convergence_data(),
                "artifact_path": artifact_path,
                "best_metrics": best_eval,
            }
        )

        if best_fitness < best_overall_score:
            best_overall_score = best_fitness
            best_overall_params = best_params

    return {
        "best_params": best_overall_params,
        "best_score": best_overall_score,
        "search_results": all_results,
        "scaler": scaler,
    }
def _create_lstm_problem(
    X_train_raw,
    y_train_raw,
    X_val_raw,
    y_val_raw,
    X_test_raw,
    y_test_raw,
    windowing_func,
    scaler,
    config,
    results_csv_path=None,
):
    """Create LSTM optimization problem with RAW data and dynamic windowing."""
    
    class LSTMProblemWithData(Problem):
        def __init__(self):
            super().__init__(dimension=4, lower=0, upper=1)
            self.X_train_raw = X_train_raw
            self.y_train_raw = y_train_raw
            self.X_val_raw = X_val_raw
            self.y_val_raw = y_val_raw
            self.X_test_raw = X_test_raw
            self.y_test_raw = y_test_raw
            self.windowing_func = windowing_func
            self.scaler = scaler
            self.config = config
            self.results_csv = results_csv_path
            self.eval_counter = 0
            self.eval_history: list[dict[str, Any]] = []
            self.best_eval: dict[str, Any] | None = None
        
        def _evaluate(self, x):
            params = _hyperparameters_from_vector(x)
            seq_len = params['seq']
            
            print(
                f"  Evaluating: units={params['units']} layers={params['num_layers']} "
                f"seq={seq_len} lr={params['learning_rate']:.5f}",
                flush=True,
            )
            
            # Apply sliding window with current seq parameter
            try:
                X_train, y_train, X_val, y_val, X_test, y_test = self.windowing_func(
                    self.X_train_raw, self.y_train_raw,
                    self.X_val_raw, self.y_val_raw,
                    self.X_test_raw, self.y_test_raw,
                    seq_length=seq_len
                )
            except Exception as e:
                print(f"    ✗ Windowing failed for seq={seq_len}: {e}", flush=True)
                return 1e6  # Return large loss on failure
            
            # Reshape targets for Keras
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            
            print(f"    Windowed shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}", flush=True)
            
            input_dim = (X_train.shape[1], X_train.shape[2])
            output_dim = 1
            
            model = _build_model(
                input_dim,
                output_dim,
                params["units"],
                params["num_layers"],
                params["learning_rate"],
            )
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.config["patience"],
                    restore_best_weights=True,
                ),
                ProgressPrinter(
                    step=50,
                    prefix=f"    [u={params['units']} l={params['num_layers']} s={seq_len}]",
                    total_epochs=self.config["num_epochs"],
                ),
            ]
            
            history = model.fit(
                X_train,
                y_train,
                epochs=self.config["num_epochs"],
                batch_size=2 ** self.config["batch_size_power"],
                verbose=0,
                shuffle=True,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )
            
            # Evaluate and compute metrics
            test_loss = model.evaluate(X_test, y_test, verbose=0)  # MSE on scaled targets
            y_pred = model.predict(X_test, verbose=0)

            # Scaled metrics (as before)
            mse_scaled = test_loss
            rmse_scaled = RMSE(y_test, y_pred)
            mae_scaled = MAE(y_test, y_pred)
            mape_scaled = MAPE(y_test, y_pred)
            r2_scaled = R2(y_test, y_pred)

            # Original-scale metrics using provided scaler (inverse transform)
            try:
                y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                y_pred_orig = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                rmse_orig = RMSE(y_test_orig, y_pred_orig)
                mae_orig = MAE(y_test_orig, y_pred_orig)
                mape_orig = MAPE(y_test_orig, y_pred_orig)
                r2_orig = R2(y_test_orig, y_pred_orig)
            except Exception as inv_exc:
                # Fallback to scaled metrics if inverse transform fails
                print(f"    ⚠️ Inverse transform failed, reporting scaled metrics only: {inv_exc}", flush=True)
                y_test_orig = None
                y_pred_orig = None
                rmse_orig = np.nan
                mae_orig = np.nan
                mape_orig = np.nan
                r2_orig = np.nan

            # Print metrics (both scales)
            print(f"    → Metrics:", flush=True)
            print(
                f"       SCALED  -> MSE: {mse_scaled:.6f}, RMSE: {rmse_scaled:.6f}, MAE: {mae_scaled:.6f}, MAPE: {mape_scaled:.2f}%, R²: {r2_scaled:.4f}",
                flush=True,
            )
            if y_test_orig is not None:
                print(
                    f"       ORIGINAL-> RMSE: {rmse_orig:.6f}, MAE: {mae_orig:.6f}, MAPE: {mape_orig:.2f}%, R²: {r2_orig:.4f}",
                    flush=True,
                )
            
            # Save results to CSV after each evaluation
            self.eval_counter += 1
            if self.results_csv:
                import pandas as pd
                import time
                from datetime import datetime

                # Derive epochs trained from history length
                try:
                    epochs_trained = len(history.history.get("loss", []))
                except Exception:
                    epochs_trained = np.nan

                result_row = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "eval_num": self.eval_counter,
                    "units": params["units"],
                    "num_layers": params["num_layers"],
                    "seq": seq_len,
                    "learning_rate": params["learning_rate"],
                    # Scaled metrics
                    "MSE_scaled": mse_scaled,
                    "RMSE_scaled": rmse_scaled,
                    "MAE_scaled": mae_scaled,
                    "MAPE_scaled": mape_scaled,
                    "R2_scaled": r2_scaled,
                    # Original-scale metrics
                    "RMSE_orig": rmse_orig,
                    "MAE_orig": mae_orig,
                    "MAPE_orig": mape_orig,
                    "R2_orig": r2_orig,
                    # Training meta
                    "epochs_trained": epochs_trained,
                }

                try:
                    df = pd.DataFrame([result_row])
                    # Robust header logic: write header if file is new or empty
                    header_needed = True
                    if self.results_csv.exists():
                        try:
                            header_needed = self.results_csv.stat().st_size == 0
                        except Exception:
                            header_needed = False
                    df.to_csv(self.results_csv, mode='a', header=header_needed, index=False)
                    print(f"    ✓ Results saved to {self.results_csv.name}", flush=True)
                except Exception as e:
                    print(f"    ✗ Failed to save results: {e}", flush=True)

            return test_loss
    
    return LSTMProblemWithData()


def _select_algorithm(name: str, population: int):
    if name == "Mod_FireflyAlgorithm":
        return Mod_FireflyAlgorithm(population_size=population)
    return FireflyAlgorithm(population_size=population)
