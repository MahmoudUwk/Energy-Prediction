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
    # User-requested search space:
    # units in [32, 512], num_layers in [1, 4], seq in [3, 32], lr in [0.005, 0.02]
    units = int(32 + x[0] * (512 - 32))  # 32..512
    num_layers = int(1 + x[1] * (4 - 1))  # 1..4
    seq = int(3 + x[2] * (32 - 3))  # 3..32
    learning_rate = 0.005 + x[3] * 0.015  # 0.005..0.02
    # Clamp to ensure bounds
    units = max(32, min(units, 512))
    num_layers = max(1, min(num_layers, 4))
    seq = max(3, min(seq, 32))
    return {
        "units": units,
        "num_layers": num_layers,
        "seq": seq,
        "learning_rate": learning_rate,
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
    from ashrae.ashrae_config import ASHRAE_RESULTS_ROOT

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
            self.best_score: float = float("inf")

        def get_best_evaluation(self):
            return self.best_eval
        
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
            
            # Evaluate and compute metrics ON VALIDATION SET (objective)
            val_loss_scaled = model.evaluate(X_val, y_val, verbose=0)
            y_val_pred = model.predict(X_val, verbose=0)

            # Scaled metrics (validation)
            val_mse_scaled = val_loss_scaled
            val_rmse_scaled = RMSE(y_val, y_val_pred)
            val_mae_scaled = MAE(y_val, y_val_pred)
            val_mape_scaled = MAPE(y_val, y_val_pred)
            val_r2_scaled = R2(y_val, y_val_pred)

            # Original-scale metrics (validation)
            try:
                y_val_orig = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
                y_val_pred_orig = self.scaler.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
                val_rmse_orig = RMSE(y_val_orig, y_val_pred_orig)
                val_mae_orig = MAE(y_val_orig, y_val_pred_orig)
                val_mape_orig = MAPE(y_val_orig, y_val_pred_orig)
                val_r2_orig = R2(y_val_orig, y_val_pred_orig)
            except Exception as inv_exc:
                print(f"    ⚠️ Inverse transform failed on val set: {inv_exc}", flush=True)
                y_val_orig = None
                y_val_pred_orig = None
                val_rmse_orig = float(val_rmse_scaled)
                val_mae_orig = float(val_mae_scaled)
                val_mape_orig = float(val_mape_scaled)
                val_r2_orig = float(val_r2_scaled)

            # Print metrics (validation)
            print("    → Validation metrics:", flush=True)
            print(
                f"       SCALED  -> MSE: {val_mse_scaled:.6f}, RMSE: {val_rmse_scaled:.6f}, MAE: {val_mae_scaled:.6f}, MAPE: {val_mape_scaled:.2f}%, R²: {val_r2_scaled:.4f}",
                flush=True,
            )
            print(
                f"       ORIGINAL-> RMSE: {val_rmse_orig:.6f}, MAE: {val_mae_orig:.6f}, MAPE: {val_mape_orig:.2f}%, R²: {val_r2_orig:.4f}",
                flush=True,
            )

            # Fitness: minimize validation RMSE on original scale
            fitness = float(val_rmse_orig)

            # Track best evaluation
            if fitness < self.best_score:
                self.best_score = fitness
                self.best_eval = {
                    "params": params,
                    "seq": seq_len,
                    "val_metrics": {
                        "RMSE_scaled": float(val_rmse_scaled),
                        "MAE_scaled": float(val_mae_scaled),
                        "MAPE_scaled": float(val_mape_scaled),
                        "R2_scaled": float(val_r2_scaled),
                        "RMSE_orig": float(val_rmse_orig),
                        "MAE_orig": float(val_mae_orig),
                        "MAPE_orig": float(val_mape_orig),
                        "R2_orig": float(val_r2_orig),
                    },
                }
            
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
                    # Validation scaled metrics
                    "VAL_MSE_scaled": val_mse_scaled,
                    "VAL_RMSE_scaled": val_rmse_scaled,
                    "VAL_MAE_scaled": val_mae_scaled,
                    "VAL_MAPE_scaled": val_mape_scaled,
                    "VAL_R2_scaled": val_r2_scaled,
                    # Validation original-scale metrics
                    "VAL_RMSE_orig": val_rmse_orig,
                    "VAL_MAE_orig": val_mae_orig,
                    "VAL_MAPE_orig": val_mape_orig,
                    "VAL_R2_orig": val_r2_orig,
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

            return fitness
    
    return LSTMProblemWithData()


def _select_algorithm(name: str, population: int):
    if name == "Mod_FireflyAlgorithm":
        return Mod_FireflyAlgorithm(population_size=population)
    return FireflyAlgorithm(population_size=population)
