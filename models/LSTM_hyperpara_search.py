from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
    ensure_three_dim,
    expand_dims,
    log_results_LSTM,
    prepare_lstm_targets,
    subset_lstm_features,
    loadDatasetObj,
    save_object,
)

# Import ASHRAE results saver for consistent saving
try:
    from ashrae.save_ashrae_results import save_ashrae_lstm_results
except ImportError:
    # Fallback if running as script
    import sys
    from pathlib import Path
    ashrae_path = Path(__file__).parent.parent / "ashrae"
    if str(ashrae_path) not in sys.path:
        sys.path.insert(0, str(ashrae_path))
    from save_ashrae_results import save_ashrae_lstm_results

# Remove ASHRAE-specific imports - dataset loading should be handled by callers

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


def run_lstm_search(X_train, y_train, X_val, y_val, X_test, y_test, scaler, seq_length):
    """
    Run LSTM hyperparameter search with pre-loaded data.
    
    Args:
        X_train: Training features (3D array: samples, timesteps, features)
        y_train: Training targets (2D array: samples, 1)
        X_val: Validation features (3D array)
        y_val: Validation targets (2D array)
        X_test: Test features (3D array)
        y_test: Test targets (2D array)
        scaler: Scaler object for inverse transformation
        seq_length: Sequence length
        
    Returns:
        dict: Search results including best parameters and metrics
    """
    cfg = LSTM_SEARCH_CONFIG
    
    print(f"Running LSTM search on data shapes:", flush=True)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}", flush=True)
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}", flush=True)
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}", flush=True)
    
    # Ensure targets are reshaped for Keras compatibility
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    # Run the search for each algorithm
    all_results = []
    best_overall_score = float('inf')
    best_overall_params = None
    
    for algorithm_name in cfg["algorithms"]:
        print(f"\n{'='*80}", flush=True)
        print(f"Running {algorithm_name} optimization...", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Create optimizer-specific problem wrapper
        problem = _create_lstm_problem(X_train, y_train, X_val, y_val, X_test, y_test, cfg)
        
        # Select and run algorithm
        algorithm = _select_algorithm(algorithm_name, cfg["population_size"])
        task = Task(problem, max_iters=cfg["iterations"], optimization_type=OptimizationType.MINIMIZATION)
        
        best_solution, best_fitness = algorithm.run(task)
        best_params = _hyperparameters_from_vector(best_solution)
        
        print(f"\n{algorithm_name} optimization complete!", flush=True)
        print(f"  Best params: {best_params}", flush=True)
        print(f"  Best fitness: {best_fitness:.6f}", flush=True)
        
        all_results.append({
            "algorithm": algorithm_name,
            "best_params": best_params,
            "best_fitness": best_fitness,
            "convergence": task.convergence_data()
        })
        
        if best_fitness < best_overall_score:
            best_overall_score = best_fitness
            best_overall_params = best_params
    
    return {
        "best_params": best_overall_params,
        "best_score": best_overall_score,
        "search_results": all_results,
        "scaler": scaler
    }


def _create_lstm_problem(X_train, y_train, X_val, y_val, X_test, y_test, config):
    """Create LSTM optimization problem with pre-loaded data."""
    
    class LSTMProblemWithData(Problem):
        def __init__(self):
            super().__init__(dimension=4, lower=0, upper=1)
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.X_test = X_test
            self.y_test = y_test
            self.config = config
        
        def _evaluate(self, x):
            params = _hyperparameters_from_vector(x)
            print(
                f"  Evaluating: units={params['units']} layers={params['num_layers']} "
                f"seq={params['seq']} lr={params['learning_rate']:.5f}",
                flush=True,
            )
            
            # Note: seq parameter is ignored here since data is pre-windowed
            # In future, could re-window data dynamically if needed
            
            input_dim = (self.X_train.shape[1], self.X_train.shape[2])
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
                    prefix=f"    [u={params['units']} l={params['num_layers']}]",
                    total_epochs=self.config["num_epochs"],
                ),
            ]
            
            model.fit(
                self.X_train,
                self.y_train,
                epochs=self.config["num_epochs"],
                batch_size=2 ** self.config["batch_size_power"],
                verbose=0,
                shuffle=True,
                validation_data=(self.X_val, self.y_val),
                callbacks=callbacks,
            )
            
            test_loss = model.evaluate(self.X_test, self.y_test, verbose=0)
            print(f"    → Test loss: {test_loss:.6f}", flush=True)
            return test_loss
    
    return LSTMProblemWithData()


def _prepare_data(option, datatype_opt, seq):
    """Legacy function - now redirects to caller scripts."""
    raise NotImplementedError(
        "LSTM _prepare_data() should be called with pre-loaded data. "
        "Use ashrae/call_lstm_search_ashrae.py for ASHRAE dataset."
    )


class LSTMHyperparameterOptimization(Problem):
    def __init__(self, config, datatype_opt):
        super().__init__(dimension=4, lower=0, upper=1)
        self.config = config
        self.datatype_opt = datatype_opt

    def _evaluate(self, x):
        params = _hyperparameters_from_vector(x)
        print(
            f"Params: units={params['units']} layers={params['num_layers']} seq={params['seq']} lr={params['learning_rate']:.5f}",
            flush=True,
        )
        data = _prepare_data(self.config["option"], self.datatype_opt, params["seq"])
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, scaler = data

        input_dim = (X_train.shape[1], X_train.shape[2])
        output_dim = 1
        y_train = prepare_lstm_targets(y_train)
        y_test = prepare_lstm_targets(y_test)
        y_val = prepare_lstm_targets(y_val)

        # Ensure targets are shape (n, 1)
        y_train = np.reshape(y_train, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))

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
                step=20,
                prefix=f"[seq={params['seq']} layers={params['num_layers']} units={params['units']}]",
                total_epochs=self.config["num_epochs"],
            ),
        ]

        model.fit(
            X_train,
            y_train,
            epochs=self.config["num_epochs"],
            batch_size=2 ** self.config["batch_size_power"],
            verbose=1,
            shuffle=True,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
        )

        return model.evaluate(X_test, y_test, verbose=0)


def _select_algorithm(name: str, population: int):
    if name == "Mod_FireflyAlgorithm":
        return Mod_FireflyAlgorithm(population_size=population)
    return FireflyAlgorithm(population_size=population)


def _save_artifacts(task: Task, best_params: Dict[str, Any], save_path: Path, prefix: str, config):
    data = task.convergence_data()
    evals = task.convergence_data(x_axis="evals")
    payload = {
        "iterations": data,
        "evaluations": evals,
        "best_params": best_params,
    }
    # Append or update best payload to allow accumulation across runs
    save_object(payload, save_path / f"Best_param{prefix}.obj")

    if config["plot_convergence"]:
        task.plot_convergence(x_axis="evals")
        plt.savefig(save_path / f"Conv_FF_eval{prefix}.png")
        plt.close()

        task.plot_convergence()
        plt.savefig(save_path / f"Conv_FF_itr{prefix}.png")
        plt.close()


def _train_with_best_params(config, datatype_opt, best_params, algorithm_name):
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        save_path_str,
        _,
        scaler,
    ) = _prepare_data(config["option"], datatype_opt, best_params["seq"])

    # For ASHRAE, y_train etc. are already scaled [0,1], no need for inverse_transf during training
    # But we need to inverse transform for metrics calculation
    pass

    y_train = prepare_lstm_targets(y_train)
    input_dim = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[-1]

    model = _build_model(
        input_dim,
        output_dim,
        best_params["units"],
        best_params["num_layers"],
        best_params["learning_rate"],
    )

    print(
        f"Best Params: units={best_params['units']} layers={best_params['num_layers']} seq={best_params['seq']} lr={best_params['learning_rate']:.5f}"
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True,
        ),
        ProgressPrinter(
            step=20,
            prefix=f"[BEST seq={best_params['seq']} layers={best_params['num_layers']} units={best_params['units']}]",
            total_epochs=config["num_epochs"],
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=config["num_epochs"],
        batch_size=2 ** (config["batch_size_power"] - 1),
        verbose=1,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    save_path = Path(save_path_str)
    if config["persist_models"]:
        model.save(save_path / f"{algorithm_name}_{datatype_opt}.keras")

    best_epoch = int(np.argmin(history.history["val_loss"]))

    # For ASHRAE, use the target_scaler from preprocessing to inverse transform
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    y_test, y_pred = y_test_orig, y_pred_orig

    rmse = RMSE(y_test, y_pred)
    mae = MAE(y_test, y_pred)
    mape = MAPE(y_test, y_pred)

    row = [
        algorithm_name,
        rmse,
        mae,
        mape,
        best_params["seq"],
        best_params["num_layers"],
        best_params["units"],
        best_epoch,
        datatype_opt,
        0,
        0,
        "all",
    ]

    log_results_LSTM(row, datatype_opt, str(save_path))
    # Save results using centralized ASHRAE saver
    saved_files = save_ashrae_lstm_results(
        metrics={"RMSE": rmse, "MAE": mae, "MAPE": mape},
        y_true=y_test,
        y_pred=y_pred,
        best_params=best_params,
        best_epoch=best_epoch,
        algorithm=algorithm_name,
        datatype=datatype_opt,
        train_time_min=0,
        test_time_s=0,
    )

    print(f"{algorithm_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")


def main():
    """Legacy main function - now redirects to caller scripts."""
    raise NotImplementedError(
        "LSTM main() should be called with pre-loaded data. "
        "Use ashrae/call_lstm_search_ashrae.py for ASHRAE dataset."
    )


if __name__ == "__main__":
    main()
