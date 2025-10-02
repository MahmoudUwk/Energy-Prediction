from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from niapy.algorithms.basic import FireflyAlgorithm
from niapy.algorithms.modified import Mod_FireflyAlgorithm
from niapy.problems import Problem
from niapy.task import OptimizationType, Task

from config import LSTM_SEARCH_CONFIG
from preprocess_data2 import (
    MAE,
    MAPE,
    RMSE,
    ensure_three_dim,
    expand_dims,
    get_SAMFOR_data,
    inverse_transf,
    log_results_LSTM,
    prepare_lstm_targets,
    save_object,
    subset_lstm_features,
    loadDatasetObj,
)

# from niapy.algorithms.basic import BeesAlgorithm

def _hyperparameters_from_vector(x: np.ndarray) -> Dict[str, Any]:
    return {
        "units": int(x[0] * 116 + 10),
        "num_layers": int(x[1] * 6) + 1,
        "seq": int(x[2] * 30 + 1),
        "learning_rate": x[3] * 2e-2 + 0.5e-3,
    }


def _build_model(input_dim, output_dim, units, num_layers, learning_rate):
    model = Sequential(name="LSTM_HP")
    model.add(LSTM(units=units, input_shape=input_dim, return_sequences=num_layers > 1))
    for idx in range(1, num_layers):
        model.add(LSTM(units=units, return_sequences=idx < num_layers - 1))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def _prepare_data(option, datatype_opt, seq):
    return get_SAMFOR_data(option, datatype_opt, seq)


class LSTMHyperparameterOptimization(Problem):
    def __init__(self, config, datatype_opt):
        super().__init__(dimension=4, lower=0, upper=1)
        self.config = config
        self.datatype_opt = datatype_opt

    def _evaluate(self, x):
        params = _hyperparameters_from_vector(x)
        data = _prepare_data(self.config["option"], self.datatype_opt, params["seq"])
        X_train, y_train, X_val, y_val, X_test, y_test, _, _, scaler = data

        input_dim = (X_train.shape[1], X_train.shape[2])
        output_dim = 1
        y_train = prepare_lstm_targets(y_train)
        y_test = prepare_lstm_targets(y_test)

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
            )
        ]

        model.fit(
            X_train,
            y_train,
            epochs=self.config["num_epochs"],
            batch_size=2 ** self.config["batch_size_power"],
            verbose=0,
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

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["patience"],
            restore_best_weights=True,
        )
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

    y_test = inverse_transf(y_test, scaler)
    y_pred = inverse_transf(model.predict(X_test), scaler)

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
    save_object({"y_test": y_test, "y_test_pred": y_pred}, save_path / f"{algorithm_name}.obj")

    print(f"{algorithm_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")


def main():
    cfg = LSTM_SEARCH_CONFIG
    option = cfg["option"]

    for datatype_opt in cfg["datatype_options"]:
        for alg_name in cfg["algorithms"]:
            algorithm = _select_algorithm(alg_name, cfg["population_size"])

            base_save_path = Path(get_SAMFOR_data(0, datatype_opt, 0, 1))

            if cfg["run_search"]:
                problem = LSTMHyperparameterOptimization(cfg, datatype_opt)
                task = Task(problem, max_iters=cfg["iterations"], optimization_type=OptimizationType.MINIMIZATION)

                best_vector, best_mse = algorithm.run(task)
                best_params = _hyperparameters_from_vector(best_vector)
                print(f"Best parameters for {datatype_opt} using {alg_name}: {best_params}")

                _save_artifacts(task, best_params, base_save_path, alg_name[0], cfg)

            best_payload = loadDatasetObj(base_save_path / f"Best_param{alg_name[0]}.obj")
            _train_with_best_params(cfg, datatype_opt, best_payload["best_params"], alg_name[0])


if __name__ == "__main__":
    main()
