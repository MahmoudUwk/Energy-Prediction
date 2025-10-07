from __future__ import annotations

import time
import numpy as np
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def main() -> None:
    print("=" * 80)
    print("LSTM-FF (Firefly) ASHRAE DATASET TRAINING - Using best tuned parameters")
    print("=" * 80)

    # Local imports to keep relative paths
    from .preprocessing_ashrae_disjoint import (
        preprocess_ashrae_disjoint_splits,
        get_ashrae_lstm_data_disjoint,
    )
    from .save_ashrae_results import get_ashrae_results_saver

    # Load ASHRAE data
    print("Loading ASHRAE dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    # Best tuned hyperparameters from search artifacts
    units = 219
    num_layers = 1
    learning_rate = 0.005
    seq_length = 27
    epochs = 300
    batch_size = 2048

    # Build LSTM windows (3D)
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_length
    )

    print(
        f"3D Data shapes: Train={X_tr_lstm.shape}, Val={X_va_lstm.shape}, Test={X_te_lstm.shape}"
    )

    # Keep train/val split for proper validation-based callbacks

    # Build model
    input_shape = (X_tr_lstm.shape[1], X_tr_lstm.shape[2])
    print(
        f"Building LSTM-FF: input_shape={input_shape}, units={units}, layers={num_layers}, lr={learning_rate}"
    )
    model = Sequential(name="LSTM_FF_Final")
    model.add(Input(shape=input_shape))
    if num_layers == 1:
        model.add(LSTM(units=units, return_sequences=False))
    else:
        model.add(LSTM(units=units, return_sequences=True))
        for i in range(1, num_layers):
            model.add(LSTM(units=units, return_sequences=(i < num_layers - 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    ]

    # Train
    print(f"\nTraining for {epochs} epochs...")
    start_train = time.time()
    history = model.fit(
        X_tr_lstm,
        y_tr_lstm.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_va_lstm, y_va_lstm.reshape(-1, 1)),
        callbacks=callbacks,
        verbose=1,
    )
    train_time_min = (time.time() - start_train) / 60.0
    print(f"Training completed in {train_time_min:.2f} minutes")

    # Test
    print("\nEvaluating on test set...")
    start_test = time.time()
    y_test_pred_scaled = model.predict(X_te_lstm, verbose=0)
    test_time_s = time.time() - start_test

    # Inverse transform
    y_test_true = scaler.inverse_transform(y_te_lstm.reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = np.maximum(y_test_pred, 0.0)

    # Metrics (original scale)
    rmse = float(np.sqrt(np.mean((y_test_true - y_test_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test_true - y_test_pred)))
    r2 = float(
        1.0
        - (np.sum((y_test_true - y_test_pred) ** 2) / np.sum((y_test_true - y_test_true.mean()) ** 2))
    )
    mask = np.abs(y_test_true) > 1.0
    mape = float(
        100.0
        * np.mean(np.abs((y_test_true[mask] - y_test_pred[mask]) / y_test_true[mask]))
    ) if np.any(mask) else float("inf")

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    print("=" * 80)
    print("LSTM-FF RESULTS (Test)")
    print("=" * 80)
    print(metrics)
    print(f"Training time: {train_time_min:.2f} minutes | Test time: {test_time_s:.2f} seconds")

    # Save in the same place/format used by plotting (lstm_search/artifacts/LSTM-FF.obj)
    saver = get_ashrae_results_saver("lstm_search", "LSTM-FF")
    # Determine best epoch if validation loss is available
    best_epoch = int(np.argmin(history.history.get("val_loss", [np.inf])) + 1) if history.history.get("val_loss") else int(len(history.history.get("loss", [])))

    model_info = {
        "best_params": {
            "units": units,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "seq": seq_length,
        },
        "sequence_length": seq_length,
        "algorithm_name": "FireflyAlgorithm",
        "epochs_trained": int(len(history.history.get("loss", []))),
        "best_epoch": best_epoch,
        "best_val_loss": float(np.min(history.history.get("val_loss", [np.nan]))) if history.history.get("val_loss") else None,
    }

    saved = saver.save_all(
        metrics=metrics,
        y_true=y_test_true,
        y_pred=y_test_pred,
        model_info=model_info,
        train_time_min=train_time_min,
        test_time_s=test_time_s,
    )

    print(f"Saved: {saved}")


if __name__ == "__main__":
    main()


