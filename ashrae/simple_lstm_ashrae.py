from __future__ import annotations

import time
from pathlib import Path
import numpy as np
from keras.layers import Dense, LSTM, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def main():

    print("=" * 80)
    print("SIMPLE LSTM ASHRAE DATASET TRAINING")
    print("=" * 80)

    from .preprocessing_ashrae_disjoint import (
        preprocess_ashrae_disjoint_splits,
        get_ashrae_lstm_data_disjoint,
    )
    from .save_ashrae_results import save_ashrae_lstm_results

    # Load ASHRAE data - same as RFR
    print("Loading ASHRAE dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    # Create LSTM sequences (3D format) - same as RFR preprocessing
    seq_length = 7  # Standardized sequence length
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_length
    )

    print(f"3D Data shapes: Train={X_tr_lstm.shape}, Val={X_va_lstm.shape}, Test={X_te_lstm.shape}")

    # Combine train and validation for final training (same as RFR)
    X_train_all = np.vstack([X_tr_lstm, X_va_lstm])
    y_train_all = np.hstack([y_tr_lstm, y_va_lstm])

    print(f"Combined training data: {X_train_all.shape}")

    # Simple LSTM hyperparameters for benchmark
    units = 15
    num_layers = 1
    learning_rate = 0.001
    epochs = 50  # Reduced for faster benchmark
    batch_size = 2048

    # Build simple LSTM model
    input_dim = (X_train_all.shape[1], X_train_all.shape[2])
    output_dim = 1

    print(f"Building LSTM: input_shape={input_dim}, units={units}, layers={num_layers}, lr={learning_rate}")

    model = Sequential(name="Simple_LSTM")
    model.add(Input(shape=input_dim))

    if num_layers == 1:
        model.add(LSTM(units=units, return_sequences=False))
    else:
        model.add(LSTM(units=units, return_sequences=True))
        for i in range(1, num_layers):
            model.add(LSTM(units=units, return_sequences=(i < num_layers - 1)))
        model.add(Dense(units))

    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    print(f"Model summary:")
    model.summary()

    # Training callbacks
    callbacks = [
        EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # Train the model
    print(f"\nTraining LSTM for {epochs} epochs...")
    start_time = time.time()

    history = model.fit(
        X_train_all,
        y_train_all.reshape(-1, 1),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.0,  # We already have separate val set
        callbacks=callbacks,
        verbose=1,  # Show progress
    )

    training_time = (time.time() - start_time) / 60.0
    print(f"Training completed in {training_time:.2f} minutes")

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    start_test = time.time()
    y_test_pred_scaled = model.predict(X_te_lstm, verbose=0)
    test_time = time.time() - start_test

    # Convert back to original scale
    y_test_true = scaler.inverse_transform(y_te_lstm.reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    # Calculate metrics
    test_rmse = float(np.sqrt(np.mean((y_test_true - y_test_pred) ** 2)))
    test_mae = float(np.mean(np.abs(y_test_true - y_test_pred)))
    test_r2 = float(1 - np.sum((y_test_true - y_test_pred) ** 2) / np.sum((y_test_true - y_test_true.mean()) ** 2))

    # MAPE calculation (avoiding division by zero)
    _mask_test = np.abs(y_test_true) > 1.0
    test_mape = float(
        100.0
        * np.mean(
            np.abs((y_test_true[_mask_test] - y_test_pred[_mask_test]) / y_test_true[_mask_test])
        )
    ) if np.any(_mask_test) else float("inf")

    # Results
    metrics = {
        "RMSE": test_rmse,
        "MAE": test_mae,
        "R2": test_r2,
        "MAPE": test_mape,
    }

    print("=" * 80)
    print("SIMPLE LSTM RESULTS")
    print("=" * 80)
    print(f"Test Metrics: {metrics}")
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Test time: {test_time:.2f} seconds")

    # Save results using same format as other models
    best_params = {
        "units": units,
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "sequence_length": seq_length,
    }
    
    saved = save_ashrae_lstm_results(
        metrics=metrics,
        y_true=y_test_true,
        y_pred=y_test_pred,
        best_params=best_params,
        best_epoch=len(history.history['loss']),
        algorithm="Simple_LSTM",
        datatype="ASHRAE",
        train_time_min=training_time,
        test_time_s=test_time,
    )

    print(f"Results saved: {saved}")


if __name__ == "__main__":
    main()
