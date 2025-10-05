"""ASHRAE LSTM Training Script

This script trains LSTM models on the preprocessed ASHRAE dataset using
the same architecture and hyperparameters as the original Portuguese dataset.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam

from .preprocessing_ashrae import (
    load_ashrae_dataset,
    preprocess_ashrae_complete,
    get_ashrae_lstm_data,
)
from .ashrae_config import ASHRAE_TRAINING_CONFIG

from tools.preprocess_data2 import (
    RMSE,
    MAE,
    MAPE,
    RMSLE,
)


def build_lstm_model(units=72, num_layers=1, input_dim=(23, 32), learning_rate=0.01):
    """Build LSTM model with specified architecture."""
    model = Sequential(name="ASHRAE_LSTM")
    model.add(Input(shape=input_dim))
    
    return_sequences = num_layers > 1
    model.add(LSTM(units=units, return_sequences=return_sequences))
    
    for layer_idx in range(1, num_layers):
        model.add(LSTM(units=units, return_sequences=layer_idx < num_layers - 1))
    
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    
    return model


def evaluate_model(model, X_test, y_test, target_scaler):
    """Evaluate model and return metrics."""
    from sklearn.metrics import r2_score
    
    y_pred = model.predict(X_test, verbose=0)
    
    # Convert back to original scale using target scaler
    y_true_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    rmse = RMSE(y_true_orig, y_pred_orig)
    mae = MAE(y_true_orig, y_pred_orig)
    mape = MAPE(y_true_orig, y_pred_orig)
    rmsle = RMSLE(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    return y_true_orig, y_pred_orig, rmse, mae, mape, rmsle, r2


def train_ashrae_lstm():
    """Main training function for ASHRAE LSTM."""
    print("=" * 60)
    print("ASHRAE LSTM Training")
    print("=" * 60)
    
    # Load and preprocess data
    print("\n1. Loading and preprocessing ASHRAE data...")
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
    
    X_train, y_train, X_test, row_ids, target_scaler = preprocess_ashrae_complete(
        train_data, test_data, building_metadata, weather_train, weather_test
    )
    
    # Prepare LSTM data
    print("\n2. Preparing LSTM sequences...")
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = get_ashrae_lstm_data(
        X_train, y_train, X_test, 
        seq_length=ASHRAE_TRAINING_CONFIG["sequence_length"],
        max_samples=ASHRAE_TRAINING_CONFIG["max_samples"]
    )
    
    print(f"   ✓ Training data: {X_train_lstm.shape}")
    print(f"   ✓ Validation data: {X_val_lstm.shape}")
    print(f"   ✓ Test data: {X_test_lstm.shape}")
    
    # Build model
    print("\n3. Building LSTM model...")
    model = build_lstm_model(
        units=72,
        num_layers=1,
        input_dim=(23, 32),
        learning_rate=0.01
    )
    
    print(f"   ✓ Model architecture: {model.count_params()} parameters")
    model.summary()
    
    # Training configuration
    epochs = 50
    batch_size = 32
    patience = 10
    
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\n4. Training LSTM model...")
    print(f"   ✓ Epochs: {epochs}")
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Early stopping patience: {patience}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train_lstm, y_train_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    train_time = (time.time() - start_time) / 60
    
    # Evaluate model
    print(f"\n5. Evaluating model...")
    start_test = time.time()
    y_true, y_pred, rmse, mae, mape, rmsle, r2 = evaluate_model(model, X_test_lstm, y_test_lstm, target_scaler)
    test_time = time.time() - start_test
    
    # Results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training time: {train_time:.2f} minutes")
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Final epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print("\nTest Metrics:")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  R²:    {r2:.4f}")
    print(f"  RMSLE: {rmsle:.4f}")
    
    # Save model
    model_path = Path("results/ashrae_lstm_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return {
        'model': model,
        'history': history,
        'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'rmsle': rmsle},
        'times': {'train': train_time, 'test': test_time},
        'predictions': {'y_true': y_true, 'y_pred': y_pred}
    }


if __name__ == "__main__":
    results = train_ashrae_lstm()
    print("\n🎉 ASHRAE LSTM training completed successfully!")
