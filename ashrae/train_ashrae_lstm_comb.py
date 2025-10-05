"""ASHRAE LSTM Training using models/LSTM_comb.py

This script adapts the existing LSTM_comb.py model for ASHRAE dataset training.
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

from tools.preprocess_data2 import (
    RMSE,
    MAE,
    MAPE,
    RMSLE,
)

# Import functions from LSTM_comb.py
from models.LSTM_comb import (
    _build_callbacks,
    _build_model,
    _best_epoch,
    _persist_results,
)


def create_ashrae_scaler_wrapper(target_scaler):
    """Wrap target_scaler to match expected interface for inverse_transf."""
    class ASHRAEScalerWrapper:
        def __init__(self, scaler):
            # Copy attributes for inverse_transf compatibility
            self.data_min_ = scaler.data_min_
            self.data_max_ = scaler.data_max_
            self._scaler = scaler
        
        def inverse_transform(self, data):
            # Use the actual target scaler
            return self._scaler.inverse_transform(data.reshape(-1, 1)).flatten()
    
    return ASHRAEScalerWrapper(target_scaler)


def adapt_ashrae_data_for_lstm_comb():
    """Adapt ASHRAE data to match LSTM_comb.py interface."""
    print("=" * 60)
    print("ASHRAE LSTM Training using LSTM_comb.py")
    print("=" * 60)
    
    # Load and preprocess ASHRAE data
    print("\n1. Loading and preprocessing ASHRAE data...")
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
    
    X_train, y_train, X_test, row_ids, target_scaler = preprocess_ashrae_complete(
        train_data, test_data, building_metadata, weather_train, weather_test
    )
    
    # Prepare LSTM data
    print("\n2. Preparing LSTM sequences...")
    from .ashrae_config import ASHRAE_TRAINING_CONFIG
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = get_ashrae_lstm_data(
        X_train, y_train, X_test, 
        seq_length=ASHRAE_TRAINING_CONFIG["sequence_length"],
        max_samples=ASHRAE_TRAINING_CONFIG["max_samples"]
    )
    
    print(f"   ✓ Training data: {X_train_lstm.shape}")
    print(f"   ✓ Validation data: {X_val_lstm.shape}")
    print(f"   ✓ Test data: {X_test_lstm.shape}")
    
    # Create scaler wrapper for interface compatibility
    scaler = create_ashrae_scaler_wrapper(target_scaler)
    
    # Prepare data in the format expected by LSTM_comb.py
    # Reshape targets to match expected format (add feature dimension)
    y_train_reshaped = y_train_lstm.reshape(-1, 1)
    y_val_reshaped = y_val_lstm.reshape(-1, 1)
    y_test_reshaped = y_test_lstm.reshape(-1, 1)
    
    return (
        X_train_lstm, y_train_reshaped,
        X_val_lstm, y_val_reshaped,
        X_test_lstm, y_test_reshaped,
        "results/ashrae",  # save_path_str
        None,  # test_time_axis (not used)
        scaler
    )


def train_ashrae_with_lstm_comb():
    """Train ASHRAE LSTM using the LSTM_comb.py framework."""
    
    # Get data in LSTM_comb format
    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        save_path_str,
        _,
        scaler,
    ) = adapt_ashrae_data_for_lstm_comb()
    
    # Configuration matching LSTM_comb.py expectations
    cfg = {
        "algorithm": "ASHRAE_LSTM",
        "epochs": 50,
        "learning_rate": 0.01,
        "batch_size_power": 5,  # 2^5 = 32 batch size
        "verbose": 1,
        "patience": 10,
        "use_callbacks": True,
        "plot_model": False,
        "model_plot_filename": "ashrae_model.png",
        "persist_models": True,
        "save_results": True,
    }
    
    # Model parameters
    units = 72
    num_layers = 1
    input_dim = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[-1]
    
    print(f"\n3. Building LSTM model using LSTM_comb.py...")
    print(f"   ✓ Units: {units}")
    print(f"   ✓ Layers: {num_layers}")
    print(f"   ✓ Input dim: {input_dim}")
    print(f"   ✓ Output dim: {output_dim}")
    
    # Build model using LSTM_comb function
    model = _build_model(
        units=units,
        num_layers=num_layers,
        input_dim=input_dim,
        output_dim=output_dim,
        learning_rate=cfg["learning_rate"],
        plot_enabled=cfg["plot_model"],
        plot_filename=cfg["model_plot_filename"]
    )
    
    print(f"   ✓ Model architecture: {model.count_params()} parameters")
    
    # Build callbacks
    callbacks = _build_callbacks(cfg)
    
    # Training configuration
    batch_size = 2 ** cfg["batch_size_power"]
    
    print(f"\n4. Training LSTM model...")
    print(f"   ✓ Epochs: {cfg['epochs']}")
    print(f"   ✓ Batch size: {batch_size}")
    print(f"   ✓ Early stopping patience: {cfg['patience']}")
    
    start_time = time.time()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=cfg["verbose"],
        shuffle=True
    )
    
    train_time = (time.time() - start_time) / 60
    
    # Evaluate model using ASHRAE evaluation
    print(f"\n5. Evaluating model...")
    start_test = time.time()
    
    # Get predictions in normalized scale
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Convert to original scale using target scaler
    y_true_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics on original scale
    from sklearn.metrics import r2_score
    rmse = RMSE(y_true_orig, y_pred_orig)
    mae = MAE(y_true_orig, y_pred_orig)
    mape = MAPE(y_true_orig, y_pred_orig)
    rmsle = RMSLE(y_true_orig, y_pred_orig)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    test_time = time.time() - start_test
    
    # Get best epoch
    best_epoch = _best_epoch(history)
    
    # Results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training time: {train_time:.2f} minutes")
    print(f"Test time: {test_time:.2f} seconds")
    print(f"Best epoch: {best_epoch}")
    print(f"Final epoch: {len(history.history['loss'])}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print("\nTest Metrics:")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  MAE:   {mae:.4f}")
    print(f"  MAPE:  {mape:.2f}%")
    print(f"  R²:    {r2:.4f}")
    print(f"  RMSLE: {rmsle:.4f}")
    
    # Save results using LSTM_comb function
    save_path = Path(save_path_str)
    save_path.mkdir(parents=True, exist_ok=True)
    
    _persist_results(
        save_path=save_path,
        datatype_opt="ASHRAE",
        alg_name=cfg["algorithm"],
        seq=23,
        num_layers=num_layers,
        units=units,
        best_epoch=best_epoch,
        used_features=X_train.shape[2],
        train_time=train_time,
        test_time=test_time,
        rmse=rmse,
        mae=mae,
        mape=mape,
        y_true=y_true_orig,
        y_pred=y_pred_orig,
    )
    
    # Save model
    model_path = save_path / "ashrae_lstm_model.keras"
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    return {
        'model': model,
        'history': history,
        'metrics': {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2, 'rmsle': rmsle},
        'times': {'train': train_time, 'test': test_time},
        'predictions': {'y_true': y_true_orig, 'y_pred': y_pred_orig}
    }


if __name__ == "__main__":
    results = train_ashrae_with_lstm_comb()
    print("\n🎉 ASHRAE LSTM training with LSTM_comb.py completed successfully!")
