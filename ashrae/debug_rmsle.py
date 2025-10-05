"""Debug RMSLE calculation for ASHRAE dataset."""

import numpy as np
import pandas as pd
from pathlib import Path

from ashrae.preprocessing_ashrae import (
    load_ashrae_dataset,
    preprocess_ashrae_complete,
    get_ashrae_lstm_data,
    inverse_transform_ashrae_predictions,
    RMSLE,
)

def debug_rmsle():
    print("=" * 60)
    print("DEBUGGING RMSLE CALCULATION")
    print("=" * 60)
    
    # Load ASHRAE data
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
    
    # Preprocess data
    X_train, y_train, X_test, row_ids = preprocess_ashrae_complete(
        train_data, test_data, building_metadata, weather_train, weather_test
    )
    
    print(f"\n1. Original meter_reading statistics:")
    print(f"   Mean: {train_data.meter_reading.mean():.2f}")
    print(f"   Std:  {train_data.meter_reading.std():.2f}")
    print(f"   Min:  {train_data.meter_reading.min():.2f}")
    print(f"   Max:  {train_data.meter_reading.max():.2f}")
    
    print(f"\n2. Log-transformed y_train statistics:")
    print(f"   Mean: {y_train.mean():.4f}")
    print(f"   Std:  {y_train.std():.4f}")
    print(f"   Min:  {y_train.min():.4f}")
    print(f"   Max:  {y_train.max():.4f}")
    
    # Test inverse transformation
    print(f"\n3. Testing inverse transformation:")
    sample_log_values = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    original_values = inverse_transform_ashrae_predictions(sample_log_values)
    print(f"   Log values:    {sample_log_values}")
    print(f"   Original vals: {original_values}")
    
    # Test RMSLE with known values
    print(f"\n4. Testing RMSLE with known values:")
    # Simulate some predictions
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic meter readings (typical range: 0-1000)
    true_values = np.random.exponential(50, n_samples)  # Exponential distribution typical for energy data
    pred_values = true_values * (1 + np.random.normal(0, 0.1, n_samples))  # Add 10% noise
    
    # Ensure non-negative
    true_values = np.maximum(true_values, 0)
    pred_values = np.maximum(pred_values, 0)
    
    rmsle_test = RMSLE(true_values, pred_values)
    print(f"   True values mean: {true_values.mean():.2f}")
    print(f"   Pred values mean: {pred_values.mean():.2f}")
    print(f"   RMSLE: {rmsle_test:.4f}")
    
    # Test with log-transformed values
    print(f"\n5. Testing RMSLE with log-transformed values:")
    log_true = np.log1p(true_values)
    log_pred = np.log1p(pred_values)
    
    rmsle_log = RMSLE(log_true, log_pred)
    print(f"   Log true mean: {log_true.mean():.4f}")
    print(f"   Log pred mean: {log_pred.mean():.4f}")
    print(f"   RMSLE (log): {rmsle_log:.4f}")
    
    # Test with inverse-transformed values
    print(f"\n6. Testing RMSLE with inverse-transformed values:")
    inv_true = inverse_transform_ashrae_predictions(log_true)
    inv_pred = inverse_transform_ashrae_predictions(log_pred)
    
    rmsle_inv = RMSLE(inv_true, inv_pred)
    print(f"   Inv true mean: {inv_true.mean():.2f}")
    print(f"   Inv pred mean: {inv_pred.mean():.2f}")
    print(f"   RMSLE (inv): {rmsle_inv:.4f}")
    
    # Compare with manual RMSLE calculation
    print(f"\n7. Manual RMSLE calculation:")
    manual_rmsle = np.sqrt(np.mean((np.log1p(pred_values) - np.log1p(true_values)) ** 2))
    print(f"   Manual RMSLE: {manual_rmsle:.4f}")
    
    # Test with actual ASHRAE data
    print(f"\n8. Testing with actual ASHRAE data:")
    # Take a small sample
    sample_size = 1000
    y_sample = y_train.iloc[:sample_size]
    
    # Simulate predictions (add some noise to log-transformed values)
    y_pred_sample = y_sample + np.random.normal(0, 0.1, sample_size)
    
    # Calculate RMSLE on log scale
    rmsle_log_scale = RMSLE(y_sample.values, y_pred_sample)
    print(f"   RMSLE on log scale: {rmsle_log_scale:.4f}")
    
    # Calculate RMSLE on original scale
    y_orig = inverse_transform_ashrae_predictions(y_sample.values)
    y_pred_orig = inverse_transform_ashrae_predictions(y_pred_sample)
    rmsle_orig_scale = RMSLE(y_orig, y_pred_orig)
    print(f"   RMSLE on original scale: {rmsle_orig_scale:.4f}")
    
    print(f"\n" + "=" * 60)
    print("CONCLUSION:")
    print("RMSLE should be calculated on the ORIGINAL SCALE values,")
    print("not on log-transformed values!")
    print("=" * 60)

if __name__ == "__main__":
    debug_rmsle()
