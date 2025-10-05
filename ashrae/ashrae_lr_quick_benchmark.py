"""Quick ASHRAE Linear Regression Benchmark with Sample Data."""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

# Add current directory to path for imports
sys.path.append('.')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from ashrae.preprocessing_ashrae import load_ashrae_dataset, preprocess_ashrae_complete
from ashrae_config import ASHRAE_TRAINING_CONFIG, ASHRAE_DATA_SPLITS


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def root_mean_squared_logarithmic_error(y_true, y_pred):
    """Calculate RMSLE."""
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def calculate_metrics(y_true, y_pred):
    """Calculate all metrics."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'RMSLE': root_mean_squared_logarithmic_error(y_true, y_pred)
    }


def main():
    """Run quick Linear Regression benchmark."""
    print("=" * 60)
    print("ASHRAE LINEAR REGRESSION QUICK BENCHMARK")
    print("=" * 60)
    
    # Load small sample of data
    print("\n1. Loading ASHRAE dataset...")
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    try:
        train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
        
        # Use very small sample for quick testing
        small_train = train_data.head(50000)  # 50K rows
        print(f"   ✓ Using sample: {len(small_train):,} training rows")
        
        # Preprocess
        print("\n2. Preprocessing data...")
        X_train, y_train, X_test, row_ids = preprocess_ashrae_complete(
            small_train, test_data, building_metadata, weather_train, weather_test
        )
        
        print(f"   ✓ Features shape: {X_train.shape}")
        print(f"   ✓ Target shape: {y_train.shape}")
        
        # Use only last timestep for Linear Regression (no sequences)
        X_train_lr = X_train.values
        y_train_lr = y_train.values
        X_test_lr = X_test.values[:len(y_train)]  # Match sizes
        
        # Get target scaler
        target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        target_scaler.fit(y_train_lr.reshape(-1, 1))
        
        print(f"   ✓ LR data shapes - Train: {X_train_lr.shape}, Test: {X_test_lr.shape}")
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False
    
    # Train model
    print("\n3. Training Linear Regression...")
    model = LinearRegression(n_jobs=-1)
    
    start_time = time.time()
    model.fit(X_train_lr, y_train_lr)
    train_time = time.time() - start_time
    
    print(f"   ✓ Training time: {train_time:.2f} seconds")
    
    # Evaluate
    print("\n4. Evaluating model...")
    start_time = time.time()
    y_pred_scaled = model.predict(X_test_lr)
    pred_time = time.time() - start_time
    
    print(f"   ✓ Prediction time: {pred_time:.2f} seconds")
    
    # Inverse transform
    y_true_orig = target_scaler.inverse_transform(y_train_lr.reshape(-1, 1)).flatten()
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_orig, y_pred_orig)
    
    print("\n📊 RESULTS:")
    print(f"   MSE:    {metrics['MSE']:.4f}")
    print(f"   RMSE:   {metrics['RMSE']:.4f}")
    print(f"   MAPE:   {metrics['MAPE']:.2f}%")
    print(f"   R2:     {metrics['R2']:.4f}")
    print(f"   RMSLE:  {metrics['RMSLE']:.4f}")
    
    print(f"\n⏱️  Timing:")
    print(f"   Training:    {train_time:.2f}s")
    print(f"   Prediction:  {pred_time:.2f}s")
    
    # Save results
    results_dir = Path("results/ashrae/linear_regression")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    pd.DataFrame([metrics]).to_csv(results_dir / "quick_metrics.csv", index=False)
    print(f"\n📁 Results saved to: {results_dir / 'quick_metrics.csv'}")
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Quick Linear Regression benchmark completed!")
    else:
        print("\n💥 Benchmark failed!")
        sys.exit(1)
