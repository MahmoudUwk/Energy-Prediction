"""ASHRAE Linear Regression Synthetic Benchmark for Pipeline Verification."""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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


def generate_synthetic_ashrae_data(n_samples=10000, n_features=30):
    """Generate synthetic ASHRAE-like data for testing."""
    print("🔧 Generating synthetic ASHRAE data...")
    
    np.random.seed(42)
    
    # Generate features (normalized to [0,1])
    X = np.random.rand(n_samples, n_features)
    
    # Generate target (electricity consumption in kWh, then normalize)
    # Create realistic electricity consumption patterns
    base_consumption = 50 + 100 * np.random.rand(n_samples)  # 50-150 kWh
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)  # Daily pattern
    weather_factor = 1 + 0.2 * np.random.rand(n_samples)  # Weather influence
    
    y_true = base_consumption * seasonal_factor * weather_factor
    
    # Add some noise
    noise = np.random.normal(0, 10, n_samples)
    y_true = np.maximum(y_true + noise, 0)  # Ensure non-negative
    
    # Normalize target to [0,1]
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    y_scaled = target_scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
    
    print(f"   ✓ Generated {n_samples:,} samples with {n_features} features")
    print(f"   ✓ Target range (original): [{y_true.min():.1f}, {y_true.max():.1f}] kWh")
    print(f"   ✓ Target range (scaled): [{y_scaled.min():.3f}, {y_scaled.max():.3f}]")
    
    return X, y_scaled, target_scaler


def main():
    """Run synthetic Linear Regression benchmark."""
    print("=" * 60)
    print("ASHRAE LINEAR REGRESSION SYNTHETIC BENCHMARK")
    print("=" * 60)
    
    # Generate synthetic data
    X, y, target_scaler = generate_synthetic_ashrae_data(
        n_samples=ASHRAE_TRAINING_CONFIG["max_samples"],
        n_features=30
    )
    
    # Split data
    print(f"\n📊 Splitting data ({ASHRAE_DATA_SPLITS['train']:.0%}/{ASHRAE_DATA_SPLITS['val']:.0%}/{ASHRAE_DATA_SPLITS['test']:.0%})...")
    
    n_total = len(X)
    n_train = int(n_total * ASHRAE_DATA_SPLITS["train"])
    n_val = int(n_total * ASHRAE_DATA_SPLITS["val"])
    n_test = n_total - n_train - n_val
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    # Combine train and validation for LR training (as per our actual pipeline)
    X_train_lr = np.vstack([X_train, X_val])
    y_train_lr = np.hstack([y_train, y_val])
    
    print(f"   • Training: {X_train_lr.shape}")
    print(f"   • Testing:  {X_test.shape}")
    
    # Train model
    print(f"\n🏋️ Training Linear Regression...")
    model = LinearRegression(
        fit_intercept=True,
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train_lr, y_train_lr)
    train_time = time.time() - start_time
    
    print(f"   ✓ Training time: {train_time:.3f} seconds")
    print(f"   • Model coefficients: {model.coef_.shape}")
    print(f"   • Model intercept: {model.intercept_:.4f}")
    
    # Evaluate model
    print(f"\n📈 Evaluating model...")
    start_time = time.time()
    y_pred_scaled = model.predict(X_test)
    pred_time = time.time() - start_time
    
    print(f"   ✓ Prediction time: {pred_time:.3f} seconds")
    
    # Inverse transform to original scale
    y_true_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_true_original, y_pred_original)
    
    print(f"\n📊 EVALUATION METRICS:")
    print(f"   MSE:    {metrics['MSE']:.4f}")
    print(f"   RMSE:   {metrics['RMSE']:.4f}")
    print(f"   MAPE:   {metrics['MAPE']:.2f}%")
    print(f"   R2:     {metrics['R2']:.4f}")
    print(f"   RMSLE:  {metrics['RMSLE']:.4f}")
    
    # Additional statistics
    print(f"\n📈 PREDICTION STATISTICS:")
    print(f"   • Actual mean: {y_true_original.mean():.2f} kWh")
    print(f"   • Predicted mean: {y_pred_original.mean():.2f} kWh")
    print(f"   • Actual std: {y_true_original.std():.2f} kWh")
    print(f"   • Predicted std: {y_pred_original.std():.2f} kWh")
    
    # Save results
    print(f"\n💾 Saving results...")
    results_dir = Path("results/ashrae/linear_regression")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "synthetic_metrics.csv", index=False)
    
    # Save model info
    model_info = {
        'model_type': 'LinearRegression',
        'n_features': X.shape[1],
        'n_train_samples': len(X_train_lr),
        'n_test_samples': len(X_test),
        'train_time_seconds': train_time,
        'prediction_time_seconds': pred_time,
        'rmse': metrics['RMSE'],
        'r2': metrics['R2']
    }
    
    info_df = pd.DataFrame([model_info])
    info_df.to_csv(results_dir / "synthetic_info.csv", index=False)
    
    print(f"   ✓ Results saved to: {results_dir}")
    
    # Feature importance (based on absolute coefficients)
    importance = np.abs(model.coef_)
    top_features = np.argsort(importance)[::-1][:10]
    
    print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES (Synthetic):")
    for i, feature_idx in enumerate(top_features):
        print(f"   {i+1:2d}. Feature {feature_idx:2d}: {importance[feature_idx]:.4f}")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SYNTHETIC BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"📊 Final Results:")
    print(f"   • Training time: {train_time:.3f}s")
    print(f"   • Prediction time: {pred_time:.3f}s")
    print(f"   • RMSE: {metrics['RMSE']:.4f}")
    print(f"   • MAPE: {metrics['MAPE']:.2f}%")
    print(f"   • R2: {metrics['R2']:.4f}")
    print(f"   • RMSLE: {metrics['RMSLE']:.4f}")
    print(f"📁 Results saved to: {results_dir}")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Synthetic Linear Regression benchmark completed successfully!")
    else:
        print("\n💥 Synthetic benchmark failed!")
        sys.exit(1)
