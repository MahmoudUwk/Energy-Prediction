"""ASHRAE Linear Regression Benchmark with Building Selection.

This script provides a baseline Linear Regression model for ASHRAE electricity 
consumption forecasting, using building selection to manage dataset size while ensuring
diverse building types for generalization.
"""

from __future__ import annotations

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

from ashrae.preprocessing_ashrae import load_ashrae_dataset, preprocess_ashrae_complete, get_ashrae_lstm_data
from ashrae_config import (
    ASHRAE_TRAINING_CONFIG,
    ASHRAE_DATA_SPLITS,
    ASHRAEDatasetAnalysis,
    calculate_ashrae_sample_shapes,
)
from ashrae_config import ASHRAE_BENCHMARK_CONFIG, ASHRAE_RESULTS_ROOT


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


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate all required evaluation metrics."""
    metrics = {}
    
    # Mean Squared Error (MSE)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Mean Absolute Percentage Error (MAPE)
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # R-squared (R2)
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Root Mean Squared Logarithmic Error (RMSLE)
    metrics['RMSLE'] = root_mean_squared_logarithmic_error(y_true, y_pred)
    
    return metrics


def main():
    """Main function to run Linear Regression benchmark with building selection."""
    print("=" * 80)
    print("ASHRAE LINEAR REGRESSION BENCHMARK WITH BUILDING SELECTION")
    print("=" * 80)
    
    # Show dataset analysis
    ASHRAEDatasetAnalysis.print_analysis()
    
    # Load and preprocess ASHRAE data with building selection
    print("\n1. Loading and preprocessing ASHRAE data with building selection...")
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    try:
        train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
        
        print(f"   ✓ Loaded full dataset: {train_data.shape:,} train rows")
        print(f"   ✓ Buildings available: {building_metadata['building_id'].nunique()}")
        
        # Preprocess with building selection
        start_time = time.time()
        X_train, y_train, X_test, row_ids = preprocess_ashrae_complete(
            train_data, test_data, building_metadata, weather_train, weather_test
        )
        preprocessing_time = time.time() - start_time
        
        print(f"   ✓ Preprocessing time: {preprocessing_time:.2f} seconds")
        print(f"   ✓ Training features shape: {X_train.shape}")
        print(f"   • Training target shape: {y_train.shape}")
        print(f"   • Test features shape: {X_test.shape}")
        print(f"   • Test row IDs shape: {row_ids.shape}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get target scaler for inverse transformation
    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler.fit(y_train.reshape(-1, 1))
    
    # For Linear Regression, we need to flatten the LSTM sequences
    print("\n2. Preparing data for Linear Regression...")
    
    # Create LSTM sequences first to get the same train/val/test split
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = get_ashrae_lstm_data(
        X_train, y_train, X_test, 
        seq_length=ASHRAE_TRAINING_CONFIG["sequence_length"],
        max_samples=ASHRAE_TRAINING_CONFIG["max_samples"]
    )
    
    print(f"   ✓ LSTM sequences - Train: {X_train_lstm.shape}, Val: {X_val_lstm.shape}, Test: {X_test_lstm.shape}")
    print(f"   • Target ranges - Train: [{y_train_lstm.min():.3f}, {y_train_lstm.max():.3f}]")
    print(f"   • Target ranges - Test: [{y_test_lstm.min():.3f}, {y_test_lstm.max():.3f}]")
    
    # Flatten sequences for Linear Regression (use only the last timestep of each sequence)
    X_train_flat = X_train_lstm[:, -1, :]  # Last timestep of each sequence
    X_val_flat = X_val_lstm[:, -1, :]
    X_test_flat = X_test_lstm[:, -1, :]
    
    print(f"   ✓ Flattened sequences - Train: {X_train_flat.shape}, Val: {X_val_flat.shape}, Test: {X_test_flat.shape}")
    
    # Combine train and validation for Linear Regression training
    X_train_lr = np.vstack([X_train_flat, X_val_flat])
    y_train_lr = np.hstack([y_train_lstm, y_val_lstm])
    
    print(f"   ✓ Combined training data: {X_train_lr.shape}")
    print(f"   ✓ Final target shape: {y_train_lr.shape}")
    
    # Train Linear Regression model
    print("\n3. Training Linear Regression model...")
    
    # Initialize Linear Regression with config parameters
    lr_config = ASHRAE_BENCHMARK_CONFIG["linear_regression"]
    model = LinearRegression(
        fit_intercept=lr_config["fit_intercept"],
        copy_X=lr_config["copy_X"],
        n_jobs=lr_config["n_jobs"]
    )
    
    # Record training time
    start_time = time.time()
    model.fit(X_train_lr, y_train_lr)
    train_time = time.time() - start_time
    
    print(f"   ✓ Model trained in {train_time:.2f} seconds")
    print(f"   • Coefficients shape: {model.coef_.shape}")
    print(f"   • Intercept: {model.intercept_:.4f}")
    
    # Evaluate model
    print("\n4. Evaluating Linear Regression model...")
    
    # Record prediction time
    start_time = time.time()
    y_pred_scaled = model.predict(X_test_flat)
    pred_time = time.time() - start_time
    
    print(f"   ✓ Predictions generated in {pred_time:.2f} seconds")
    print(f"   • Scaled predictions range: [{y_pred_scaled.min():.4f}, {y_pred_scaled.max():.4f}]")
    
    # Inverse transform to get original scale
    y_true_original = target_scaler.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"   • Original scale predictions range: [{y_pred_original.min():.2f}, {y_pred_original.max():.2f}]")
    print(f"   • Original scale true values range: [{y_true_original.min():.2f}, {y_true_original.max():.2f}]")
    
    # Calculate all metrics on original scale
    metrics = calculate_all_metrics(y_true_original, y_pred_original)
    
    # Print metrics
    print("\n   📊 EVALUATION METRICS (Original Scale):")
    print(f"      MSE:  {metrics['MSE']:.4f}")
    print(f"      RMSE: {metrics['RMSE']:.4f}")
    print(f"      MAPE: {metrics['MAPE']:.2f}%")
    print(f"      R2:   {metrics['R2']:.4f}")
    print(f"      RMSLE: {metrics['RMSLE']:.4f}")
    
    # Additional analysis
    print(f"\n   📈 PREDICTION ANALYSIS:")
    print(f"      Actual mean: {y_true_original.mean():.2f} kWh")
    print(f"      Predicted mean: {y_pred_original.mean():.2f} kWh")
    print(f"      Actual std: {y_true_original.std():.2f} kWh")
    print(f"      Predicted std: {y_pred_original.std():.2f} kWh")
    print(f"      Prediction error distribution: RMSE={metrics['RMSE']:.2f} kWh")
    
    # Performance summary
    total_samples = len(y_true_original)
    samples_within_10kwh = np.sum(np.abs(y_true_original - y_pred_original) <= 10) / total_samples * 100
    samples_within_50kwh = np.sum(np.abs(y_true_original - y_pred_original <= 50) / total_samples * 100
    
    print(f"\n   📈 ACCURACY ANALYSIS:")
    print(f"      Predictions within ±10 kWh: {samples_within_10kwh:.1f}%")
    print(f"      Predictions within ±50 kWh: {samples_within_50kwh:.1f}%")
    
    # Save results
    print("\n5. Saving results...")
    results_dir = ASHRAE_RESULTS_ROOT / "linear_regression"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / "metrics.csv", index=False)
    print(f"   ✓ Metrics saved to: {results_dir / 'metrics.csv'}")
    
    # Save model coefficients
    coef_df = pd.DataFrame({
        'feature_index': range(len(model.coef_)),
        'coefficient': model.coef_
    })
    coef_df.to_csv(results_dir / "coefficients.csv", index=False)
    print(f"   ✓ Coefficients saved to: {results_dir / 'coefficients.csv'}")
    
    # Save training info
    sample_shapes = calculate_ashrae_sample_shapes(ASHRAE_TRAINING_CONFIG["max_samples"])
    training_info = {
        'model_type': 'LinearRegression',
        'buildings_used': ASHRAE_TRAINING_CONFIG["max_buildings"],
        'building_selection_strategy': ASHRAE_TRAINING_CONFIG["building_selection_strategy"],
        'preprocessing_time_seconds': preprocessing_time,
        'train_time_seconds': train_time,
        'prediction_time_seconds': pred_time,
        'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown',
        'n_train_samples': len(X_train_lr),
        'n_test_samples': len(y_test_original),
        'fit_intercept': model.fit_intercept,
        'sample_shapes': sample_shapes,
        'rmse': metrics['RMSE'],
        'r2': metrics['R2'],
        'mape': metrics['MAPE']
    }
    
    info_df = pd.DataFrame([training_info])
    info_df.to_csv(results_dir / "training_info.csv", index=False)
    print(f"   ✓ Training info saved to: {results_dir / 'training_info.csv'}")
    
    # Feature importance (based on absolute coefficients)
    importance = np.abs(model.coef_)
    top_features = np.argsort(importance)[::-1][:20]
    
    print(f"\n   🔝 TOP 20 MOST IMPORTANT FEATURES:")
    for i, feature_idx in enumerate(top_features):
        print(f"      {i+1:2d}. Feature {feature_idx:3d}: {importance[feature_idx]:.6f}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("LINEAR REGRESSION BENCHMARK WITH BUILDING SELECTION COMPLETED")
    print("=" * 80)
    print(f"🏗️ BUILDING SELECTION:")
    print(f"   • Max buildings: {ASHRAE_TRAINING_CONFIG['max_buildings']}")
    print(f"   • Selection strategy: {ASHRAE_TRAINING_CONFIG['building_selection_strategy']}")
    print(f"   • Buildings used: {ASHRAE_TRAINING_CONFIG['max_buildings']} out of {train_data['building_id'].nunique()}")
    
    print(f"📊 PERFORMANCE SUMMARY:")
    print(f"   • Preprocessing time: {preprocessing_time:.2f}s")
    print(f"   • Training time: {train_time:.2f}s")
    print(f"   • Prediction time: {pred_time:.2f}s")
    print(f"   • RMSE: {metrics['RMSE']:.4f}")
    print(f"   • MAPE: {metrics['MAPE']:.2f}%")
    print(f"   • R2: {metrics['R2']:.4f}")
    print(f"   • RMSLE: {metrics['RMSLE']:.4f}")
    print(f"\n📁 Results saved to: {results_dir}")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ASHRAE Linear Regression benchmark with building selection completed!")
    else:
        print("\n💥 ASHRAE Linear Regression benchmark failed!")
        sys.exit(1)
