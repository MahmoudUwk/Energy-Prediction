"""ASHRAE Linear Regression Benchmark for Electricity Load Forecasting.

This script provides a baseline Linear Regression model for ASHRAE electricity 
consumption forecasting with building selection, implementing the same preprocessing 
and evaluation as LSTM models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Any, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from .preprocessing_ashrae import (
    load_ashrae_dataset,
    preprocess_ashrae_complete,
    get_ashrae_lstm_data,
)
from .ashrae_config import (
    ASHRAE_TRAINING_CONFIG,
    ASHRAE_METRICS_CONFIG,
    ASHRAE_BENCHMARK_CONFIG,
    ASHRAE_RESULTS_ROOT,
    ASHRAEDatasetAnalysis,
    calculate_ashrae_sample_shapes,
)

from tools.preprocess_data2 import RMSLE


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    return 100 * np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all required evaluation metrics."""
    metrics = {}
    
    # Mean Squared Error (MSE)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error (RMSE)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    
    # Mean Absolute Error (MAE)
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    metrics['MAPE'] = mean_absolute_percentage_error(y_true, y_pred)
    
    # R-squared (R2)
    metrics['R2'] = r2_score(y_true, y_pred)
    
    # Root Mean Squared Logarithmic Error (RMSLE)
    metrics['RMSLE'] = RMSLE(y_true, y_pred)
    
    return metrics


def prepare_data_for_linear_regression() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare ASHRAE data for Linear Regression training.
    
    Returns:
        Tuple of (X_train_flat, y_train, X_test_flat, y_test, target_scaler)
    """
    print("=" * 80)
    print("ASHRAE LINEAR REGRESSION BENCHMARK")
    print("=" * 80)
    
    # Show dataset analysis
    ASHRAEDatasetAnalysis.print_analysis()
    
    # Load and preprocess ASHRAE data
    print("\n1. Loading and preprocessing ASHRAE data...")
    data_path = Path(__file__).parent.parent / "dataset/ASHRAE/ashrae-energy-prediction"
    
    train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
    
    X_train, y_train, X_test, row_ids, target_scaler = preprocess_ashrae_complete(
        train_data, test_data, building_metadata, weather_train, weather_test
    )
    
    print(f"   ✓ Training features shape: {X_train.shape}")
    print(f"   ✓ Training target shape: {y_train.shape}")
    print(f"   ✓ Test features shape: {X_test.shape}")
    
    # For Linear Regression, we need to flatten the LSTM sequences
    print("\n2. Preparing data for Linear Regression...")
    
    # Create LSTM sequences first to get the same train/val/test split
    X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = get_ashrae_lstm_data(
        X_train, y_train, X_test, 
        seq_length=ASHRAE_TRAINING_CONFIG["sequence_length"],
        max_samples=ASHRAE_TRAINING_CONFIG["max_samples"]
    )
    
    print(f"   ✓ LSTM sequences - Train: {X_train_lstm.shape}, Val: {X_val_lstm.shape}, Test: {X_test_lstm.shape}")
    
    # Flatten sequences for Linear Regression (use only the last timestep of each sequence)
    X_train_flat = X_train_lstm[:, -1, :]  # Last timestep of each sequence
    X_val_flat = X_val_lstm[:, -1, :]
    X_test_flat = X_test_lstm[:, -1, :]
    
    print(f"   ✓ Flattened sequences - Train: {X_train_flat.shape}, Val: {X_val_flat.shape}, Test: {X_test_flat.shape}")
    
    # Combine train and validation for Linear Regression training
    X_train_lr = np.vstack([X_train_flat, X_val_flat])
    y_train_lr = np.hstack([y_train_lstm, y_val_lstm])
    
    print(f"   ✓ Combined training data: {X_train_lr.shape}")
    
    return X_train_lr, y_train_lr, X_test_flat, y_test_lstm, target_scaler


def train_linear_regression_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[LinearRegression, float]:
    """Train Linear Regression model."""
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
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"   ✓ Model trained in {train_time:.2f} seconds")
    print(f"   ✓ Coefficients shape: {model.coef_.shape}")
    print(f"   ✓ Intercept: {model.intercept_:.4f}")
    
    return model, train_time


def evaluate_model(
    model: LinearRegression, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    target_scaler: MinMaxScaler
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """Evaluate Linear Regression model and calculate metrics."""
    print("\n4. Evaluating Linear Regression model...")
    
    # Record prediction time
    start_time = time.time()
    y_pred_scaled = model.predict(X_test)
    pred_time = time.time() - start_time
    
    print(f"   ✓ Predictions generated in {pred_time:.2f} seconds")
    print(f"   • Scaled predictions range: [{y_pred_scaled.min():.4f}, {y_pred_scaled.max():.4f}]")
    
    # Inverse transform to get original scale
    y_true_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    print(f"   • Original scale predictions range: [{y_pred_original.min():.2f}, {y_pred_original.max():.2f}]")
    print(f"   • Original scale true values range: [{y_true_original.min():.2f}, {y_true_original.max():.2f}]")
    
    # Calculate all metrics on original scale
    metrics = calculate_all_metrics(y_true_original, y_pred_original)
    
    # Print metrics
    print("\n   📊 EVALUATION METRICS:")
    print(f"      MSE:   {metrics['MSE']:.4f}")
    print(f"      RMSE:  {metrics['RMSE']:.4f}")
    print(f"      MAE:   {metrics['MAE']:.4f}")
    print(f"      MAPE:  {metrics['MAPE']:.2f}%")
    print(f"      R²:    {metrics['R2']:.4f}")
    print(f"      RMSLE: {metrics['RMSLE']:.4f}")
    
    return metrics, y_true_original, y_pred_original, pred_time


def save_results(
    metrics: Dict[str, float], 
    model: LinearRegression, 
    train_time: float, 
    pred_time: float
) -> Path:
    """Save benchmark results to files."""
    print("\n5. Saving results...")
    
    # Create results directory
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
        'max_samples': ASHRAE_TRAINING_CONFIG["max_samples"],
        'train_fraction': ASHRAE_TRAINING_CONFIG["train_fraction"],
        'val_fraction': ASHRAE_TRAINING_CONFIG["val_fraction"],
        'test_fraction': ASHRAE_TRAINING_CONFIG["test_fraction"],
        'train_time_seconds': train_time,
        'prediction_time_seconds': pred_time,
        'n_features': len(model.coef_),
        'fit_intercept': model.fit_intercept,
        **{f'n_{k}_samples': v for k, v in sample_shapes.items() if k != 'total'},
        'rmse': metrics['RMSE'],
        'mae': metrics['MAE'],
        'r2': metrics['R2'],
        'mape': metrics['MAPE'],
        'rmsle': metrics['RMSLE']
    }
    
    info_df = pd.DataFrame([training_info])
    info_df.to_csv(results_dir / "training_info.csv", index=False)
    print(f"   ✓ Training info saved to: {results_dir / 'training_info.csv'}")
    
    return results_dir


def analyze_feature_importance(model: LinearRegression, results_dir: Path):
    """Analyze and save feature importance based on coefficients."""
    print("\n6. Analyzing feature importance...")
    
    # Get absolute coefficients as importance scores
    importance = np.abs(model.coef_)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature_index': range(len(importance)),
        'importance': importance,
        'coefficient': model.coef_
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Save importance rankings
    importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
    print(f"   ✓ Feature importance saved to: {results_dir / 'feature_importance.csv'}")
    
    # Print top 10 most important features
    print("\n   🔝 TOP 10 MOST IMPORTANT FEATURES:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"      {i+1:2d}. Feature {int(row['feature_index']):3d}: {row['importance']:6.4f} (coef: {row['coefficient']: .4f})")


def main():
    """Main function to run Linear Regression benchmark."""
    try:
        # Prepare data
        X_train, y_train, X_test, y_test, target_scaler = prepare_data_for_linear_regression()
        
        # Train model
        model, train_time = train_linear_regression_model(X_train, y_train)
        
        # Evaluate model
        metrics, y_true, y_pred, pred_time = evaluate_model(model, X_test, y_test, target_scaler)
        
        # Save results
        results_dir = save_results(metrics, model, train_time, pred_time)
        
        # Analyze feature importance
        analyze_feature_importance(model, results_dir)
        
        # Final summary
        print("\n" + "=" * 80)
        print("LINEAR REGRESSION BENCHMARK COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📊 Final Results:")
        print(f"   • Training time: {train_time:.2f} seconds")
        print(f"   • Prediction time: {pred_time:.2f} seconds")
        print(f"   • RMSE: {metrics['RMSE']:.4f}")
        print(f"   • MAE: {metrics['MAE']:.4f}")
        print(f"   • MAPE: {metrics['MAPE']:.2f}%")
        print(f"   • R²: {metrics['R2']:.4f}")
        print(f"   • RMSLE: {metrics['RMSLE']:.4f}")
        print(f"\n📁 Results saved to: {results_dir}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Linear Regression benchmark: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    if success:
        print("\n🎉 ASHRAE Linear Regression benchmark completed successfully!")
    else:
        print("\n💥 ASHRAE Linear Regression benchmark failed!")
        sys.exit(1)

