"""ASHRAE Linear Regression Benchmark with Disjoint Building Splits.

This script uses separate buildings for train/val/test to ensure better generalization testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Dict, Tuple

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from .preprocessing_ashrae_disjoint import preprocess_ashrae_disjoint_splits
from .ashrae_config import (
    ASHRAE_TRAINING_CONFIG,
    ASHRAE_BENCHMARK_CONFIG,
    ASHRAE_RESULTS_ROOT,
)

from tools.preprocess_data2 import RMSLE


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate all required evaluation metrics."""
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'RMSLE': RMSLE(y_true, y_pred)
    }


def main():
    """Main function to run Linear Regression benchmark with disjoint building splits."""
    print("=" * 80)
    print("ASHRAE LINEAR REGRESSION BENCHMARK - DISJOINT BUILDING SPLITS")
    print("=" * 80)
    
    try:
        # Load preprocessed data with disjoint splits
        print("\n1. Loading preprocessed data with disjoint building splits...")
        start_time = time.time()
        
        X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = preprocess_ashrae_disjoint_splits(
            target_samples=ASHRAE_TRAINING_CONFIG["max_samples"],
            train_fraction=ASHRAE_TRAINING_CONFIG["train_fraction"],
            val_fraction=ASHRAE_TRAINING_CONFIG["val_fraction"],
            test_fraction=ASHRAE_TRAINING_CONFIG["test_fraction"]
        )
        
        preprocessing_time = time.time() - start_time
        
        print(f"\n   ✓ Preprocessing time: {preprocessing_time:.2f} seconds")
        print(f"   ✓ Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"   ✓ Val: {X_val.shape[0]:,} samples")
        print(f"   ✓ Test: {X_test.shape[0]:,} samples")
        
        # Combine train and validation for Linear Regression
        print("\n2. Preparing Linear Regression data...")
        X_train_lr = pd.concat([X_train, X_val], axis=0).values
        y_train_lr = np.hstack([y_train, y_val])
        X_test_lr = X_test.values
        
        print(f"   ✓ Combined training data: {X_train_lr.shape}")
        print(f"   ✓ Test data: {X_test_lr.shape}")
        
        # Train model
        print("\n3. Training Linear Regression model...")
        lr_config = ASHRAE_BENCHMARK_CONFIG["linear_regression"]
        model = LinearRegression(
            fit_intercept=lr_config["fit_intercept"],
            copy_X=lr_config["copy_X"],
            n_jobs=lr_config["n_jobs"]
        )
        
        start_time = time.time()
        model.fit(X_train_lr, y_train_lr)
        train_time = time.time() - start_time
        
        print(f"   ✓ Model trained in {train_time:.2f} seconds")
        print(f"   ✓ Coefficients: {model.coef_.shape}")
        print(f"   ✓ Intercept: {model.intercept_:.4f}")
        
        # Evaluate model
        print("\n4. Evaluating model on test set...")
        start_time = time.time()
        y_pred_scaled = model.predict(X_test_lr)
        pred_time = time.time() - start_time
        
        print(f"   ✓ Predictions generated in {pred_time:.2f} seconds")
        
        # Inverse transform to original scale
        y_true_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        print(f"   • True values range: [{y_true_orig.min():.2f}, {y_true_orig.max():.2f}] kWh")
        print(f"   • Predictions range: [{y_pred_orig.min():.2f}, {y_pred_orig.max():.2f}] kWh")
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_true_orig, y_pred_orig)
        
        print("\n   📊 EVALUATION METRICS (Original Scale):")
        print(f"      MSE:   {metrics['MSE']:.4f}")
        print(f"      RMSE:  {metrics['RMSE']:.4f} kWh")
        print(f"      MAE:   {metrics['MAE']:.4f} kWh")
        print(f"      MAPE:  {metrics['MAPE']:.2f}%")
        print(f"      R²:    {metrics['R2']:.4f}")
        print(f"      RMSLE: {metrics['RMSLE']:.4f}")
        
        # Save results
        print("\n5. Saving results...")
        results_dir = ASHRAE_RESULTS_ROOT / "linear_regression_disjoint"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(results_dir / "metrics.csv", index=False)
        print(f"   ✓ Metrics saved to: {results_dir / 'metrics.csv'}")
        
        # Save coefficients
        coef_df = pd.DataFrame({
            'feature_index': range(len(model.coef_)),
            'coefficient': model.coef_
        })
        coef_df.to_csv(results_dir / "coefficients.csv", index=False)
        
        # Save training info
        training_info = {
            'model_type': 'LinearRegression',
            'split_strategy': 'disjoint_buildings',
            'n_train_samples': len(y_train_lr),
            'n_test_samples': len(y_test),
            'n_features': X_train_lr.shape[1],
            'preprocessing_time_seconds': preprocessing_time,
            'train_time_seconds': train_time,
            'prediction_time_seconds': pred_time,
            'fit_intercept': model.fit_intercept,
            **metrics
        }
        
        info_df = pd.DataFrame([training_info])
        info_df.to_csv(results_dir / "training_info.csv", index=False)
        print(f"   ✓ Training info saved to: {results_dir / 'training_info.csv'}")
        
        # Feature importance
        importance = np.abs(model.coef_)
        importance_df = pd.DataFrame({
            'feature_index': range(len(importance)),
            'importance': importance,
            'coefficient': model.coef_
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(results_dir / "feature_importance.csv", index=False)
        
        print(f"\n   🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"      {i+1:2d}. Feature {int(row['feature_index']):2d}: "
                  f"{row['importance']:6.4f} (coef: {row['coefficient']: .4f})")
        
        # Final summary
        print("\n" + "=" * 80)
        print("LINEAR REGRESSION BENCHMARK WITH DISJOINT SPLITS COMPLETED")
        print("=" * 80)
        print(f"📊 Final Results:")
        print(f"   • Preprocessing: {preprocessing_time:.2f}s")
        print(f"   • Training: {train_time:.2f}s")
        print(f"   • Prediction: {pred_time:.2f}s")
        print(f"   • RMSE: {metrics['RMSE']:.4f} kWh")
        print(f"   • MAE: {metrics['MAE']:.4f} kWh")
        print(f"   • MAPE: {metrics['MAPE']:.2f}%")
        print(f"   • R²: {metrics['R2']:.4f}")
        print(f"   • RMSLE: {metrics['RMSLE']:.4f}")
        print(f"\n📁 Results saved to: {results_dir}")
        print("\n✨ Using DISJOINT building splits for better generalization testing")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
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

