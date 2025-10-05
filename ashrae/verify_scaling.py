"""Verify scaling pipeline for ASHRAE preprocessing."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ashrae.preprocessing_ashrae_disjoint import preprocess_ashrae_disjoint_splits
from ashrae.ashrae_config import ASHRAE_TRAINING_CONFIG

def verify_scaling_pipeline():
    """Verify that scaling is applied correctly throughout the pipeline."""
    print("=" * 80)
    print("VERIFYING ASHRAE SCALING PIPELINE")
    print("=" * 80)
    
    # Get preprocessed data
    print("\n1. Loading preprocessed data...")
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = preprocess_ashrae_disjoint_splits(
        target_samples=ASHRAE_TRAINING_CONFIG["max_samples"],
        train_fraction=ASHRAE_TRAINING_CONFIG["train_fraction"],
        val_fraction=ASHRAE_TRAINING_CONFIG["val_fraction"],
        test_fraction=ASHRAE_TRAINING_CONFIG["test_fraction"]
    )
    
    print(f"\n2. Checking scaled data ranges...")
    
    # Check feature scaling
    print(f"\n   📊 FEATURE SCALING (should be [0, 1]):")
    for col in X_train.columns[:5]:  # Check first 5 features
        print(f"      {col:30s}: Train [{X_train[col].min():.4f}, {X_train[col].max():.4f}], "
              f"Test [{X_test[col].min():.4f}, {X_test[col].max():.4f}]")
    
    # Check target scaling
    print(f"\n   🎯 TARGET SCALING (should be [0, 1]):")
    print(f"      Train: [{y_train.min():.4f}, {y_train.max():.4f}]")
    print(f"      Val:   [{y_val.min():.4f}, {y_val.max():.4f}]")
    print(f"      Test:  [{y_test.min():.4f}, {y_test.max():.4f}]")
    
    # Test inverse transformation
    print(f"\n3. Testing inverse transformation...")
    
    # Inverse transform a few samples
    sample_scaled = y_test[:5]
    sample_original = target_scaler.inverse_transform(sample_scaled.reshape(-1, 1)).flatten()
    
    print(f"\n   📈 SAMPLE INVERSE TRANSFORMATION:")
    print(f"      Scaled values:   {sample_scaled}")
    print(f"      Original values: {sample_original}")
    
    # Test full inverse transform
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    print(f"\n   🔄 FULL INVERSE TRANSFORMATION:")
    print(f"      Original target range: [{y_test_original.min():.2f}, {y_test_original.max():.2f}] kWh")
    print(f"      Original target mean:  {y_test_original.mean():.2f} kWh")
    print(f"      Original target std:   {y_test_original.std():.2f} kWh")
    
    # Verify scaler attributes
    print(f"\n4. Verifying scaler properties...")
    print(f"   Target scaler data_min_: {target_scaler.data_min_[0]:.2f}")
    print(f"   Target scaler data_max_: {target_scaler.data_max_[0]:.2f}")
    print(f"   Target scaler feature_range: {target_scaler.feature_range}")
    
    # Simulate prediction workflow
    print(f"\n5. Simulating prediction workflow...")
    print(f"\n   📝 CORRECT WORKFLOW:")
    print(f"      1. Train on SCALED features (X_train) and SCALED targets (y_train)")
    print(f"      2. Predict on SCALED test features (X_test) → SCALED predictions")
    print(f"      3. Inverse transform SCALED predictions → ORIGINAL scale predictions")
    print(f"      4. Inverse transform SCALED targets → ORIGINAL scale targets")
    print(f"      5. Calculate metrics on ORIGINAL scale (unscaled) data")
    
    # Example with dummy predictions
    y_pred_scaled = y_test  # Assume perfect predictions for demo
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"\n   ✅ METRICS CALCULATION (on ORIGINAL scale):")
    print(f"      RMSE: {rmse:.2f} kWh (should be 0 for perfect predictions)")
    print(f"      R²: {r2:.4f} (should be 1.0 for perfect predictions)")
    
    # Check for common mistakes
    print(f"\n6. Checking for common mistakes...")
    
    # Mistake 1: Calculate metrics on scaled data
    rmse_scaled = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    print(f"\n   ❌ WRONG: Metrics on SCALED data:")
    print(f"      RMSE on scaled: {rmse_scaled:.4f} (meaningless!)")
    
    # Mistake 2: Not inverse transforming predictions
    print(f"\n   ❌ WRONG: Comparing scaled predictions with original targets")
    print(f"      Would mix scales and give incorrect metrics")
    
    print(f"\n" + "=" * 80)
    print("SCALING VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"\n✅ Pipeline is correct if:")
    print(f"   • Features in [0, 1] range")
    print(f"   • Targets in [0, 1] range")
    print(f"   • Inverse transform returns original kWh values")
    print(f"   • Metrics calculated on original (unscaled) values")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    verify_scaling_pipeline()

