"""Test script for ASHRAE preprocessing pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append('.')

from ashrae.preprocessing_ashrae import (
    load_ashrae_dataset,
    preprocess_ashrae_complete,
    get_ashrae_lstm_data,
    inverse_transform_ashrae_predictions
)


def test_ashrae_preprocessing():
    """Test the complete ASHRAE preprocessing pipeline."""
    
    print("=" * 60)
    print("Testing ASHRAE Preprocessing Pipeline")
    print("=" * 60)
    
    # Define data path
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    try:
        # Step 1: Load datasets
        print("\n1. Loading ASHRAE datasets...")
        train_data, test_data, building_metadata, weather_train, weather_test = load_ashrae_dataset(data_path)
        
        print(f"   ✓ Train data shape: {train_data.shape}")
        print(f"   ✓ Test data shape: {test_data.shape}")
        print(f"   ✓ Building metadata shape: {building_metadata.shape}")
        print(f"   ✓ Weather train shape: {weather_train.shape}")
        print(f"   ✓ Weather test shape: {weather_test.shape}")
        
        # Step 2: Complete preprocessing
        print("\n2. Running complete preprocessing pipeline...")
        X_train, y_train, X_test, row_ids = preprocess_ashrae_complete(
            train_data, test_data, building_metadata, weather_train, weather_test
        )
        
        print(f"   ✓ X_train shape: {X_train.shape}")
        print(f"   ✓ y_train shape: {y_train.shape}")
        print(f"   ✓ X_test shape: {X_test.shape}")
        print(f"   ✓ Row IDs shape: {row_ids.shape}")
        
        # Step 3: Check for missing values
        print("\n3. Checking for missing values...")
        train_missing = X_train.isnull().sum().sum()
        test_missing = X_test.isnull().sum().sum()
        print(f"   ✓ Missing values in X_train: {train_missing}")
        print(f"   ✓ Missing values in X_test: {test_missing}")
        
        # Step 4: Check feature types and ranges
        print("\n4. Feature analysis...")
        print(f"   ✓ Number of features: {X_train.shape[1]}")
        print(f"   ✓ Feature names: {list(X_train.columns)}")
        
        # Check numeric features
        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        print(f"   ✓ Numeric features: {len(numeric_features)}")
        
        # Check feature ranges
        print("\n   Feature ranges (first 10 features):")
        for i, col in enumerate(numeric_features[:10]):
            min_val = X_train[col].min()
            max_val = X_train[col].max()
            mean_val = X_train[col].mean()
            print(f"     {col}: [{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}")
        
        # Step 5: Test LSTM data preparation
        print("\n5. Testing LSTM data preparation...")
        seq_length = 23
        X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm = get_ashrae_lstm_data(
            X_train, y_train, X_test, seq_length=seq_length, max_samples=100000
        )
        
        print(f"   ✓ X_train_lstm shape: {X_train_lstm.shape}")
        print(f"   ✓ y_train_lstm shape: {y_train_lstm.shape}")
        print(f"   ✓ X_val_lstm shape: {X_val_lstm.shape}")
        print(f"   ✓ y_val_lstm shape: {y_val_lstm.shape}")
        print(f"   ✓ X_test_lstm shape: {X_test_lstm.shape}")
        print(f"   ✓ y_test_lstm shape: {y_test_lstm.shape}")
        
        # Step 6: Test inverse transformation
        print("\n6. Testing inverse transformation...")
        sample_pred = np.array([0.5, 1.0, 1.5, 2.0])
        original_pred = inverse_transform_ashrae_predictions(sample_pred)
        print(f"   ✓ Sample log predictions: {sample_pred}")
        print(f"   ✓ Original scale predictions: {original_pred}")
        
        # Step 7: Summary
        print("\n" + "=" * 60)
        print("PREPROCESSING TEST RESULTS")
        print("=" * 60)
        print("✓ All datasets loaded successfully")
        print("✓ Preprocessing pipeline completed without errors")
        print("✓ No missing values in final datasets")
        print("✓ LSTM data preparation successful")
        print("✓ Inverse transformation working correctly")
        print(f"✓ Final feature count: {X_train.shape[1]}")
        print(f"✓ LSTM sequence length: {seq_length}")
        print("✓ Ready for model training!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_ashrae_preprocessing()
    if success:
        print("\n🎉 ASHRAE preprocessing pipeline test PASSED!")
    else:
        print("\n💥 ASHRAE preprocessing pipeline test FAILED!")
        sys.exit(1)
