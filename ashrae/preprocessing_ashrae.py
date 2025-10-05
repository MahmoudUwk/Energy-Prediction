"""ASHRAE-specific data preprocessing utilities for energy prediction experiments.

This module provides preprocessing functions specifically designed for the ASHRAE
Great Energy Predictor III dataset, preserving the original preprocessing code
for the Portuguese dataset while adding ASHRAE-specific functionality.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

# Import ASHRAE configuration
from .ashrae_config import (
    ASHRAE_TRAINING_CONFIG,
    ASHRAE_FEATURE_CONFIG,
    ASHRAE_DATA_SPLITS,
    ASHRAEDatasetAnalysis,
)

# Import helper functions from existing preprocessing modules
from tools.preprocess_data2 import (
    sliding_windows2d_lstm,
    RMSE,
    MAE,
    MAPE,
    RMSLE,
    compute_metrics,
    persist_model_results,
    plot_test,
    log_results,
    log_results_LSTM,
)


def pivot_ashrae_meters(df: pd.DataFrame, test_mode: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Pivot ASHRAE data to have electricity as target and all meter types as features.
    
    Args:
        df: DataFrame with meter readings
        test_mode: If True, return features only; if False, return features and electricity target
        
    Returns:
        Tuple of (features_df, electricity_target_series) or (features_df, None) for test
    """
    if test_mode:
        # For test data, we need to create the same structure as training data
        # but we don't have meter_reading values, so we create zeros
        # We need to ensure test data has all meter types for each building/timestamp
        
        # First, check if test data has the expected structure
        expected_cols = ['row_id', 'building_id', 'meter', 'timestamp']
        if not all(col in df.columns for col in expected_cols):
            raise ValueError(f"Test data missing expected columns. Found: {df.columns.tolist()}")
        
        # Create a pivot-like structure for test data
        # Since we don't have meter readings, we'll create placeholder rows
        # with zeros for all meter types
        unique_combinations = df[['building_id', 'timestamp']].drop_duplicates()
        
        test_data_expanded = []
        for _, row in unique_combinations.iterrows():
            # Create a row for this building/timestamp with all meter types
            test_row = {
                'building_id': row['building_id'],
                'timestamp': row['timestamp'],
                'chilledwater': 0,  # No actual readings available
                'steam': 0,
                'hotwater': 0
            }
            test_data_expanded.append(test_row)
        
        meter_pivot = pd.DataFrame(test_data_expanded)
        
        # Note: NO log transform to match Portuguese dataset methodology
        # Zeros remain zeros, which is fine for MinMaxScaler
        
        return meter_pivot, None
        
    else:
        # For training data, we have meter readings and can pivot properly
        meter_pivot = df.pivot_table(
            index=['building_id', 'timestamp'],
            columns='meter',
            values='meter_reading',
            fill_value=0
        ).reset_index()
        
        # Rename columns to be descriptive
        meter_pivot.columns = ['building_id', 'timestamp', 'electricity', 'chilledwater', 'steam', 'hotwater']
        
        # Extract electricity as target, keep others as features
        # Note: NO log transform to match Portuguese dataset methodology
        y = meter_pivot['electricity'].copy()
        X = meter_pivot.drop('electricity', axis=1)
        return X, y


def prepare_ashrae_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    building_metadata: pd.DataFrame,
    weather_train: pd.DataFrame,
    weather_test: pd.DataFrame,
    test_mode: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare ASHRAE dataset with electricity as target and all meter types as features.
    
    Args:
        train_data: Training data with meter readings
        test_data: Test data without meter readings
        building_metadata: Building characteristics
        weather_train: Weather data for training period
        weather_test: Weather data for test period
        test_mode: If True, return test data; if False, return training data
        
    Returns:
        Tuple of (features_df, target_series) for training or (features_df, row_ids) for test
        
    Note:
        Uses electricity (meter = 0) as target variable while keeping all meter types
        (electricity, chilledwater, steam, hotwater) as features for multi-variate modeling.
    """
    # Select appropriate datasets based on mode
    if test_mode:
        data = test_data.copy()
        weather_data = weather_test.copy()
        # Preserve row_ids before pivot
        original_row_ids = data['row_id'].copy() if 'row_id' in data.columns else None
    else:
        data = train_data.copy()
        weather_data = weather_train.copy()
        original_row_ids = None
    
    # Pivot meter readings to have electricity as target and all meter types as features
    meter_data, y = pivot_ashrae_meters(data, test_mode)
    
    # Merge building metadata
    X = meter_data.merge(building_metadata, on="building_id", how="left")
    
    # Merge weather data
    X = X.merge(weather_data, on=["site_id", "timestamp"], how="left")
    
    # Convert timestamp to datetime
    X.timestamp = pd.to_datetime(X.timestamp, format="%Y-%m-%d %H:%M:%S")
    
    # Optional log transform for square feet (disabled when False for MinMax-only preprocessing)
    if ASHRAE_FEATURE_CONFIG.get("square_feet_log_transform", False):
        X.square_feet = np.log1p(X.square_feet)
    
    if not test_mode:
        # Sort by timestamp for training data
        X.sort_values("timestamp", inplace=True)
        X.reset_index(drop=True, inplace=True)
        y.index = X.index  # Align target with sorted features
    
    # Define US holidays for the dataset period
    holidays = [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
        "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
        "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
        "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
        "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
        "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
        "2019-01-01"
    ]
    
    # Create temporal features
    X["hour"] = X.timestamp.dt.hour
    X["weekday"] = X.timestamp.dt.weekday
    X["is_holiday"] = (X.timestamp.dt.date.astype("str").isin(holidays)).astype(int)
    
    # Drop unnecessary features
    drop_features = ["timestamp", "sea_level_pressure", "wind_direction", "wind_speed"]
    X.drop(drop_features, axis=1, inplace=True)
    
    # Apply one-hot encoding to categorical features
    X = pd.get_dummies(X, columns=["primary_use"])
    
    if test_mode:
        # For test data, we need to handle row_ids differently since pivot changes structure
        # We'll return the original row_ids if available, otherwise create sequential ones
        if original_row_ids is not None:
            # Since we created new rows in pivot, we need to map back to original row_ids
            # For now, we'll return a simple mapping approach
            row_ids = pd.Series(range(len(X)))  # Placeholder: will need proper mapping
        else:
            row_ids = pd.Series(range(len(X)))  # Create sequential IDs
        return X, row_ids
    else:
        # For training data, y already contains electricity target from pivot function
        return X, y


def normalize_ashrae_features(df: pd.DataFrame, train_stats: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize ASHRAE features using MinMaxScaler for consistency with original dataset.
    
    Args:
        df: DataFrame with features to normalize
        train_stats: Optional dict with MinMaxScaler and training statistics
        
    Returns:
        Tuple of (Normalized DataFrame, stats dict)
    """
    from sklearn.preprocessing import MinMaxScaler
    from config import MINMAX_FEATURE_RANGE
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if train_stats is None:
        # Fit on training data
        scaler = MinMaxScaler(feature_range=MINMAX_FEATURE_RANGE)
        scaler.fit(df[numeric_cols])
        train_stats = {'scaler': scaler, 'numeric_cols': numeric_cols}
    else:
        # Use training statistics
        scaler = train_stats['scaler']
        numeric_cols = train_stats['numeric_cols']
    
    # Apply MinMax normalization
    df_copy = df.copy()
    df_copy[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df_copy, train_stats


def impute_ashrae_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform mean imputation for missing values in ASHRAE numeric features.
    
    Args:
        df: DataFrame with potential missing values
        
    Returns:
        DataFrame with imputed values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()))
    return df


def remove_dew_temperature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove dew_temperature feature due to correlation with site_id.
    
    Args:
        df: DataFrame potentially containing dew_temperature
        
    Returns:
        DataFrame without dew_temperature
    """
    if "dew_temperature" in df.columns:
        df.drop(columns=["dew_temperature"], inplace=True)
    return df


def preprocess_ashrae_complete(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    building_metadata: pd.DataFrame,
    weather_train: pd.DataFrame,
    weather_test: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.Series, "MinMaxScaler"]:
    """
    Complete ASHRAE preprocessing pipeline with building selection.
    
    Args:
        train_data: Training data
        test_data: Test data
        building_metadata: Building metadata
        weather_train: Training weather data
        weather_test: Test weather data
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
        
    Note:
        Processes only electricity meters (meter = 0) for direct comparison
        with the original Portuguese dataset's electrical load forecasting.
        Uses building selection to manage dataset size.
    """
    print("🏗️ Selecting buildings for memory efficiency...")
    selected_metadata = ASHRAEDatasetAnalysis.select_buildings(
        building_metadata,
        max_buildings=ASHRAE_TRAINING_CONFIG["max_buildings"],
        strategy=ASHRAE_TRAINING_CONFIG["building_selection_strategy"]
    )
    
    # Prepare training data
    X_train, y_train = prepare_ashrae_data(
        train_data, test_data, selected_metadata, weather_train, weather_test, test_mode=False
    )
    
    # Prepare test data
    X_test, row_ids = prepare_ashrae_data(
        train_data, test_data, selected_metadata, weather_train, weather_test, test_mode=True
    )
    
    # Apply preprocessing steps to training data
    X_train = impute_ashrae_missing_values(X_train)
    X_train, train_stats = normalize_ashrae_features(X_train)  # Fit on training data
    X_train = remove_dew_temperature(X_train)
    
    # Normalize target variable (MinMax-only as per configuration)
    from sklearn.preprocessing import MinMaxScaler
    from config import MINMAX_FEATURE_RANGE
    target_scaler = MinMaxScaler(feature_range=MINMAX_FEATURE_RANGE)
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    
    # Apply preprocessing steps to test data using training statistics
    X_test = impute_ashrae_missing_values(X_test)
    X_test, _ = normalize_ashrae_features(X_test, train_stats)  # Use training stats
    X_test = remove_dew_temperature(X_test)
    
    return X_train, y_train_scaled, X_test, row_ids, target_scaler


def get_ashrae_lstm_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seq_length: int = None,
    train_fraction: float = None,
    val_fraction: float = None,
    max_samples: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare ASHRAE data for LSTM training with sliding windows.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        seq_length: Sequence length for LSTM
        train_fraction: Fraction of data for training
        val_fraction: Fraction of data for validation
        max_samples: Maximum number of samples to process (memory limit)
        
    Returns:
        Tuple of (X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, X_test_lstm, y_test_lstm)
    """
    # Use config values if not provided
    if seq_length is None:
        seq_length = ASHRAE_TRAINING_CONFIG["sequence_length"]
    if train_fraction is None:
        train_fraction = ASHRAE_DATA_SPLITS["train"]
    if val_fraction is None:
        val_fraction = ASHRAE_DATA_SPLITS["val"]
    if max_samples is None:
        max_samples = ASHRAE_TRAINING_CONFIG["max_samples"]
    
    # Limit data size to prevent memory issues - use SEQUENTIAL cropping for time series
    if len(X_train) > max_samples:
        print(f"Warning: Limiting data from {len(X_train)} to {max_samples} samples (sequential) for memory efficiency")
        X_train = X_train.iloc[:max_samples]  # Take first max_samples sequentially
        if hasattr(y_train, 'iloc'):
            y_train = y_train.iloc[:max_samples]
        else:
            y_train = y_train[:max_samples]
    
    # Combine features and targets for proper sequencing
    # Target must be FIRST column for sliding_windows2d_lstm
    if hasattr(y_train, 'values'):
        train_data = np.column_stack([y_train.values, X_train.values])
    else:
        train_data = np.column_stack([y_train, X_train.values])
    
    # Calculate split indices
    train_len = int(train_fraction * len(train_data))
    val_len = int(val_fraction * len(train_data))
    
    # Split data
    train_portion = train_data[:train_len]
    val_portion = train_data[train_len:train_len + val_len]
    test_portion = train_data[train_len + val_len:]
    
    # Create LSTM sequences
    X_train_lstm, y_train_lstm = sliding_windows2d_lstm(train_portion, seq_length)
    X_val_lstm, y_val_lstm = sliding_windows2d_lstm(val_portion, seq_length)
    X_test_lstm, y_test_lstm = sliding_windows2d_lstm(test_portion, seq_length)
    
    return (
        X_train_lstm,
        np.squeeze(y_train_lstm),
        X_val_lstm,
        np.squeeze(y_val_lstm),
        X_test_lstm,
        np.squeeze(y_test_lstm)
    )


def load_ashrae_dataset(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ASHRAE dataset files.
    
    Args:
        data_path: Path to ASHRAE dataset directory
        
    Returns:
        Tuple of (train_data, test_data, building_metadata, weather_train, weather_test)
    """
    train_data = pd.read_csv(data_path / 'train.csv')
    test_data = pd.read_csv(data_path / 'test.csv')
    building_metadata = pd.read_csv(data_path / 'building_metadata.csv')
    weather_train = pd.read_csv(data_path / 'weather_train.csv')
    weather_test = pd.read_csv(data_path / 'weather_test.csv')
    
    return train_data, test_data, building_metadata, weather_train, weather_test


def inverse_transform_ashrae_predictions(y_pred: np.ndarray, target_scaler=None) -> np.ndarray:
    """
    Inverse transform ASHRAE predictions from normalized scale back to original scale.
    
    Args:
        y_pred: Predictions in normalized scale [0, 1]
        target_scaler: Fitted MinMaxScaler for target variable
        
    Returns:
        Predictions in original scale
    """
    if target_scaler is not None:
        return target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        # If no scaler provided, assume predictions are already in original scale
        return y_pred
