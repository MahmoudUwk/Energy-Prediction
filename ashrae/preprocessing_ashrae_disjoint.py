"""ASHRAE preprocessing with DISJOINT building splits for train/val/test.

This module implements building-level splits to ensure train/val/test come from
completely separate buildings for better generalization testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from .ashrae_config import (
    ASHRAE_TRAINING_CONFIG,
    ASHRAE_FEATURE_CONFIG,
    ASHRAE_DATA_SPLITS,
)

from tools.preprocess_data2 import sliding_windows2d_lstm


def select_buildings_for_disjoint_splits(
    building_metadata: pd.DataFrame,
    train_data: pd.DataFrame,
    target_train_samples: int,
    target_val_samples: int,
    target_test_samples: int,
    strategy: str = "balanced"
) -> Tuple[List[int], List[int], List[int]]:
    """
    Select buildings to achieve target sample counts with disjoint splits.
    
    Args:
        building_metadata: Building metadata DataFrame
        train_data: Training data (electricity only, sorted)
        target_train_samples: Target number of training samples
        target_val_samples: Target number of validation samples  
        target_test_samples: Target number of test samples
        strategy: Selection strategy ("balanced", "largest_first")
        
    Returns:
        Tuple of (train_building_ids, val_building_ids, test_building_ids)
    """
    # Count samples per building
    building_counts = train_data.groupby('building_id').size().reset_index(name='sample_count')
    building_counts = building_counts.sort_values('sample_count', ascending=False)
    
    print(f"\n   📊 Building sample distribution:")
    print(f"      • Total buildings with electricity data: {len(building_counts)}")
    print(f"      • Samples per building - Min: {building_counts['sample_count'].min():,}, "
          f"Max: {building_counts['sample_count'].max():,}, "
          f"Mean: {building_counts['sample_count'].mean():.0f}")
    
    train_buildings = []
    val_buildings = []
    test_buildings = []
    
    train_samples = 0
    val_samples = 0
    test_samples = 0
    
    # Allocate buildings to splits
    for _, row in building_counts.iterrows():
        building_id = row['building_id']
        count = row['sample_count']
        
        # Allocate to the split that needs more samples
        if train_samples < target_train_samples:
            train_buildings.append(building_id)
            train_samples += count
        elif val_samples < target_val_samples:
            val_buildings.append(building_id)
            val_samples += count
        elif test_samples < target_test_samples:
            test_buildings.append(building_id)
            test_samples += count
        else:
            break  # We have enough samples
    
    print(f"\n   🏗️ DISJOINT BUILDING ALLOCATION:")
    print(f"      • Train: {len(train_buildings)} buildings → {train_samples:,} samples")
    print(f"      • Val: {len(val_buildings)} buildings → {val_samples:,} samples")
    print(f"      • Test: {len(test_buildings)} buildings → {test_samples:,} samples")
    print(f"      • Total: {len(train_buildings) + len(val_buildings) + len(test_buildings)} buildings")
    
    return train_buildings, val_buildings, test_buildings


def load_and_filter_ashrae_data(
    data_path: Path,
    target_meter: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and filter ASHRAE data for electricity only."""
    print("📂 Loading ASHRAE dataset...")
    
    train_data = pd.read_csv(data_path / 'train.csv')
    test_data = pd.read_csv(data_path / 'test.csv')
    building_metadata = pd.read_csv(data_path / 'building_metadata.csv')
    weather_train = pd.read_csv(data_path / 'weather_train.csv')
    weather_test = pd.read_csv(data_path / 'weather_test.csv')
    
    print(f"   ✓ Total training rows: {len(train_data):,}")
    print(f"   ✓ Total buildings: {len(building_metadata):,}")
    
    # Filter for electricity (meter = 0)
    print(f"\n   🔌 Filtering for electricity (meter = {target_meter})...")
    train_data = train_data[train_data['meter'] == target_meter].copy()
    print(f"   ✓ Electricity rows: {len(train_data):,}")
    
    # Convert timestamp to datetime and sort
    print(f"   📅 Sorting by building and timestamp (ensuring sequential data)...")
    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    train_data = train_data.sort_values(['building_id', 'timestamp']).reset_index(drop=True)
    
    return train_data, test_data, building_metadata, weather_train, weather_test


def prepare_split_data(
    train_data: pd.DataFrame,
    building_ids: List[int],
    building_metadata: pd.DataFrame,
    weather_train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for a specific split (train/val/test)."""
    # Filter for selected buildings
    split_data = train_data[train_data['building_id'].isin(building_ids)].copy()
    
    # Merge with building metadata
    X = split_data.merge(building_metadata, on='building_id', how='left')
    
    # Convert weather timestamp to datetime for merging
    weather_train_copy = weather_train.copy()
    weather_train_copy['timestamp'] = pd.to_datetime(weather_train_copy['timestamp'])
    
    # Merge with weather
    X = X.merge(weather_train_copy, on=['site_id', 'timestamp'], how='left')
    
    # Extract target (electricity consumption)
    y = split_data['meter_reading'].copy()
    
    # Feature engineering
    X['hour'] = X['timestamp'].dt.hour
    X['weekday'] = X['timestamp'].dt.weekday
    
    # US holidays
    holidays = [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
        "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
        "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
        "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
        "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
        "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
        "2019-01-01"
    ]
    X["is_holiday"] = (X['timestamp'].dt.date.astype("str").isin(holidays)).astype(int)
    
    # Drop unnecessary columns
    drop_cols = ['timestamp', 'meter', 'meter_reading', 'sea_level_pressure', 
                 'wind_direction', 'wind_speed', 'dew_temperature']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # One-hot encode categorical
    X = pd.get_dummies(X, columns=['primary_use'])
    
    return X, y


def preprocess_ashrae_disjoint_splits(
    target_samples: int = 250_000,
    train_fraction: float = 0.40,
    val_fraction: float = 0.20,
    test_fraction: float = 0.40
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, "MinMaxScaler"]:
    """
    Preprocess ASHRAE with DISJOINT building splits.
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, target_scaler)
    """
    print("=" * 80)
    print("ASHRAE PREPROCESSING WITH DISJOINT BUILDING SPLITS")
    print("=" * 80)
    
    data_path = Path(__file__).parent.parent / "dataset/ASHRAE/ashrae-energy-prediction"
    
    # Load and filter data
    train_data, _, building_metadata, weather_train, _ = load_and_filter_ashrae_data(
        data_path, target_meter=ASHRAE_TRAINING_CONFIG["target_meter"]
    )
    
    # Calculate target samples per split
    target_train = int(target_samples * train_fraction)
    target_val = int(target_samples * val_fraction)
    target_test = int(target_samples * test_fraction)
    
    print(f"\n🎯 TARGET SAMPLES:")
    print(f"   • Total: {target_samples:,}")
    print(f"   • Train ({train_fraction:.0%}): {target_train:,}")
    print(f"   • Val ({val_fraction:.0%}): {target_val:,}")
    print(f"   • Test ({test_fraction:.0%}): {target_test:,}")
    
    # Select buildings for disjoint splits
    train_buildings, val_buildings, test_buildings = select_buildings_for_disjoint_splits(
        building_metadata, train_data, target_train, target_val, target_test
    )
    
    # Prepare each split
    print(f"\n📦 Preparing split data...")
    X_train, y_train = prepare_split_data(train_data, train_buildings, building_metadata, weather_train)
    X_val, y_val = prepare_split_data(train_data, val_buildings, building_metadata, weather_train)
    X_test, y_test = prepare_split_data(train_data, test_buildings, building_metadata, weather_train)
    
    print(f"   ✓ Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"   ✓ Val: {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"   ✓ Test: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
    
    # Ensure all splits have same features
    all_features = set(X_train.columns) | set(X_val.columns) | set(X_test.columns)
    for col in all_features:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_val.columns:
            X_val[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0
    
    # Sort columns to ensure consistency
    X_train = X_train[sorted(X_train.columns)]
    X_val = X_val[sorted(X_val.columns)]
    X_test = X_test[sorted(X_test.columns)]
    
    # Impute missing values
    print(f"\n🔧 Preprocessing features...")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    # Preserve raw building_id for grouping; do not scale it
    numeric_cols_wo_id = [c for c in numeric_cols if c != 'building_id']
    
    # Calculate train means, replacing any remaining NaNs with 0
    train_means = X_train[numeric_cols_wo_id].mean().fillna(0)
    
    X_train[numeric_cols_wo_id] = X_train[numeric_cols_wo_id].fillna(train_means)
    X_val[numeric_cols_wo_id] = X_val[numeric_cols_wo_id].fillna(train_means)
    X_test[numeric_cols_wo_id] = X_test[numeric_cols_wo_id].fillna(train_means)
    
    # Final check: replace any remaining NaNs with 0
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)
    
    # Normalize features (MinMax) - FIT ON TRAIN ONLY, APPLY TO VAL/TEST
    from sklearn.preprocessing import MinMaxScaler
    from config import MINMAX_FEATURE_RANGE
    
    feature_scaler = MinMaxScaler(feature_range=MINMAX_FEATURE_RANGE)
    feature_scaler.fit(X_train[numeric_cols_wo_id])
    
    # Transform each split
    X_train[numeric_cols_wo_id] = feature_scaler.transform(X_train[numeric_cols_wo_id])
    X_val[numeric_cols_wo_id] = feature_scaler.transform(X_val[numeric_cols_wo_id])
    X_test[numeric_cols_wo_id] = feature_scaler.transform(X_test[numeric_cols_wo_id])
    
    # Normalize targets (MinMax) - FIT ON TRAIN ONLY, APPLY TO VAL/TEST
    target_scaler = MinMaxScaler(feature_range=MINMAX_FEATURE_RANGE)
    target_scaler.fit(y_train.values.reshape(-1, 1))
    
    # Transform each split
    y_train_scaled = target_scaler.transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    
    print(f"   ✓ Features normalized (MinMax [{MINMAX_FEATURE_RANGE[0]}, {MINMAX_FEATURE_RANGE[1]}])")
    print(f"   ✓ Targets normalized (MinMax [{MINMAX_FEATURE_RANGE[0]}, {MINMAX_FEATURE_RANGE[1]}])")
    
    print("\n" + "=" * 80)
    
    return X_train, y_train_scaled, X_val, y_val_scaled, X_test, y_test_scaled, target_scaler


def get_ashrae_lstm_data_disjoint(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    seq_length: int = 23,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build LSTM sequences per split ensuring windows DO NOT cross building boundaries.
    Assumes 'building_id' column exists in X_* and is NOT scaled.
    """
    def windows_for_split(X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features_cols = [c for c in X.columns if c != 'building_id']
        X_np = X[features_cols].values
        bld = X['building_id'].values
        # Concatenate target first for compatibility with sliding_windows2d_lstm
        data = np.column_stack([y, X_np])
        X_list, y_list = [], []
        # Generate windows per building
        for building in np.unique(bld):
            idx = np.where(bld == building)[0]
            if idx.size <= seq_length:
                continue
            data_b = data[idx, :]
            X_b, y_b = sliding_windows2d_lstm(data_b, seq_length)
            X_list.append(X_b)
            y_list.append(y_b)
        if not X_list:
            return np.zeros((0, seq_length, data.shape[1])), np.zeros((0, 1))
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all

    X_tr_lstm, y_tr_lstm = windows_for_split(X_train, y_train)
    X_va_lstm, y_va_lstm = windows_for_split(X_val, y_val)
    X_te_lstm, y_te_lstm = windows_for_split(X_test, y_test)

    return (
        X_tr_lstm,
        np.squeeze(y_tr_lstm),
        X_va_lstm,
        np.squeeze(y_va_lstm),
        X_te_lstm,
        np.squeeze(y_te_lstm),
    )


if __name__ == "__main__":
    # Test the preprocessing
    X_train, y_train, X_val, y_val, X_test, y_test, target_scaler = preprocess_ashrae_disjoint_splits()
    
    print("\n✅ PREPROCESSING COMPLETE")
    print(f"   • Train: {len(y_train):,} samples")
    print(f"   • Val: {len(y_val):,} samples")
    print(f"   • Test: {len(y_test):,} samples")
    print(f"   • Features: {X_train.shape[1]}")

