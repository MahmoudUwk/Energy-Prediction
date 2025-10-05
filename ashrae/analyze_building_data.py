"""Analyze ASHRAE building data to determine optimal building selection for 250K samples."""

import pandas as pd
from pathlib import Path
import numpy as np

def analyze_ashrae_building_data():
    """Analyze how many samples each building provides."""
    print("=" * 80)
    print("ASHRAE BUILDING DATA ANALYSIS")
    print("=" * 80)
    
    data_path = Path("dataset/ASHRAE/ashrae-energy-prediction")
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_data = pd.read_csv(data_path / 'train.csv')
    building_metadata = pd.read_csv(data_path / 'building_metadata.csv')
    
    print(f"   ✓ Total training rows: {len(train_data):,}")
    print(f"   ✓ Total buildings: {building_metadata['building_id'].nunique():,}")
    
    # Filter for electricity only (meter = 0)
    print("\n2. Filtering for electricity (meter = 0)...")
    electricity_data = train_data[train_data['meter'] == 0].copy()
    print(f"   ✓ Electricity rows: {len(electricity_data):,}")
    
    # Convert timestamp to datetime and sort
    print("\n3. Analyzing temporal sequence...")
    electricity_data['timestamp'] = pd.to_datetime(electricity_data['timestamp'])
    electricity_data = electricity_data.sort_values(['building_id', 'timestamp'])
    
    # Count samples per building
    print("\n4. Counting samples per building...")
    building_counts = electricity_data.groupby('building_id').size().reset_index(name='sample_count')
    building_counts = building_counts.sort_values('sample_count', ascending=False)
    
    # Merge with building metadata
    building_analysis = building_counts.merge(building_metadata, on='building_id', how='left')
    
    print(f"\n   📊 SAMPLE DISTRIBUTION:")
    print(f"      • Min samples per building: {building_counts['sample_count'].min():,}")
    print(f"      • Max samples per building: {building_counts['sample_count'].max():,}")
    print(f"      • Mean samples per building: {building_counts['sample_count'].mean():.0f}")
    print(f"      • Median samples per building: {building_counts['sample_count'].median():.0f}")
    
    # Calculate cumulative samples
    building_counts['cumulative_samples'] = building_counts['sample_count'].cumsum()
    
    # Find how many buildings needed for 250K samples
    target_samples = 250_000
    buildings_needed = (building_counts['cumulative_samples'] >= target_samples).idxmax() + 1
    actual_samples = building_counts.loc[buildings_needed - 1, 'cumulative_samples']
    
    print(f"\n   🎯 TARGET: {target_samples:,} samples")
    print(f"      • Buildings needed: {buildings_needed}")
    print(f"      • Actual samples: {actual_samples:,}")
    
    # Show top buildings
    print(f"\n   🔝 TOP 20 BUILDINGS BY SAMPLE COUNT:")
    for idx, row in building_analysis.head(20).iterrows():
        print(f"      {idx+1:3d}. Building {int(row['building_id']):4d}: "
              f"{int(row['sample_count']):6,} samples | {row['primary_use']:20s} | "
              f"{int(row['square_feet']):10,} sqft")
    
    # Analysis for different split strategies
    print(f"\n   📈 BUILDING SPLIT STRATEGIES FOR 250K SAMPLES:")
    
    # Strategy 1: All from same buildings (sequential)
    print(f"\n      Strategy 1: Sequential from {buildings_needed} buildings")
    print(f"         • Total samples: {actual_samples:,}")
    print(f"         • Train (40%): ~{int(actual_samples * 0.4):,} samples")
    print(f"         • Val (20%): ~{int(actual_samples * 0.2):,} samples")
    print(f"         • Test (40%): ~{int(actual_samples * 0.4):,} samples")
    print(f"         • Data is sequential within each building")
    
    # Strategy 2: Disjoint buildings
    train_target = int(target_samples * 0.4)  # 100K
    val_target = int(target_samples * 0.2)    # 50K
    test_target = int(target_samples * 0.4)   # 100K
    
    # Find buildings for train
    train_buildings = (building_counts['cumulative_samples'] >= train_target).idxmax() + 1
    train_samples = building_counts.loc[train_buildings - 1, 'cumulative_samples']
    
    # Find buildings for val (starting after train)
    val_start = train_buildings
    val_counts = building_counts.iloc[val_start:].copy()
    val_counts['cumulative_samples'] = val_counts['sample_count'].cumsum()
    val_buildings = (val_counts['cumulative_samples'] >= val_target).idxmax() + 1
    val_samples = val_counts.loc[val_buildings - 1 + val_start, 'sample_count'].sum() if val_buildings > 0 else val_counts['sample_count'].sum()
    
    # Find buildings for test (starting after val)
    test_start = val_start + val_buildings
    test_counts = building_counts.iloc[test_start:].copy()
    test_counts['cumulative_samples'] = test_counts['sample_count'].cumsum()
    test_buildings = (test_counts['cumulative_samples'] >= test_target).idxmax() + 1
    test_samples = test_counts.loc[test_buildings - 1 + test_start, 'sample_count'].sum() if test_buildings > 0 else test_counts['sample_count'].sum()
    
    print(f"\n      Strategy 2: Disjoint buildings (separate buildings for train/val/test)")
    print(f"         • Train: {train_buildings} buildings, ~{int(train_samples):,} samples")
    print(f"         • Val: {val_buildings} buildings, ~{int(val_samples):,} samples")  
    print(f"         • Test: {test_buildings} buildings, ~{int(test_samples):,} samples")
    print(f"         • Total buildings needed: {train_buildings + val_buildings + test_buildings}")
    print(f"         • Each split uses different buildings (better generalization)")
    
    # Save analysis
    output_path = Path("results/ashrae/building_analysis.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    building_analysis.to_csv(output_path, index=False)
    print(f"\n   ✓ Building analysis saved to: {output_path}")
    
    # Save recommended config
    print(f"\n   💡 RECOMMENDED CONFIGURATION:")
    print(f"      For DISJOINT building splits (better generalization):")
    print(f"      • Use ~{train_buildings + val_buildings + test_buildings} buildings total")
    print(f"      • Assign first {train_buildings} buildings to train")
    print(f"      • Assign next {val_buildings} buildings to validation")
    print(f"      • Assign next {test_buildings} buildings to test")
    print(f"      • This ensures each split learns from different building patterns")
    
    print("\n" + "=" * 80)
    return building_counts, building_analysis

if __name__ == "__main__":
    building_counts, building_analysis = analyze_ashrae_building_data()

