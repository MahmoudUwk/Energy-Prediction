"""ASHRAE Dataset Configuration and Analysis"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple

# Base paths
BASE_DIR = Path(__file__).resolve().parent
ASHRAE_DATA_ROOT = BASE_DIR / "dataset/ASHRAE/ashrae-energy-prediction"
ASHRAE_RESULTS_ROOT = BASE_DIR / "results/ashrae"

# =============================================================================
# ASHRAE DATASET SIZE ANALYSIS
# =============================================================================

class ASHRAEDatasetAnalysis:
    """Analysis of ASHRAE Great Energy Predictor III dataset size and structure."""
    
    # Raw dataset sizes (from official ASHRAE data)
    RAW_TRAIN_ROWS = 20_216_100  # ~20.2 million training readings
    RAW_TEST_ROWS = 41_697_600    # ~41.7 million test readings
    RAW_BUILDINGS = 1_449         # Number of unique buildings
    RAW_SITES = 16                # Number of different sites/climate zones
    RAW_METER_TYPES = 4           # Electricity (0), ChilledWater (1), Steam (2), HotWater (3)
    
    # Expected processed dataset sizes
    # After pivot: one row per building_id/timestamp/meter_type combination
    # Assuming hourly readings for ~1 year per building/meter_type
    ESTIMATED_HOURLY_READINGS_PER_YEAR = 8760  # 365 * 24
    ESTIMATED_PROCESSED_TRAIN_ROWS = RAW_BUILDINGS * RAW_METER_TYPES * ESTIMATED_HOURLY_READINGS_PER_YEAR
    ESTIMATED_PROCESSED_TRAIN_ROWS = 1_449 * 4 * 8760  # ~50.8 million rows (theoretical maximum)
    
    # Realistic processed dataset (considering data gaps, different time periods)
    REALISTIC_PROCESSED_TRAIN_ROWS = 15_000_000  # ~15 million (estimated)
    REALISTIC_PROCESSED_TEST_ROWS = 30_000_000   # ~30 million (estimated)
    
    # After electricity-only filtering (target = electricity, features = all meter types)
    ELECTRICITY_ONLY_TRAIN_ROWS = REALISTIC_PROCESSED_TRAIN_ROWS // 4  # ~3.75 million
    ELECTRICITY_ONLY_TEST_ROWS = REALISTIC_PROCESSED_TEST_ROWS // 4    # ~7.5 million
    
    @classmethod
    def print_analysis(cls) -> None:
        """Print comprehensive ASHRAE dataset analysis."""
        print("=" * 80)
        print("ASHRAE DATASET SIZE ANALYSIS")
        print("=" * 80)
        
        print("\n📊 RAW DATASET:")
        print(f"   • Training rows: {cls.RAW_TRAIN_ROWS:,} (~20.2 million)")
        print(f"   • Test rows: {cls.RAW_TEST_ROWS:,} (~41.7 million)")
        print(f"   • Buildings: {cls.RAW_BUILDINGS:,}")
        print(f"   • Sites: {cls.RAW_SITES}")
        print(f"   • Meter types: {cls.RAW_METER_TYPES}")
        
        print(f"\n🏗️ AFTER PREPROCESSING:")
        print(f"   • Theoretical max (hourly × 1yr): {cls.ESTIMATED_PROCESSED_TRAIN_ROWS:,}")
        print(f"   • Realistic estimate: {cls.REALISTIC_PROCESSED_TRAIN_ROWS:,}")
        print(f"   • After electricity-only target: {cls.ELECTRICITY_ONLY_TRAIN_ROWS:,}")
        
        print(f"\n⚡ TRAINING CONFIGURATION:")
        print(f"   • Max samples for training: {ASHRAE_TRAINING_CONFIG['max_samples']:,}")
        train_samples = int(ASHRAE_TRAINING_CONFIG['max_samples'] * ASHRAE_TRAINING_CONFIG['train_fraction'])
        val_samples = int(ASHRAE_TRAINING_CONFIG['max_samples'] * ASHRAE_TRAINING_CONFIG['val_fraction'])
        test_samples = int(ASHRAE_TRAINING_CONFIG['max_samples'] * ASHRAE_TRAINING_CONFIG['test_fraction'])
        
        print(f"   • Final training samples: {train_samples:,}")
        print(f"   • Final validation samples: {val_samples:,}")
        print(f"   • Final test samples: {test_samples:,}")
        
        print(f"\n📈 MEMORY ESTIMATES:")
        print(f"   • Full dataset: ~{cls.REALISTIC_PROCESSED_TRAIN_ROWS / 1_000_000:.1f}M rows")
        print(f"   • Limited dataset: {ASHRAE_TRAINING_CONFIG['max_samples']:,} rows (manageable)")
        print(f"   • Portuguese dataset: ~82,944 rows")
        print(f"   • Size ratio (ASHRAE/Portuguese): {ASHRAE_TRAINING_CONFIG['max_samples'] / 82_944:.1f}x")
        
        print("\n" + "=" * 80)

    @classmethod
    def select_buildings(
        cls, 
        building_metadata: pd.DataFrame,
        max_buildings: int = None,
        strategy: str = "diverse"
    ) -> pd.DataFrame:
        """
        Select a subset of buildings based on strategy.
        
        Args:
            building_metadata: Building metadata DataFrame
            max_buildings: Maximum number of buildings to select
            strategy: Selection strategy ("diverse", "random", "largest")
            
        Returns:
            Filtered building metadata
        """
        if max_buildings is None:
            max_buildings = ASHRAE_TRAINING_CONFIG["max_buildings"]
        
        if max_buildings >= len(building_metadata):
            print(f"   • Using all {len(building_metadata)} buildings (max_buildings >= available)")
            return building_metadata
        
        if strategy == "diverse":
            # Select buildings with different primary_use types
            unique_uses = building_metadata['primary_use'].unique()
            selected_buildings = []
            
            for use_type in unique_uses:
                buildings_by_use = building_metadata[building_metadata['primary_use'] == use_type]
                n_to_select = max(1, max_buildings // len(unique_uses))
                selected_buildings.extend(buildings_by_use.head(n_to_select).to_list())
                
                if len(selected_buildings) >= max_buildings:
                    break
            
            selected_buildings = selected_buildings[:max_buildings]
            
        elif strategy == "random":
            selected_buildings = building_metadata['building_id'].sample(
                n=min(max_buildings, len(building_metadata)), 
                random_state=42
            ).tolist()
            
        elif strategy == "largest":
            # Select buildings with most data (assume square_feet indicates building size)
            selected_buildings = building_metadata.nlargest(max_buildings, 'square_feet')['building_id'].tolist()
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        selected_metadata = building_metadata[building_metadata['building_id'].isin(selected_buildings)]
        
        print(f"   • Selected {len(selected_metadata)} buildings using '{strategy}' strategy")
        print(f"   • Primary use types: {sorted(selected_metadata['primary_use'].unique())}")
        
        return selected_metadata

# =============================================================================
# ASHRAE TRAINING CONFIGURATION
# =============================================================================

# Data split configuration (same as Portuguese dataset for consistency)
ASHRAE_DATA_SPLITS = {
    "train": 0.40,    # 40% for training (100k of 250k)
    "val": 0.20,       # 20% for validation (50k of 250k)
    "test": 0.40,      # 40% for testing (100k of 250k)
}

# Verify splits sum to 1.0
assert abs(sum(ASHRAE_DATA_SPLITS.values()) - 1.0) < 1e-6, "Data splits must sum to 1.0"

# LSTM training configuration (consistent with Portuguese dataset)
ASHRAE_TRAINING_CONFIG = {
    # Data parameters
    "sequence_length": 23,           # Same as Portuguese dataset
    "max_samples": 250_000,         # Increased limit for experiments
    "train_fraction": ASHRAE_DATA_SPLITS["train"],
    "val_fraction": ASHRAE_DATA_SPLITS["val"], 
    "test_fraction": ASHRAE_DATA_SPLITS["test"],
    
    # Building selection for memory management
    "max_buildings": 100,           # Limit number of buildings to use
    "building_selection_strategy": "diverse",  # "diverse", "random", "largest"
    "ensure_diversity": True,         # Ensure different building types
    
    # Model hyperparameters (from paper Table II)
    "lstm_units": 72,               # Fixed as per paper
    "lstm_layers": 1,               # Fixed as per paper
    "learning_rate": 0.01,          # Fixed as per paper
    "batch_size": 32,               # 2^5 = 32
    
    # Training parameters
    "epochs": 200,
    "patience": 50,
    "verbose": 1,
    "validation_split": 0.0,        # We use explicit validation set
    "use_callbacks": True,
    
    # Feature configuration
    "target_meter": 0,              # Electricity (meter = 0)
    "feature_meters": [0, 1, 2, 3], # All meter types as features
    "include_weather": True,
    "include_building": True,
    "include_temporal": True,
    
    # Normalization (consistent with Portuguese dataset)
    "feature_range": (0.0, 1.0),   # MinMaxScaler range
    "normalize_target": True,       # Normalize target to [0,1]
    
    # Output configuration
    "save_results": True,
    "save_models": True,
    "plot_results": False,
}

# Sliding window configuration
ASHRAE_SLIDING_WINDOW_CONFIG = {
    "sequence_length": ASHRAE_TRAINING_CONFIG["sequence_length"],
    "forecast_horizon": 1,          # Single-step ahead forecasting
    "target_position": 0,           # Target is first column (as per sliding_windows2d_lstm)
}

# Feature engineering configuration
ASHRAE_FEATURE_CONFIG = {
    # Temporal features
    "hour": True,
    "weekday": True,
    "is_holiday": True,
    "holidays": [
        "2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
        "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
        "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
        "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
        "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
        "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
        "2019-01-01"
    ],
    
    # Building features
    "square_feet_log_transform": False,
    "year_built": True,
    "floor_count": True,
    "primary_use_onehot": True,
    
    # Weather features  
    "air_temperature": True,
    "cloud_coverage": True,
    "precip_depth_1_hr": True,
    
    # Features to drop
    "drop_features": [
        "timestamp", "sea_level_pressure", "wind_direction", "wind_speed"
    ],
    
    # Meter configuration
    "meter_mapping": {
        0: "Electricity",
        1: "ChilledWater", 
        2: "Steam",
        3: "HotWater"
    }
}

# Evaluation metrics configuration
ASHRAE_METRICS_CONFIG = {
    "required_metrics": ["MSE", "RMSE", "MAPE", "R2", "RMSLE"],
    "target_transform": "inverse",  # Transform back to original scale for evaluation
    "precision": 4,                 # Decimal places for results
}

# Benchmark configuration
ASHRAE_BENCHMARK_CONFIG = {
    "linear_regression": {
        "fit_intercept": True,
        "normalize": False,          # Already normalized
        "copy_X": True,
        "n_jobs": -1,                # Use all CPU cores
    },
    
    "model_comparison": {
        "baseline_models": ["LinearRegression"],
        "advanced_models": ["LSTM"],
        "save_comparison": True,
    }
}

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_ashrae_data_splits() -> Tuple[float, float, float]:
    """Get train, validation, test split fractions."""
    return (
        ASHRAE_DATA_SPLITS["train"],
        ASHRAE_DATA_SPLITS["val"], 
        ASHRAE_DATA_SPLITS["test"]
    )

def get_ashrae_sample_size(max_samples: int = None) -> int:
    """Get the maximum number of samples to use for training."""
    if max_samples is None:
        return ASHRAE_TRAINING_CONFIG["max_samples"]
    return min(max_samples, ASHRAE_TRAINING_CONFIG["max_samples"])

def calculate_ashrae_sample_shapes(total_samples: int) -> Dict[str, int]:
    """Calculate train/val/test sample counts given total samples."""
    train_count = int(total_samples * ASHRAE_DATA_SPLITS["train"])
    val_count = int(total_samples * ASHRAE_DATA_SPLITS["val"])
    test_count = total_samples - train_count - val_count  # Remaining samples
    
    return {
        "train": train_count,
        "val": val_count,
        "test": test_count,
        "total": total_samples
    }

def print_ashrae_config_summary() -> None:
    """Print a summary of ASHRAE configuration."""
    print("=" * 60)
    print("ASHRAE DATASET CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Data Splits:")
    splits = get_ashrae_data_splits()
    print(f"   • Train: {splits[0]:.1%}")
    print(f"   • Validation: {splits[1]:.1%}")
    print(f"   • Test: {splits[2]:.1%}")
    
    print(f"\n🏗️ Model Configuration:")
    print(f"   • LSTM Units: {ASHRAE_TRAINING_CONFIG['lstm_units']}")
    print(f"   • LSTM Layers: {ASHRAE_TRAINING_CONFIG['lstm_layers']}")
    print(f"   • Sequence Length: {ASHRAE_TRAINING_CONFIG['sequence_length']}")
    print(f"   • Learning Rate: {ASHRAE_TRAINING_CONFIG['learning_rate']}")
    print(f"   • Batch Size: {ASHRAE_TRAINING_CONFIG['batch_size']}")
    print(f"   • Max Samples: {ASHRAE_TRAINING_CONFIG['max_samples']:,}")
    
    print(f"\n📈 Sample Distribution:")
    shapes = calculate_ashrae_sample_shapes(ASHRAE_TRAINING_CONFIG['max_samples'])
    print(f"   • Training: {shapes['train']:,}")
    print(f"   • Validation: {shapes['val']:,}")
    print(f"   • Test: {shapes['test']:,}")
    print(f"   • Total: {shapes['total']:,}")
    
    print(f"\n🎯 Target Configuration:")
    print(f"   • Target Meter: {ASHRAE_TRAINING_CONFIG['target_meter']} (Electricity)")
    print(f"   • Feature Meters: {ASHRAE_TRAINING_CONFIG['feature_meters']}")
    print(f"   • Normalize Target: {ASHRAE_TRAINING_CONFIG['normalize_target']}")
    print(f"   • Feature Range: {ASHRAE_TRAINING_CONFIG['feature_range']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Run analysis when script is executed directly
    ASHRAEDatasetAnalysis.print_analysis()
    print_ashrae_config_summary()
