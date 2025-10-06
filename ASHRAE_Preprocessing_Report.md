# ASHRAE Energy Prediction Dataset: Preprocessing Methodology

## Dataset Description

The ASHRAE Great Energy Predictor III dataset comprises multi-building energy consumption measurements collected across 1,449 buildings spanning 16 sites over a two-year period (2016-2018). The dataset contains 20,216,100 total measurements across four meter types: electricity (meter=0), chilled water (meter=1), steam (meter=2), and hot water (meter=3).

### Data Schema

**Primary Tables:**
- **train.csv**: Building energy consumption measurements
  - `building_id`: Unique building identifier (1,449 buildings)
  - `meter`: Meter type classification (0: electricity, 1: chilled water, 2: steam, 3: hot water)
  - `timestamp`: Measurement timestamp (hourly resolution)
  - `meter_reading`: Energy consumption in kWh (target variable)

- **building_metadata.csv**: Building characteristics
  - `site_id`: Geographic site identifier (16 sites)
  - `primary_use`: Building type classification (EnergyStar categories)
  - `square_feet`: Gross floor area
  - `year_built`: Construction year
  - `floor_count`: Number of floors

- **weather_train.csv**: Meteorological conditions
  - `site_id`: Weather station identifier
  - `air_temperature`: Ambient temperature (°C)
  - `cloud_coverage`: Sky coverage (oktas)
  - `precip_depth_1_hr`: Precipitation (mm)
  - `sea_level_pressure`: Atmospheric pressure (mbar)
  - `wind_direction`: Wind direction (0-360°)
  - `wind_speed`: Wind velocity (m/s)
  - `dew_temperature`: Dew point temperature (°C)

## Preprocessing Pipeline

### Data Integration

The preprocessing pipeline implements a multi-table join strategy:

1. **Primary Join**: `train.csv` ↔ `building_metadata.csv` on `building_id`
2. **Weather Integration**: Result ↔ `weather_train.csv` on `[site_id, timestamp]`
3. **Temporal Sorting**: Chronological ordering by `[building_id, timestamp]`

### Feature Engineering

**Temporal Features:**
- `hour`: Hour of day (0-23)
- `weekday`: Day of week (0-6)
- `is_holiday`: Binary indicator for US federal holidays (2016-2018)

**Building Features:**
- `square_feet`: Gross floor area (continuous)
- `year_built`: Construction year (continuous)
- `floor_count`: Number of floors (continuous)
- `primary_use`: One-hot encoded building type categories

**Weather Features:**
- `air_temperature`: Ambient temperature (°C)
- `cloud_coverage`: Sky coverage fraction
- `precip_depth_1_hr`: Hourly precipitation depth

**Feature Selection:**
Excluded features due to multicollinearity or redundancy:
- `sea_level_pressure`, `wind_direction`, `wind_speed`, `dew_temperature`
- Original `meter` field (replaced with categorical encoding)

### Data Normalization

**Scaling Strategy:**
- **Method**: Min-Max normalization to [0, 1] range
- **Fitting**: Scalers fitted exclusively on training data
- **Application**: Fitted scalers applied to validation and test sets
- **Rationale**: Prevents data leakage and ensures realistic generalization assessment

**Target Variable Processing:**
- **Variable**: `meter_reading` (kWh)
- **Transformation**: Min-Max scaling only (no logarithmic transformation)
- **Inverse Transformation**: Applied during model evaluation for metric computation

### Disjoint Building Splits

**Split Strategy:**
To evaluate model generalization across unseen buildings, the dataset employs disjoint building allocation:

- **Training Buildings**: 6 buildings (52,704 samples)
- **Validation Buildings**: 6 buildings (52,704 samples)  
- **Test Buildings**: 12 buildings (105,408 samples)
- **Total**: 24 buildings, 210,816 samples

**Building Selection Criteria:**
Buildings are allocated based on sample count distribution to ensure balanced representation across splits while maintaining temporal continuity within each building.

### Time Series Windowing

**Sequence Generation:**
For recurrent neural network architectures, temporal sequences are generated using a sliding window approach:

- **Window Length**: 23 timesteps
- **Step Size**: 1 timestep
- **Boundary Constraint**: No window crosses building boundaries
- **Output Format**: 3D tensors `(samples, timesteps, features)`

**Final Data Dimensions:**
- **Training**: (52,704, 23, 17)
- **Validation**: (52,704, 23, 17)
- **Test**: (105,408, 23, 17)

Where the feature dimension (17) includes:
- 3 temporal features (hour, weekday, is_holiday)
- 3 building features (square_feet, year_built, floor_count)
- 3 weather features (air_temperature, cloud_coverage, precip_depth_1_hr)
- 8 categorical features (primary_use one-hot encoding)

## Implementation Details

**Preprocessing Module**: `ashrae/preprocessing_ashrae_disjoint.py`
**Configuration**: `ashrae/ashrae_config.py`
**Model Integration**: Caller scripts handle dataset loading and model execution

**Key Design Principles:**
1. **Reproducibility**: Fixed random seeds and deterministic preprocessing
2. **Scalability**: Configurable building counts and sample sizes
3. **Modularity**: Separation of data loading and model training logic
4. **Consistency**: Unified preprocessing pipeline across all model architectures

This preprocessing methodology ensures robust evaluation of energy prediction models while maintaining realistic generalization assessment through disjoint building splits and proper data scaling practices.