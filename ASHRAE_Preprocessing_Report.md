## ASHRAE Dataset and Preprocessing Summary

### Dataset Schema

**train.csv**
- `building_id` - Foreign key for building metadata
- `meter` - Meter ID code: {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}
- `timestamp` - When measurement was taken
- `meter_reading` - Target variable. Energy consumption in kWh (or equivalent)
  - Note: Site 0 electric meter readings are in kBTU
  - Contains real data with measurement error

**building_metadata.csv**
- `site_id` - Foreign key for weather files
- `building_id` - Foreign key for training.csv
- `primary_use` - Primary category of activities (EnergyStar property type)
- `square_feet` - Gross floor area of the building
- `year_built` - Year building was opened
- `floor_count` - Number of floors of the building

**weather_[train/test].csv**
- `site_id` - Weather station site identifier
- `air_temperature` - Degrees Celsius
- `cloud_coverage` - Portion of sky covered in clouds, in oktas
- `dew_temperature` - Degrees Celsius
- `precip_depth_1_hr` - Millimeters
- `sea_level_pressure` - Millibar/hectopascals
- `wind_direction` - Compass direction (0-360)
- `wind_speed` - Meters per second

**test.csv**
- `row_id` - Row id for submission file
- `building_id` - Building id code
- `meter` - The meter id code
- `timestamp` - Timestamps for test data period

### Dataset Overview
- Source: ASHRAE Great Energy Predictor III - Great Energy Predictor III Competition
- Files used: `train.csv`, `test.csv`, `building_metadata.csv`, `weather_train.csv`, `weather_test.csv`
- Target: `meter_reading` (Energy consumption in kWh or equivalent, log-transformed via log1p)
- Scope: Multi-building, multi-site, multi-meter types; time-stamped readings
- Note: Site 0 electric meter readings are in kBTU (not kWh)

### Merging Strategy
1. Join `train/test` with `building_metadata` on `building_id`
2. Join with `weather_train/test` on `[site_id, timestamp]`
3. Convert `timestamp` to datetime; sort training by time

### Feature Engineering
- Temporal: `hour`, `weekday`, `is_holiday` (US holidays 2016-2019)
- Building: `square_feet` log1p, `year_built`, `floor_count`
- Weather: `air_temperature`, `cloud_coverage`, `precip_depth_1_hr`
- Categorical One-hot:
  - `primary_use` (EnergyStar property type categories)
  - `meter_category` (0: electricity, 1: chilledwater, 2: steam, 3: hotwater)

### Feature Selection / Drops
- Dropped: `timestamp`, `sea_level_pressure`, `wind_direction`, `wind_speed`, `dew_temperature` (correlated), original `meter`

### Normalization
- Method: MinMaxScaler (0.0 to 1.0 range), FIT ON TRAIN ONLY; applied to val/test
- Columns: numeric features only; `building_id` is excluded from scaling and used for windowing
- Important: Uses `MINMAX_FEATURE_RANGE` from `config.py`
- Target normalization: FIT ON TRAIN ONLY; applied to val/test (val/test may exceed [0,1] if outside train range)

### Target Preparation
- Target: `meter_reading` (kWh); no log transform (MinMax only)
- Train scaler fitted on train target only; inverse-transform predictions for metrics

### Sequencing (for LSTM)
- Sequence length: 23
- Disjoint building splits: Configurable via `ashrae_config.py`
  - `train_buildings`: 6 (configurable)
  - `val_buildings`: None (auto-allocated)
  - `test_buildings`: None (auto-allocated)
- Windows generated per building; no window crosses building boundary
- Current allocation: ~52k train, ~26k val, ~52k test windows

### Output Shapes (disjoint-split windows)
- `X_train_lstm`: (~52k, 23, ~17) - 6 train buildings × ~8.7k samples each
- `y_train_lstm`: (~52k,)
- `X_val_lstm`: (~26k, 23, ~17) - 3 val buildings × ~8.7k samples each
- `y_val_lstm`: (~26k,)
- `X_test_lstm`: (~52k, 23, ~17) - 6 test buildings × ~8.7k samples each
- `y_test_lstm`: (~52k,)

### Consistency vs Original Dataset
- Original (Portuguese): MinMax scaling (0.0-1.0); sequential windows; single building
- ASHRAE: MinMax scaling (0.0-1.0) with train-only fitting; disjoint building splits; per-building windows; multi-building
- ✅ Both datasets use consistent scaling methodology with proper train-only fitting
- ✅ ASHRAE uses disjoint building evaluation for better generalization testing

### Implementation Notes
- Core preprocessing: `ashrae/preprocessing_ashrae_disjoint.py`
- Caller scripts: `ashrae/call_svr_ashrae.py`, `ashrae/call_samfor_ashrae.py`, `ashrae/call_lstm_search_ashrae.py`
- Configuration: `ashrae/ashrae_config.py` with configurable building counts
- Disjoint building evaluation for better generalization testing
- All scripts use relative imports and unified preprocessing pipeline


