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

### Target Preparation
- Target: `meter_reading` (kWh); no log transform (MinMax only)
- Train scaler fitted on train target only; inverse-transform predictions for metrics

### Sequencing (for LSTM)
- Sequence length: 23
- Disjoint building splits: Train=12, Val=6, Test=12 buildings (approx. 105k/53k/105k rows)
- Windows generated per building; no window crosses building boundary

### Output Shapes (disjoint-split windows)
- `X_train_lstm`: (~105132, 23, ~17)
- `y_train_lstm`: (~105132,)
- `X_val_lstm`: (~52566, 23, ~17)
- `y_val_lstm`: (~52566,)
- `X_test_lstm`: (~105132, 23, ~17)
- `y_test_lstm`: (~105132,)

### Consistency vs Original Dataset
- Original: MinMax scaling (0.0-1.0); sequential windows
- ASHRAE: MinMax scaling (0.0-1.0) with train-only fitting; disjoint building splits; per-building windows
- ✅ Both datasets use consistent scaling methodology with proper train-only fitting

### Implementation Notes
- Code: `ashrae/preprocessing_ashrae.py` (moved from tools/ for organization)
- Test script: `ashrae/test_ashrae_preprocessing.py`
- Training scripts: `ashrae/train_ashrae_lstm_comb.py`, `ashrae/train_ashrae_lstm.py`
- Preprocessing preserves original project code for Portuguese dataset
- All import paths updated to reflect new ashrae/ folder structure


