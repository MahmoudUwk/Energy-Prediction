# Dataset Preprocessing Comparison: Portuguese vs ASHRAE

## Overview
This document compares the preprocessing pipelines for the original Portuguese dataset and the ASHRAE dataset in the ModLSTM Energy Prediction Toolkit.

## Key Differences Summary

### 1. **Dataset Structure & Complexity**
| Aspect | Portuguese Dataset | ASHRAE Dataset |
|--------|-------------------|----------------|
| **Source** | Single building/utility data | Multi-building, multi-site competition data |
| **Scope** | 3 core electrical measurements | 4 meter types across multiple building types |
| **Data Sources** | 1 CSV file | 5 CSV files (train, test, metadata, weather×2) |
| **Building Context** | Single entity context | 1,449 buildings across 16 sites |

### 2. **Core Features**
| Feature Type | Portuguese Dataset | ASHRAE Dataset |
|--------------|-------------------|----------------|
| **Target Variable** | P (Active Power) | meter_reading (electricity only) |
| **Power Features** | Q (Reactive Power), V (Voltage) | chilledwater, steam, hotwater (all log-transformed) |
| **Temporal Features** | Minute, DOW (Day of Week), H (Hour) | hour, weekday, is_holiday |
| **Building Features** | None | square_feet (log1p), year_built, floor_count, primary_use (one-hot) |
| **Weather Features** | None | air_temperature, cloud_coverage, precip_depth_1_hr |
| **Feature Approach** | 3 electrical measurements | **Multi-variate**: electricity target + other meter types as features |

### 3. **Feature Engineering**
#### Portuguese Dataset
```python
# Simple temporal features only
df["Minute"] = data.index.minute.astype(float)
df["DOW"] = data.index.dayofweek.astype(float)  
df["H"] = data.index.hour.astype(float)
```

#### ASHRAE Dataset
```python
# Multi-variate feature engineering
# Pivot: electricity as target, all meter types as features
# Temporal: hour, weekday, is_holiday (US holidays 2016-2019)
# Building: square_feet (log1p), year_built, floor_count
# Categorical: primary_use (one-hot)
# Weather integration with building metadata
# Features: chilledwater, steam, hotwater (log-transformed)
```

### 4. **Data Complexity**
| Metric | Portuguese Dataset | ASHRAE Dataset |
|--------|-------------------|----------------|
| **Base Features** | 3 (P, Q, V) | ~20-22 (all meter types + building/weather) |
| **Final Feature Count** | 6 (including temporal) | ~32-35 (multi-variate features) |
| **Target Variable** | log1p(P) | log1p(electricity) |
| **Multi-variate Features** | Q, V (related electrical measurements) | chilledwater, steam, hotwater (different energy types) |
| **Missing Value Handling** | Built into pipeline | Explicit mean imputation |
| **Feature Selection** | None (uses all 3 core features) | Drops correlated features + pivot operation |

### 5. **Preprocessing Pipeline Steps**

#### Portuguese Dataset (Simple)
1. Load single CSV file
2. Extract P, Q, V columns
3. Create temporal features (Minute, DOW, H)
4. Apply MinMaxScaler (0.0-1.0)
5. Create sliding windows for LSTM

#### ASHRAE Dataset (Complex)
1. Load 5 CSV files (train, test, metadata, weather×2)
2. **Pivot meter readings**: electricity as target, all meter types as features
3. Merge datasets on building_id and site_id/timestamp
4. Feature engineering:
   - Temporal features + holiday detection
   - Building characteristics
   - Log transform square_feet
   - One-hot encoding for primary_use
   - Log transform all meter readings
5. Missing value imputation
6. Drop correlated/unnecessary features
7. Apply MinMaxScaler (0.0-1.0) - **FIXED for consistency**
8. Create sliding windows for LSTM

### 6. **Target Variable Characteristics**
| Aspect | Portuguese Dataset | ASHRAE Dataset |
|--------|-------------------|----------------|
| **Variable** | Active Power (P) | meter_reading (electricity only) |
| **Units** | Likely kW/kWh | kWh (Site 0 electricity in kBTU) |
| **Meter Types** | Single electrical measurement | **Electricity only** (filtered from 4 types) |
| **Data Quality** | Utility-grade measurements | Real data with measurement error |
| **Transformation** | log1p(P) | log1p(meter_reading) |

### 7. **Model Input Dimensions**
| Dataset | Input Shape (samples, timesteps, features) |
|---------|---------------------------------------------|
| **Portuguese** | (~82,944, 23, 6) |
| **ASHRAE** | (~70,000, 23, 32) |

### 8. **Normalization Consistency**
✅ **BOTH NOW USE**: MinMaxScaler (0.0 to 1.0 range)
- **Portuguese**: Always used MinMaxScaler
- **ASHRAE**: Fixed from Z-score to MinMaxScaler for consistency

### 9. **Time-based Features**
| Feature | Portuguese Dataset | ASHRAE Dataset |
|---------|-------------------|----------------|
| **Minute-level** | ✅ Yes (0-59) | ❌ No |
| **Hour** | ✅ Yes (0-23) | ✅ Yes (0-23) |
| **Day of Week** | ✅ Yes (0-6) | ✅ Yes (0-6) |
| **Holidays** | ❌ No | ✅ Yes (US holidays 2016-2019) |

### 10. **Domain-Specific Considerations**
- **Portuguese Dataset**: Focused on electrical load forecasting for single entity
- **ASHRAE Dataset**: General building energy prediction across multiple:
  - Building types (Educational, Office, Retail, etc.)
  - Climate zones (16 different sites)
  - Energy end-uses (electricity, cooling, heating, steam)

## Implications for Model Performance

### Advantages of Portuguese Dataset
- Simpler feature space may reduce overfitting
- Cleaner, utility-grade measurements
- Consistent single-building context

### Advantages of ASHRAE Dataset  
- Rich feature set enables better generalization
- Diverse building types and climate conditions
- Multiple energy end-uses for comprehensive modeling
- Real-world data variability improves robustness

## Recent Update: Electricity-Only Focus

**✅ CHANGE IMPLEMENTED**: ASHRAE dataset now focuses on **electricity meters only** (meter = 0) for direct comparison with the Portuguese dataset.

### Impact of This Change
- **Feature Reduction**: Eliminates meter_category one-hot encoding (4 fewer features)
- **Target Consistency**: Both datasets now predict electrical energy consumption
- **Fair Comparison**: Removes energy type as a confounding variable
- **Reduced Complexity**: Simpler model input while retaining building/weather richness

### Updated Feature Count
- **Portuguese**: 6 features total
- **ASHRAE**: ~28-30 features (electricity only, still ~5x more complex)

## Conclusion

While both datasets follow the same core methodology (MinMax scaling, log transformation, sliding windows), the **ASHRAE dataset remains significantly more complex** even with electricity-only filtering:
- **~5x more features** after engineering (but more focused)
- **Multi-domain context** (building + weather + temporal)
- **Diverse building types** vs single entity context
- **Real-world data challenges** (missing values, measurement error)

The preprocessing pipeline has been **standardized** with **electricity-only focus** to use identical normalization and target variables, ensuring fair comparison between the datasets while preserving ASHRAE's rich contextual features.
