# ASHRAE Dataset Validation Report

## Summary
This report documents the validation and correction of ASHRAE dataset preprocessing scripts to ensure consistency with the original Portuguese dataset methodology described in the paper.

## Issues Found and Fixed

### ❌ Critical Issue: Normalization Inconsistency
- **Original Dataset**: Uses `MinMaxScaler` (range 0.0 to 1.0) 
- **ASHRAE Dataset**: Was using Z-score normalization (standardization)
- **Impact**: Violates paper's methodology, would affect model performance comparability

### ✅ Fix Applied
Updated `ashrae/preprocessing_ashrae.py` to use `MinMaxScaler` with same configuration as original dataset:
- Import: `from sklearn.preprocessing import MinMaxScaler`
- Configuration: `feature_range=MINMAX_FEATURE_RANGE` from `config.py`
- Consistent training/test split methodology

## Code Organization Improvements

### ✅ New Structure
Created dedicated `ashrae/` folder for ASHRAE-specific scripts:
```
ashrae/
├── __init__.py
├── preprocessing_ashrae.py     # Fixed normalization, updated imports
├── train_ashrae_lstm_comb.py   # Updated import paths
└── test_ashrae_preprocessing.py # Updated import paths
```

### ✅ Import Updates
- `ashrae/train_ashrae_lstm_comb.py`: Updated import from `tools.preprocessing_ashrae` → `ashrae.preprocessing_ashrae`
- `ashrae/test_ashrae_preprocessing.py`: Updated import from `tools.preprocessing_ashrae` → `ashrae.preprocessing_ashrae`  
- `ashrae/preprocessing_ashrae.py`: Updated import from `.preprocess_data2` → `tools.preprocess_data2`

## Validation of Preprocessing Pipeline

### ✅ Consistent Methodology Verified
1. **Target Variable**: Both datasets predict **electricity consumption** (Portuguese: Active Power P, ASHRAE: meter_reading electricity)
2. **Feature Engineering**: Both datasets use proper time-based features (hour, weekday, holidays)
3. **Normalization**: Both now use MinMaxScaler with same range (0.0 to 1.0)
4. **Target Transformation**: Both use `log1p` for target variable (`log(1+y)`)
5. **Missing Value Handling**: Both use mean imputation
6. **Sliding Window**: Both use same `sliding_windows2d_lstm` function with **23-step input, 1-step ahead forecasting**
7. **Train/Val/Test Split**: Both use same sequential splitting methodology

### ✅ ASHRAE-Specific Features Preserved
- Building metadata integration
- Weather data merging
- **Electricity as target**, **all meter types as features** (chilledwater, steam, hotwater)
- US holidays for 2016-2019
- Log transformation for square feet
- Removal of correlated features (dew_temperature, sea_level_pressure, etc.)
- One-hot encoding for building primary_use categories
- **Pivot operation** to separate electricity target from other meter features

## Updated Commands

From project root:
```bash
# Train ASHRAE dataset (using LSTM_comb.py framework)
python ashrae/train_ashrae_lstm_comb.py

# Train ASHRAE dataset (standalone implementation)
python ashrae/train_ashrae_lstm.py

# Test ASHRAE preprocessing pipeline
python ashrae/test_ashrae_preprocessing.py

# Debug RMSLE calculation for ASHRAE
python ashrae/debug_rmsle.py
```

## Final File Structure

```
ashrae/
├── __init__.py                           # Package initialization
├── preprocessing_ashrae.py              # Fixed preprocessing with MinMaxScaler
├── train_ashrae_lstm_comb.py            # Training using LSTM_comb.py framework
├── train_ashrae_lstm.py                 # Standalone training implementation
├── test_ashrae_preprocessing.py         # Preprocessing validation tests
└── debug_rmsle.py                       # RMSLE calculation debugging
```

## Compliance with Paper Requirements

### ✅ Technical Requirements Met
- **ModLSTM Hyperparameters**: Units=72, N_Layer=1, Sequence=23, L_Rate=0.010
- **Input Features**: P, Q, V equivalents + Time features (properly scaled)
- **Target Variable**: `log(1+y)` transformation with `exp(y)-1` inverse
- **Evaluation Metrics**: RMSE, MAE, MAPE, R², RMSLE

### ✅ Reproducibility Features
- Consistent normalization across datasets
- Proper sequential train/val/test splits
- Fixed random state capabilities
- Early stopping configuration

## Status: ✅ VALIDATED AND CORRECTED

The ASHRAE preprocessing pipeline now correctly follows the same methodology as the original Portuguese dataset, ensuring fair comparison and compliance with the paper's approach.
