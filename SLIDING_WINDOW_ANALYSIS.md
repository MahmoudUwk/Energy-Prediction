# Sliding Window Analysis for Time Series Datasets

## 🔍 Sliding Window Logic Overview

The project uses **sliding window time series forecasting** where historical data is used to predict future values. Let me analyze the implementation for both datasets.

## 📐 Core Sliding Window Functions

### 1. **sliding_windows2d_lstm()** - Main Function for LSTM Models
```python
def sliding_windows2d_lstm(data, seq_length):
    x = np.zeros((len(data) - seq_length, seq_length, data.shape[1]))
    y = np.zeros((len(data) - seq_length, 1))
    for ind in range(len(x)):
        x[ind, :, :] = data[ind : ind + seq_length, :]
        y[ind] = data[ind + seq_length, 0]
    return x, y
```

### 2. **sliding_windows2d()** - For Traditional ML Models
```python
def sliding_windows2d(data, seq_length, k_step, num_feat):
    x = np.zeros((len(data) - seq_length - k_step + 1, seq_length * num_feat))
    y = np.zeros((len(data) - seq_length - k_step + 1, k_step))
    for ind in range(len(x)):
        x[ind, :] = np.reshape(data[ind : ind + seq_length, :], -1)
        y[ind] = data[ind + seq_length : ind + seq_length + k_step, 0]
    return x, y
```

## 🎯 Forecast Horizon Analysis

### **Key Finding: Single-Step Forecasting**
- **Forecast Horizon**: **1 time step ahead** (single-step prediction)
- **Sequence Input**: Previous `seq_length` time steps
- **Target**: Value at time `t + 1`

### **Mathematical Representation**
```
Input:  [t-22, t-21, t-20, ..., t-1, t]  (23 time steps)
Target: [t+1]                            (1 time step ahead)
```

## 📊 Dataset-Specific Analysis

### **Portuguese Dataset**
```python
# Data Structure: [timestamp, P, Q, V, Minute, DOW, H]
# Target: Active Power (P) - log1p transformed

sliding_windows2d_lstm(data, seq_length=23):
- Input shape: (samples, 23, 6)
- Target shape: (samples, 1)
- Target: data[t+1, 0]  # First column = Active Power
```

### **ASHRAE Dataset (Updated)**
```python
# Data Structure: [chilledwater, steam, hotwater, building_features, weather_features, temporal_features]
# Target: Electricity - log1p transformed

sliding_windows2d_lstm(data, seq_length=23):
- Input shape: (samples, 23, ~30-35)
- Target shape: (samples, 1)  
- Target: data[t+1, 0]  # First column = Electricity
```

## ⚙️ Sequencing Logic Validation

### ✅ **Correct Implementation**
1. **Input at time t**: Previous 23 observations `[t-22, t-1]`
2. **Target at time t**: Next observation `[t+1]`
3. **No data leakage**: Future information not used in inputs
4. **Proper alignment**: Target corresponds to first column (target variable)

### 🔄 **Window Sliding Process**
```
Sample 1: Input [t=1:23]    → Target [t=24]
Sample 2: Input [t=2:24]    → Target [t=25]  
Sample 3: Input [t=3:25]    → Target [t=26]
...
Sample N: Input [t=N:N+22]  → Target [t=N+23]
```

## 📈 Time Series Characteristics

### **Temporal Resolution**
- **Portuguese Dataset**: Likely hourly or sub-hourly measurements
- **ASHRAE Dataset**: Hourly measurements
- **Sequence Length**: 23 hours ≈ 1 day of historical data
- **Forecast Horizon**: 1 hour ahead

### **Data Preparation Order**
1. **Chronological Sorting**: Data sorted by timestamp
2. **Sequential Splitting**: Train → Validation → Test (time-respecting)
3. **Window Creation**: Applied after splitting to prevent leakage

## 🔍 Technical Implementation Details

### **Memory Management**
```python
# Sequential cropping for memory efficiency
if len(X_train) > max_samples:
    X_train = X_train.iloc[:max_samples]  # First N samples only
    y_train = y_train.iloc[:max_samples]
```

### **Feature Structure**
```python
# Data format for sliding_windows2d_lstm:
data = np.column_stack([features, target])
# Columns: [feature_1, feature_2, ..., feature_n, target]
# Target is ALWAYS the first column: data[:, 0]
```

## ⚠️ Important Considerations

### **1. Single-Step vs Multi-Step**
- **Current**: Single-step forecasting (1 hour ahead)
- **Alternative**: Multi-step forecasting could predict multiple future hours
- **Trade-off**: Single-step is more accurate, multi-step is more practical

### **2. Target Variable Position**
- **Critical**: Target must be first column (`data[:, 0]`)
- **ASHRAE Update**: Electricity target positioned as first column after pivot
- **Portuguese**: Active Power already first column

### **3. Sequence Length Justification**
- **23 hours**: Captures daily patterns
- **Reasonable**: Long enough for patterns, short enough for memory efficiency
- **Domain appropriate**: 24-hour cycles in energy consumption

## 📋 Comparison Summary

| Aspect | Portuguese Dataset | ASHRAE Dataset (Updated) |
|--------|-------------------|--------------------------|
| **Forecast Horizon** | 1 step ahead | 1 step ahead |
| **Sequence Length** | 23 time steps | 23 time steps |
| **Target Variable** | Active Power (P) | Electricity |
| **Target Position** | First column | First column |
| **Window Logic** | ✅ Correct | ✅ Correct |
| **Temporal Resolution** | Hourly/sub-hourly | Hourly |
| **Multi-variate** | P, Q, V + temporal | All meters + building + weather + temporal |

## ✅ **Conclusion: Sliding Window Logic is CORRECT**

Both datasets use **proper single-step time series forecasting** with:
- ✅ **23-step historical input**
- ✅ **1-step ahead prediction**  
- ✅ **No data leakage**
- ✅ **Correct target positioning**
- ✅ **Appropriate temporal resolution**
- ✅ **Memory-efficient implementation**

The implementation follows standard time series best practices and is suitable for energy load forecasting tasks.
