# Energy Prediction Toolkit

A comprehensive machine learning toolkit for building energy consumption prediction, featuring advanced LSTM models with Firefly Algorithm optimization and support for multiple datasets.

## 🎯 Overview

This project implements and evaluates various energy prediction models including:
- **ModLSTM**: Modified LSTM with Firefly Algorithm optimization
- **Standard LSTM**: Baseline LSTM implementation
- **SAMFOR**: Sequential Attention-based Model for Forecasting
- **SVR/RFR**: Support Vector Regression and Random Forest baselines

The toolkit supports two datasets:
- **Portuguese Dataset**: Single-household energy consumption data
- **ASHRAE Dataset**: Multi-building commercial energy consumption data

## 📁 Repository Structure

```
Energy-Prediction/
├── models/                     # Model implementations
│   ├── LSTM_comb.py           # Main LSTM training pipeline
│   ├── LSTM_hyperpara_search.py # Firefly Algorithm hyperparameter optimization
│   ├── SAMFOR_trial1.py        # SAMFOR model implementation
│   └── SVR_energy_data_paper2.py # SVR/RFR baseline models
├── tools/                      # Data preprocessing utilities
│   ├── preprocess_data2.py    # Shared utilities for Portuguese dataset
│   ├── preprocessing_ashrae.py # ASHRAE-specific preprocessing
│   └── resample_dataset.py    # Data resampling utilities
├── dataset/                    # Datasets
│   ├── 1Hz/                   # Original Portuguese dataset (1Hz)
│   ├── resampled data/        # Resampled Portuguese data
│   └── ASHRAE/                # ASHRAE Great Energy Predictor III dataset
├── results/                    # Model outputs and results
│   ├── 1s/                    # Portuguese dataset results
│   └── ashrae/                # ASHRAE dataset results
├── niapy/                      # Firefly Algorithm implementation
└── config.py                  # Configuration settings
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- scikit-learn
- pandas, numpy
- matplotlib, seaborn

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Energy-Prediction
```

2. Create a virtual environment:
```bash
conda create -n FF python=3.8
conda activate FF
```

3. Install dependencies:
```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn
```

### Usage

#### Portuguese Dataset Training

1. **Train LSTM model**:
```bash
python models/LSTM_comb.py
```

2. **Run hyperparameter optimization**:
```bash
python models/LSTM_hyperpara_search.py
```

3. **Train baseline models**:
```bash
python models/SVR_energy_data_paper2.py
python models/SAMFOR_trial1.py
```

4. **Generate comparative results**:
```bash
python plot_results.py
```

#### ASHRAE Dataset

- **SVR baseline (disjoint buildings)**
```bash
conda run -n FF python -m ashrae.call_svr_ashrae
```

- **SAMFOR baseline**
```bash
conda run -n FF python -m ashrae.call_samfor_ashrae
```

- **LSTM Hyperparameter Search (overnight)**
```bash
conda run -n FF python -m ashrae.call_lstm_search_ashrae
```

## 📊 Datasets

### Portuguese Dataset
- **Source**: Single-household energy consumption data
- **Frequency**: 1Hz (resampled to 1s)
- **Features**: Power consumption measurements
- **Preprocessing**: MinMax scaling, sliding window sequencing
- **Sequence Length**: 23 timesteps

### ASHRAE Dataset
- **Source**: ASHRAE Great Energy Predictor III (Kaggle)
- **Scope**: Multi-building commercial energy consumption
- **Features**: Building metadata, weather data, temporal features
- **Preprocessing**: MinMax scaling (fit on train only), one-hot encoding; `building_id` preserved and not scaled for windowing
- **Sequence Length**: 23 timesteps
- **Sample Size**: ~250,000 total rows using disjoint building splits (Train≈105k, Val≈53k, Test≈105k)
 - **Resampling**: None for ASHRAE 1s/1Hz; dataset already at target granularity

## 🔧 Key Features

### Advanced LSTM Models
- **ModLSTM**: Modified LSTM architecture with optimized hyperparameters
- **Firefly Algorithm**: Bio-inspired optimization for hyperparameter tuning
- **Early Stopping**: Prevents overfitting during training
- **Sequence-to-One**: Predicts next timestep energy consumption

### Comprehensive Preprocessing
- **Feature Engineering**: Temporal, building, and weather features
- **Missing Value Imputation**: Mean imputation for numerical features
- **Normalization**: Dataset-specific scaling (MinMax vs Z-score)
- **Sequential Cropping**: Time-series consistent data sampling

### Evaluation Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **RMSLE**: Root Mean Squared Logarithmic Error
- **R²**: Coefficient of Determination

## 🔬 Research Context

### Problem Statement
Building energy consumption prediction is crucial for:
- **Smart Grid Management**: Optimizing energy distribution and demand response
- **Building Optimization**: Improving HVAC system efficiency and occupant comfort
- **Sustainability Goals**: Supporting Net-Zero Energy Building (NZEB) initiatives
- **Cost Reduction**: Minimizing energy costs through predictive maintenance

### Challenges in Energy Prediction
1. **Temporal Dependencies**: Energy consumption exhibits complex time-series patterns
2. **Multi-scale Variability**: Patterns vary across hours, days, seasons, and years
3. **External Factors**: Weather, occupancy, and building characteristics significantly impact consumption
4. **Data Quality**: Missing values, outliers, and measurement errors in sensor data
5. **Generalizability**: Models trained on single buildings may not generalize to diverse building types

### Model Architecture Rationale
- **LSTM Networks**: Capture long-term dependencies in time-series data
- **ModLSTM**: Enhanced LSTM with optimized architecture for energy prediction
- **Firefly Algorithm**: Bio-inspired optimization for hyperparameter tuning
- **Attention Mechanisms**: Focus on relevant temporal patterns (SAMFOR)
- **Ensemble Methods**: Combine multiple models for robust predictions

## 🛠️ Configuration

Key hyperparameters (from Table II):
- **LSTM Units**: 72
- **Learning Rate**: 0.010
- **Sequence Length**: 23
- **Batch Size**: 64
- **Epochs**: 50 (with early stopping)

## 📝 Implementation Details

### Data Preprocessing Pipeline
1. **Data Loading**: Load raw datasets
2. **Merging**: Join building metadata and weather data
3. **Feature Engineering**: Create temporal and categorical features
4. **Normalization**: Apply dataset-specific scaling
5. **Sequencing**: Generate sliding window sequences for LSTM
6. **Splitting**: Train/validation/test splits

### Model Training Pipeline
1. **Model Building**: Construct LSTM architecture
2. **Compilation**: Configure optimizer and loss function
3. **Training**: Fit model with early stopping
4. **Evaluation**: Compute metrics on test set
5. **Persistence**: Save model and results

### Dataset Diversity Challenge
This toolkit addresses a critical limitation in energy prediction research: **Limited Dataset Diversity**. Most studies focus on single buildings or homogeneous datasets, limiting model generalizability.

**Our Approach**:
- **Multi-Dataset Validation**: Portuguese residential + ASHRAE commercial data
- **Diverse Building Types**: Single-family homes vs. multi-building commercial complexes
- **Different Climates**: Portuguese climate vs. multiple US climate zones
- **Varied Meter Types**: Electricity vs. multiple utility types (electricity, chilled water, steam, hot water)
- **Scale Differences**: Single building vs. thousands of buildings

### Methodological Contributions
1. **Unified Preprocessing Pipeline**: Consistent feature engineering across datasets
2. **Adaptive Normalization**: Dataset-specific scaling strategies
3. **Memory-Efficient Processing**: Sequential cropping for large datasets
4. **Comprehensive Evaluation**: Multiple metrics on original-scale data
5. **Reproducible Research**: Complete preprocessing and training pipelines

## 📚 Background Literature

### Energy Prediction in Buildings
Building energy consumption prediction has evolved from simple regression models to sophisticated deep learning approaches. Key developments include:

- **Traditional Methods**: Linear regression, ARIMA, and support vector machines
- **Machine Learning**: Random forests, gradient boosting, and neural networks
- **Deep Learning**: LSTM, GRU, and transformer-based models
- **Hybrid Approaches**: Combining physical models with data-driven methods

### Time Series Forecasting Challenges
Energy consumption exhibits unique characteristics:
- **Seasonality**: Daily, weekly, and annual patterns
- **Non-stationarity**: Changing patterns over time
- **External Dependencies**: Weather, occupancy, and building operations
- **Multi-scale Dynamics**: Short-term fluctuations and long-term trends

### Optimization in Neural Networks
Hyperparameter optimization is crucial for model performance:
- **Grid Search**: Exhaustive but computationally expensive
- **Random Search**: More efficient than grid search
- **Bayesian Optimization**: Model-based optimization
- **Bio-inspired Algorithms**: Firefly, particle swarm, genetic algorithms

## 📚 References

- ASHRAE Great Energy Predictor III: [Kaggle Competition](https://www.kaggle.com/c/ashrae-energy-prediction)
- Firefly Algorithm: Yang, X.S. (2008). Nature-Inspired Metaheuristic Algorithms
- LSTM Networks: Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory
- Energy Prediction: Ahmad, T. et al. (2018). A review on renewable energy and electricity requirement forecasting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce `max_samples` in preprocessing
2. **Import Errors**: Ensure all dependencies are installed
3. **Path Issues**: Check dataset paths in configuration files
4. **CUDA Issues**: Verify TensorFlow GPU installation

### Support

For issues and questions:
- Check the troubleshooting section
- Review the preprocessing reports
- Examine the test scripts for examples

---

**Note**: This toolkit is designed for research purposes and addresses the challenge of limited dataset diversity in energy prediction research.