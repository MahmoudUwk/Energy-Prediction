## Project Guidelines & Technical Summary

This document provides concise guidelines and essential technical details for the AI Agent working on the **ModLSTM** Energy Prediction Toolkit.

### 1. File Organization & Commands

| Category | Command/Location | Detail |
| :--- | :--- | :--- |
| **Portuguese Training** | `python models/LSTM_comb.py` | Trains the final LSTM model on the original dataset. |
| **Optimization** | `python models/LSTM_hyperpara_search.py` | Executes the **Modified Firefly (MFF)** hyperparameter search. |
| **Baselines** | `python models/SVR_energy_data_paper2.py` | Runs SVR/RFR baselines. |
| **New Dataset** | `python ashrae/train_ashrae_lstm_comb.py` | Trains ModLSTM on the new **ASHRAE** dataset. |
| **Results** | `python plot_results.py` | Generates all comparative plots (e.g., Fig. 6-10). |
| **Preprocessing** | Utilities in `tools/` | Keep existing scripts separate; **ASHRAE-specific** scripts in `ashrae/` folder. |
| **Model/Data Files** | Models in `models/`; Results in `results/`. | Models are saved with `.obj` extension. |

### 2. Code Style & Standards

| Standard | Detail |
| :--- | :--- |
| **Imports** | Grouped; use `from __future__ import annotations`; use absolute imports. |
| **Formatting** | PEP 8 (snake\_case/PascalCase); line length max 100 chars; use `pathlib.Path` for file ops. |
| **Configuration** | All hyperparameters must be stored in `config.py` for centralized control. |
| **Reproducibility** | Set `random_state` (scikit-learn) and use early stopping (Keras/TensorFlow). |

### 3. Key Technical Requirements from Paper

| Parameter | Value | Context |
| :--- | :--- | :--- |
| **Model Type** | **ModLSTM (LSTM + MFF)** | Hybrid model for non-linear time-series forecasting. |
| **Optimization** | **MFF (Modified Firefly)** | Optimizes LSTM hyperparameters. |
| **Core Technique** | **Sliding Window** | Used to convert time-series data to the 3D tensor format ($samples, timesteps, features$) for LSTM input. **This logic exists in the codebase and must be reused.** |
| **ModLSTM Hyperparameters** | **Units=72**, **N\_Layer=1**, **Sequence=23**, **L\_Rate=0.010** | Use these fixed values for the ModLSTM configuration. (Source: Table II) |
| **Input Features** | Must include (at minimum): **Active Power (P), Reactive Power (Q), Voltage (V)**, and **Time-based Features** (e.g., Hour, Weekday, Is\_Holiday). | Must be properly scaled (Z-score normalized). |
| **Target Variable** | $\log(1+y)$ (log-transformed meter reading). | The final prediction must be returned to the original scale using $\exp(y)-1$. |
| **Evaluation Metrics** | **RMSE, MAE, MAPE, $R^2$** | These four metrics must be calculated for all comparative results. |