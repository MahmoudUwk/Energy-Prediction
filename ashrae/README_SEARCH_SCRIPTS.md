# LSTM Hyperparameter Search Scripts

## Overview
The LSTM hyperparameter search has been refactored into a reusable function with separate caller scripts for each algorithm.

## Files

### Core Function
- **`call_lstm_search_ashrae.py`** - Contains `run_lstm_search_ashrae()` function that performs the search

### Caller Scripts
1. **`run_firefly.py`** - Runs standard Firefly Algorithm (FA)
2. **`run_mod_firefly.py`** - Runs Modified Firefly Algorithm (ModFF)

## Usage

### Run Modified Firefly Algorithm (Recommended)
```bash
cd /home/student/mahmoud_project/Energy-Prediction
nohup python3 -u ashrae/run_mod_firefly.py > ashrae/logs/modff_search.log 2>&1 &
tail -f ashrae/logs/modff_search.log
```

### Run Standard Firefly Algorithm
```bash
cd /home/student/mahmoud_project/Energy-Prediction
nohup python3 -u ashrae/run_firefly.py > ashrae/logs/fa_search.log 2>&1 &
tail -f ashrae/logs/fa_search.log
```

### Run Both Algorithms (Default)
```bash
cd /home/student/mahmoud_project/Energy-Prediction
nohup python3 -u ashrae/call_lstm_search_ashrae.py > ashrae/logs/both_search.log 2>&1 &
tail -f ashrae/logs/both_search.log
```

## Output Files

Each algorithm produces **separate** output files to avoid conflicts:

### Modified Firefly Algorithm (ModFF)
- Progress CSV: `results/ashrae/search_progress_ModFireflyAlgorithm_ModFF.csv`
- Iteration results: `results/ashrae/lstm_search/search_iterations_Mod_FireflyAlgorithm.csv`
- Convergence plots: `results/ashrae/lstm_search/Conv_FF_*.png`

### Standard Firefly Algorithm (FA)
- Progress CSV: `results/ashrae/search_progress_FireflyAlgorithm_FA.csv`
- Iteration results: `results/ashrae/lstm_search/search_iterations_FireflyAlgorithm.csv`
- Convergence plots: `results/ashrae/lstm_search/Conv_FF_*.png`

### Both Algorithms
- Progress CSV: `results/ashrae/search_progress_ModFireflyAlgorithmFireflyAlgorithm.csv`
- Iteration results: Both algorithm CSVs in `results/ashrae/lstm_search/`

## CSV Columns

### Progress CSV
- `timestamp`, `phase`, `datatype`, `algorithm`, `units`, `layers`, `seq`, `learning_rate`, `save_path`, `artifact`

### Iteration Results CSV
- `timestamp`, `eval_num`, `units`, `num_layers`, `seq`, `learning_rate`
- `MSE`, `RMSE`, `MAE`, `MAPE`, `R2`

## Function Signature

```python
def run_lstm_search_ashrae(
    algorithms: List[str],
    output_suffix: str = ""
) -> dict:
    """
    Run LSTM hyperparameter search on ASHRAE dataset.
    
    Args:
        algorithms: List of algorithm names, e.g., ["Mod_FireflyAlgorithm"]
        output_suffix: Suffix for output files, e.g., "_ModFF"
    
    Returns:
        dict with keys: best_params, best_score, search_results, scaler
    """
```

## Running in Parallel

You can run both algorithms **simultaneously** on different terminals/sessions:

```bash
# Terminal 1: ModFF
nohup python3 -u ashrae/run_mod_firefly.py > ashrae/logs/modff.log 2>&1 &

# Terminal 2: FA  
nohup python3 -u ashrae/run_firefly.py > ashrae/logs/fa.log 2>&1 &
```

Each will use separate CSV files and won't conflict.

## Monitoring Progress

```bash
# Watch ModFF progress
tail -f ashrae/logs/modff.log

# Watch FA progress
tail -f ashrae/logs/fa.log

# Check iteration results in real-time
watch -n 5 'tail -20 results/ashrae/lstm_search/search_iterations_Mod_FireflyAlgorithm.csv'
```
