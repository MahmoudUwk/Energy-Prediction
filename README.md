## Energy Prediction Toolkit

### Overview
This project bundles the cleaned set of scripts required to train energy-prediction models (SAMFOR, SVR/RFR, LSTM variants) and to aggregate their serialized outputs into comparative visualizations and tables.

### Repository Structure
- **`preprocess_data2.py`** — Shared data utilities: dataset loaders (`get_SAMFOR_data`), scaling helpers, metrics, logging, and `save_object` / `loadDatasetObj` primitives.
- **`SAMFOR_trial1.py`** — Trains the SAMFOR baseline (or SVR augment) and writes `SAMFOR.obj` alongside plots/CSV metrics.
- **`SVR_energy_data_paper2.py`** — Fits Random Forest and SVR baselines, emitting `RFR.obj` and `SVR.obj` plus logged metrics.
- **`LSTM_comb.py`** — Implements the standalone LSTM workflow that saves `LSTM.obj`.
- **`LSTM_hyperpara_search.py`** — Runs Firefly-based hyperparameter search, saving `FireflyAlgorithm.obj`, `Mod_FireflyAlgorithm.obj`, and matching `Best_param*.obj` convergence bundles.
- **`plot_results.py`** — Main "plug" script that consumes the `.obj` files above to generate scatter plots, bar charts, convergence plots, and LaTeX tables.
- **`sklearn.svm.LinearSVR`** — Linear Support Vector Regression used inside SAMFOR pipelines.
- **`niapy/`** — Vendored copy of the NiaPy optimization library required for Firefly algorithms.

### Setup
1. Create a Python environment (3.8+ recommended).
2. Install dependencies:
   ```bash
   pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn niapy
   ```
   If using the vendored `niapy/`, add the repository root to `PYTHONPATH` or install locally via `pip install -e .` from the `niapy/` directory.
3. Adjust dataset/result paths in `preprocess_data2.py` to point to your local data directories (several Windows-style paths are hard-coded).

### Usage
1. **Generate artifacts**
   - Run `SAMFOR_trial1.py`, `SVR_energy_data_paper2.py`, `LSTM_comb.py`, and `LSTM_hyperpara_search.py` as needed. Each script persists its `.obj` artifact to the directory returned by `get_SAMFOR_data(...)` and logs metrics via the provided helpers.
2. **Aggregate results**
   - Execute `plot_results.py`. It loads the saved `.obj` files and produces comparative plots plus LaTeX-formatted tables in the configured results folder.

### Notes
- Training scripts assume long training schedules and large batch sizes; tune hyperparameters if hardware resources are limited.
- Generated outputs (`*.obj`, images, CSV logs) are meant to live outside version control—consider adding them to `.gitignore`.
- When extending the toolkit, keep new scripts aligned with the shared helpers in `preprocess_data2.py` to guarantee compatibility with `plot_results.py`.
