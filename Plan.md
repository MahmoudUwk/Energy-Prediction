## Plan: Addressing Reviewer Feedback with ASHRAE Dataset

### 1. Objective and Context

The current code base produced results on the single-household Portuguese dataset (from `dataset/1Hz` to `dataset/resampled data`). A **major review** highlighted the **Limited Dataset Diversity** as a critical weakness.

The primary objective is to enhance the generalizability and robustness of the proposed **ModLSTM** model by validating its performance on a new, diverse dataset, **ASHRAE Great Energy Predictor III** (located in `dataset/ASHRAE/ashrae-energy-prediction`). This introduces the complexity of **diverse, non-residential building energy consumption patterns** and multiple climates, directly addressing the reviewers' core concern and strengthening the paper's claims regarding Net-Zero Energy Buildings (NZEBs).

### 2. Concise Implementation Plan

The plan focuses on quickly obtaining and presenting validation results on the new dataset, mirroring the existing results structure without complex new analysis methods.

| Step | Action | Objective |
| :--- | :--- | :--- |
| **2.1. Data Preprocessing** | **Adapt** the current preprocessing/feature engineering scripts (by referencing `dataset/ASHRAE/eagleusu-checkpoint2.ipynb` for necessary steps like merging, one-hot encoding, and feature creation) to process the ASHRAE data. Preserve the existing scripts for the Portuguese data. | Create a clean, feature-rich, and scaled input matrix from the ASHRAE files, ready for sequencing. |
| **2.2. Sequencing** | **Utilize the existing sliding window (sequencing) code** from the current codebase. Apply it to the preprocessed ASHRAE data using the sequence length from **Table II (Sequence=23)**. | Generate the 3D numpy arrays ($samples, timesteps, features$) required by the ModLSTM and benchmark LSTMs. |
| **2.3. Model Re-training** | **Utilize the existing model definition code** for: 1) **LSTM (Untuned)** and 2) **ModLSTM (Optimized)** using the best MFF hyperparameters from **Table II** (`Units=72`, `L_Rate=0.010`, `Sequence=23`). **Train** both models on the ASHRAE training split. | Prepare the models to predict ASHRAE energy consumption. |
| **2.4. Benchmark & ModLSTM Testing** | **Utilize the existing testing/prediction code.** Run predictions on the ASHRAE test data using the retrained ModLSTM and LSTM benchmark. For non-DL benchmarks, train and test simple **SVR** and **RFR** on the *flattened* ASHRAE data. | Generate prediction outputs for comparison. |
| **2.5. Comparative Results** | **Utilize the existing metrics calculation code** (RMSE, MAE, MAPE, R²) to evaluate ModLSTM and the selected benchmarks on the ASHRAE test set. | Produce a new results table (like **Table VI**) to prove generalizability. |
| **2.6. Visualization** | **Utilize the existing plotting code** to generate new comparative plots (e.g., a **bar chart** like Fig. 6-9 comparing metrics, or a **time-series plot** like Fig. 10 showing prediction vs. actual) for the ASHRAE data. | Provide visual evidence of generalizability for the reviewers. |