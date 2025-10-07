from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

from tools.preprocess_data2 import loadDatasetObj

RESULTS_ROOT = Path("results/ashrae")
FIGURES_DIR = RESULTS_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Paper-ready styling parameters
fs = 17  # Font size
barWidth = 0.3
linewidth = 1.5
separation = 30
colors = sns.color_palette("magma", 6*separation).as_hex()[::separation]
fromatt = 'eps'  # 'eps' or 'png'

MODEL_DIRS = {
    "svr": "SVR",
    "samfor": "SAMFOR", 
    "rfr": "RFR",
    "lstm": "LSTM",
    "lstm_search": "LSTM-Search",
}

# Algorithm renaming for paper
alg_rename = {
    'LSTM-FF': 'LSTM FF',
    'LSTM-ModFF': 'LSTM Modified FF (Proposed)',
    'LSTM': 'LSTM without HP tuning',
    'RFR': 'RFR',
    'SAMFOR': 'SAMFOR',
    'SVR': 'SVR'
}

alg_rename_short = {
    'LSTM-FF': 'LSTM FF',
    'LSTM-ModFF': 'Proposed',
    'LSTM': 'LSTM',
    'RFR': 'RFR',
    'SAMFOR': 'SAMFOR',
    'SVR': 'SVR'
}

Metric = ['RMSE (kWh)', 'MAE (kWh)', "MAPE (%)", "R square score"]
Metric_name = ['RMSE', 'MAE', "MAPE", "r2_score"]


def read_metrics(model_key: str) -> pd.DataFrame:
    model_dir = RESULTS_ROOT / model_key
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    df["Model"] = MODEL_DIRS.get(model_key, model_key.upper())
    # Normalize column names to expected set
    col_map = {"MSE": "MSE", "RMSE": "RMSE", "MAE": "MAE", "MAPE": "MAPE", "R2": "R2", "RMSLE": "RMSLE"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    # Coerce numeric if present
    for c in ["RMSE", "MAE", "R2", "MAPE", "RMSLE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def latest_artifact(model_key: str) -> Path | None:
    art_dir = RESULTS_ROOT / model_key / "artifacts"
    if not art_dir.exists():
        return None
    objs = sorted(art_dir.glob("*.obj"), key=lambda p: p.stat().st_mtime, reverse=True)
    return objs[0] if objs else None


def read_lstm_search_best_rows() -> pd.DataFrame:
    """Read best artifacts from lstm_search and convert to metrics rows.

    Uses original-scale validation metrics from the best_evaluation payload.
    """
    art_dir = RESULTS_ROOT / "lstm_search" / "artifacts"
    if not art_dir.exists():
        return pd.DataFrame()
    rows: List[Dict[str, float | str]] = []
    for obj_path in sorted(art_dir.glob("best_*.obj")):
        try:
            data = loadDatasetObj(str(obj_path))
            algorithm = data.get("algorithm") if isinstance(data, dict) else None
            timestamp = data.get("timestamp") if isinstance(data, dict) else None
            content = data.get("payload", data) if isinstance(data, dict) else {}
            best_eval = content.get("best_evaluation", {})
            vm = best_eval.get("val_metrics", {})
            # Prefer original-scale metrics
            rmse = vm.get("RMSE_orig", vm.get("VAL_RMSE_orig"))
            mae = vm.get("MAE_orig", vm.get("VAL_MAE_orig"))
            mape = vm.get("MAPE_orig", vm.get("VAL_MAPE_orig"))
            r2 = vm.get("R2_orig", vm.get("VAL_R2_orig"))
            # Label mapping
            label = "LSTM-Search"
            name = obj_path.name
            alg_marker = algorithm or name
            if "Mod_FireflyAlgorithm" in alg_marker:
                label = "LSTM-ModFF"
            elif "FireflyAlgorithm" in alg_marker:
                label = "LSTM-FF"
            # Only include if any metric found
            if any(v is not None for v in [rmse, mae, mape, r2]):
                row = {
                    "Model": label,
                    "RMSE": float(rmse) if rmse is not None else float("nan"),
                    "MAE": float(mae) if mae is not None else float("nan"),
                    "MAPE": float(mape) if mape is not None else float("nan"),
                    "R2": float(r2) if r2 is not None else float("nan"),
                }
                if timestamp is not None:
                    row["timestamp"] = timestamp
                rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def read_complexity_metrics(model_key: str) -> pd.DataFrame:
    """Read complexity metrics (training time, testing time, etc.) from model artifacts."""
    model_dir = RESULTS_ROOT / model_key
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(metrics_path)
    df["Model"] = MODEL_DIRS.get(model_key, model_key.upper())
    
    # Extract complexity-related columns
    complexity_cols = ["Model"]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["time", "epoch", "iteration", "eval", "param"]):
            complexity_cols.append(col)
    
    # Add common complexity metrics if available
    if "train_time_min" in df.columns:
        complexity_cols.append("train_time_min")
    if "test_time_s" in df.columns:
        complexity_cols.append("test_time_s")
    if "epochs_trained" in df.columns:
        complexity_cols.append("epochs_trained")
    if "timestamp" in df.columns:
        complexity_cols.append("timestamp")
    
    available_cols = [col for col in complexity_cols if col in df.columns]
    return df[available_cols] if available_cols else pd.DataFrame()


def read_lstm_search_complexity() -> pd.DataFrame:
    """Read complexity metrics from LSTM search results."""
    lstm_search_dir = RESULTS_ROOT / "lstm_search"
    if not lstm_search_dir.exists():
        return pd.DataFrame()
    
    rows: List[Dict[str, float | str]] = []
    
    # Read from search iteration CSVs
    csv_files = list(lstm_search_dir.glob("search_iterations_*.csv"))
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
                
            # Extract algorithm name from filename
            filename = csv_file.stem
            if "Mod_FireflyAlgorithm" in filename:
                label = "LSTM-ModFF"
            elif "FireflyAlgorithm" in filename:
                label = "LSTM-FF"
            else:
                label = filename.replace("search_iterations_", "").replace("_FA", "").replace("_ModFF", "")
            
            # Calculate complexity metrics
            total_evals = len(df)
            avg_epochs = df["epochs_trained"].mean() if "epochs_trained" in df.columns else 0
            total_epochs = df["epochs_trained"].sum() if "epochs_trained" in df.columns else 0
            
            # Estimate training time (assuming ~2-3 minutes per evaluation based on terminal output)
            estimated_train_time_min = total_evals * 2.5  # Conservative estimate
            
            row = {
                "Model": label,
                "total_evaluations": total_evals,
                "avg_epochs_per_eval": round(avg_epochs, 1),
                "total_epochs": int(total_epochs),
                "estimated_train_time_min": round(estimated_train_time_min, 1),
                "hyperparameter_search": "Yes",
                "optimization_algorithm": "Firefly" if "FireflyAlgorithm" in filename else "ModFirefly"
            }
            rows.append(row)
            
        except Exception as e:
            print(f"Error reading complexity data from {csv_file.name}: {e}")
            continue
    
    return pd.DataFrame(rows)


def build_complexity_table() -> pd.DataFrame:
    """Build comprehensive complexity analysis table."""
    tables = []
    
    # Read complexity metrics from regular models
    for key in MODEL_DIRS:
        if key != "lstm_search":  # Handle LSTM search separately
            complexity_df = read_complexity_metrics(key)
            if not complexity_df.empty:
                tables.append(complexity_df)
    
    # Add LSTM search complexity metrics
    lstm_complexity = read_lstm_search_complexity()
    if not lstm_complexity.empty:
        tables.append(lstm_complexity)
    
    if not tables:
        return pd.DataFrame()
    
    df = pd.concat(tables, ignore_index=True)
    
    # Keep latest entry per model if duplicates
    try:
        if "timestamp" in df.columns:
            df = (
                df.sort_values("timestamp")
                .groupby("Model", as_index=False)
                .tail(1)
            )
        else:
            df = df.groupby("Model", as_index=False).tail(1)
    except Exception:
        df = df.groupby("Model", as_index=False).tail(1)
    
    # Standardize column names and add missing columns with defaults
    complexity_columns = {
        "Model": "Model",
        "train_time_min": "Training Time (min)",
        "test_time_s": "Testing Time (s)",
        "epochs_trained": "Epochs Trained",
        "total_evaluations": "Total Evaluations",
        "avg_epochs_per_eval": "Avg Epochs/Eval",
        "total_epochs": "Total Epochs",
        "hyperparameter_search": "HP Search",
        "optimization_algorithm": "Optimization"
    }
    
    # Rename existing columns
    df = df.rename(columns={k: v for k, v in complexity_columns.items() if k in df.columns})
    
    # Add default values for missing columns
    defaults = {
        "Training Time (min)": 0.0,
        "Testing Time (s)": 0.0,
        "Epochs Trained": 0,
        "Total Evaluations": 1,
        "Avg Epochs/Eval": 0.0,
        "Total Epochs": 0,
        "HP Search": "No",
        "Optimization": "None"
    }
    
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
    
    # Select and order columns
    final_columns = ["Model", "Training Time (min)", "Testing Time (s)", "Epochs Trained", 
                    "Total Evaluations", "Avg Epochs/Eval", "Total Epochs", "HP Search", "Optimization"]
    available_columns = [col for col in final_columns if col in df.columns]
    
    return df[available_columns]


def load_predictions(model_key: str) -> Tuple[np.ndarray, np.ndarray] | None:
    obj_path = latest_artifact(model_key)
    if obj_path is None:
        # Special case for LSTM search: pull the appropriate labeled artifact
        if model_key == "lstm_search":
            # Prefer ModFF then FF, or any
            art_dir = RESULTS_ROOT / "lstm_search" / "artifacts"
            candidates = [
                art_dir / "LSTM-ModFF.obj",
                art_dir / "LSTM-FF.obj",
            ]
            for cand in candidates:
                if cand.exists():
                    obj_path = cand
                    break
            if obj_path is None or not obj_path.exists():
                return None
        else:
            return None
    data = loadDatasetObj(str(obj_path))
    # Support unified envelope {algorithm, model_name, timestamp, payload}
    content = data.get("payload", data) if isinstance(data, dict) else {}
    y_true = np.asarray(content.get("y_test", []))
    y_pred = np.asarray(content.get("y_test_pred", []))
    if y_true.size == 0 or y_pred.size == 0:
        return None
    return y_true, y_pred


def build_metrics_table() -> pd.DataFrame:
    tables = [read_metrics(k) for k in MODEL_DIRS]
    tables = [t for t in tables if not t.empty]
    # Add LSTM search best metrics from artifacts
    lstm_best = read_lstm_search_best_rows()
    if not lstm_best.empty:
        tables.append(lstm_best)
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)
    # Keep last row per model (latest run) if duplicates; fallback to last row
    try:
        if "timestamp" in df.columns:
            df = (
                df.sort_values("timestamp")
                .groupby("Model", as_index=False)
                .tail(1)
            )
        else:
            df = df.groupby("Model", as_index=False).tail(1)
    except Exception:
        df = df.groupby("Model", as_index=False).tail(1)
    # Restrict to paper columns only (exclude RMSLE, MSE, timestamp)
    allowed = ["Model", "RMSE", "MAE", "R2", "MAPE"]
    cols = [c for c in allowed if c in df.columns]
    return df[cols]


def plot_bar(results_data, barWidth, linewidth, full_file_path, ind_plot, Metric, Metric_name):
    """Plot bar chart in paper style matching plot_results.py"""
    fig = plt.figure(figsize=(14, 8), dpi=150)
    
    data_res = []
    indeces = []
    indeces_short = []
    
    for counter, (model_name, y_true, y_pred) in enumerate(results_data):
        # Calculate metrics
        RMSE_i = np.sqrt(np.mean((y_true - y_pred) ** 2))
        MAE_i = np.mean(np.abs(y_true - y_pred))
        MAPE_i = 100.0 * np.mean(np.abs((y_true - y_pred) / y_true))
        r2_score_i = r2_score(y_true, y_pred)
        
        row = [RMSE_i, MAE_i, MAPE_i, r2_score_i]
        data_res.append(row)
        
        # Get proper labels
        full_label = alg_rename.get(model_name, model_name)
        short_label = alg_rename_short.get(model_name, model_name)
        indeces.append(full_label)
        indeces_short.append(short_label)
        
        # Plot bar
        plt.bar(counter, row[ind_plot], color=colors[counter], linewidth=linewidth, 
                width=barWidth, hatch="xxxx", edgecolor=colors[counter], 
                label=full_label, fill=False)
    
    # Styling
    plt.xlabel('Algorithm', fontweight='bold', fontsize=fs)
    plt.ylabel(Metric[ind_plot], fontweight='bold', fontsize=fs)
    plt.xticks([r for r in range(len(indeces_short))], indeces_short)
    _, b = plt.ylim()
    plt.ylim([0, b*1.1])
    plt.title('Bar plot of the ' + Metric[ind_plot] + ' for different algorithms', fontsize=fs)
    plt.legend(prop={'size': fs}, loc='best')
    plt.gca().grid(True)
    
    return data_res

def plot_metrics_bar(df: pd.DataFrame) -> None:
    """Create bar plots for all metrics in paper style"""
    metrics = [c for c in ["RMSE", "MAE", "R2", "MAPE"] if c in df.columns]
    if not metrics:
        print("No standard metrics to plot.")
        return
    
    # Collect prediction data for bar plotting
    results_data = []
    for _, row in df.iterrows():
        model_name = row["Model"]
        # Load predictions for this model
        predictions = load_predictions_by_name(model_name)
        if predictions is not None:
            y_true, y_pred = predictions
            results_data.append((model_name, y_true, y_pred))
    
    if not results_data:
        print("No prediction data found for bar plots.")
        return
    
    # Create bar plots for each metric
    for i, metric in enumerate(metrics):
        data_res = plot_bar(results_data, barWidth, linewidth, None, i, Metric, Metric_name)
        plt.savefig(FIGURES_DIR / f'bar_plot_{Metric_name[i]}.{fromatt}', bbox_inches='tight', format=fromatt)
        plt.show()


def load_predictions_by_name(model_name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """Load predictions by model name (reverse lookup)"""
    # Find the key for this model name
    for key, label in MODEL_DIRS.items():
        if label == model_name:
            return load_predictions(key)
    return None

def plot_scatter_per_model() -> None:
    """Create scatter plots in paper style matching plot_results.py"""
    entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for key, label in MODEL_DIRS.items():
        loaded = load_predictions(key)
        if loaded is None:
            continue
        y_true, y_pred = loaded
        entries.append((label, y_true, y_pred))
    
    if not entries:
        print("No predictions found for scatter plots.")
        return
    
    # Paper-style scatter plot layout
    fig, axs = plt.subplots(int(len(entries)/2), 2, figsize=(15, 8), dpi=150, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.5, wspace=0.1)
    axs = axs.ravel()
    
    for i, (label, y_true, y_pred) in enumerate(entries):
        # Get proper label
        full_label = alg_rename.get(label, label)
        
        axs[i].scatter(y_true, y_pred, alpha=0.5, s=15, color=colors[i], marker='o', linewidth=1.5)
        axs[i].set_title(full_label, fontsize=14, x=0.27, y=0.62)
        
        max_val_i = max(max(y_true), max(y_pred))
        X_line = np.arange(0, max_val_i, max_val_i/200)
        axs[i].plot(X_line, X_line)
        
        if i in [0, 2, 4]:
            axs[i].set_ylabel('Predicted values (kWh)', fontsize=fs)
        axs[i].set_xlabel('Actual values (kWh)', fontsize=fs)
    
    fig.suptitle('Scatter plot of predictions vs real values for different algorithms for energy consumption (kWh)', 
                 fontsize=fs, x=0.5, y=0.95)
    plt.xticks(rotation=25)
    plt.savefig(FIGURES_DIR / f'scatter_plot.{fromatt}', bbox_inches='tight', format=fromatt)


def plot_timeseries_snippets(window: int = 2000) -> None:
    """Create time series plots in paper style matching plot_results.py"""
    entries: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for key, label in MODEL_DIRS.items():
        loaded = load_predictions(key)
        if loaded is None:
            continue
        y_true, y_pred = loaded
        entries.append((label, y_true[:window], y_pred[:window]))
    
    if not entries:
        print("No predictions found for time series plots.")
        return
    
    # Paper-style time series layout
    fig, axs = plt.subplots(len(entries), 1, figsize=(20, 11), dpi=150, facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    axs = np.array(axs).reshape(-1)
    
    for i, (label, y_true, y_pred) in enumerate(entries):
        # Get proper label
        full_label = alg_rename.get(label, label)
        
        axs[i].plot(y_true, color='red', linewidth=1.5, alpha=0.6)
        axs[i].plot(y_pred, color='blue', linewidth=0.8)
        axs[i].set_title(full_label, fontsize=14, x=0.5, y=0.6)
        axs[i].set_ylabel('Energy consumption (kWh)', fontsize=8)
    
    axs[-1].set_xlabel('Timestamp (hours)', fontsize=fs)
    plt.legend(['Actual', 'Predicted'])
    
    f_title = 'Energy Prediction for ' + str(int(len(y_true)/24)) + ' days ahead'
    fig.suptitle(f_title, fontsize=fs, x=0.5, y=0.92)
    plt.savefig(FIGURES_DIR / f'PredictionsVsReal.{fromatt}', bbox_inches='tight', format=fromatt)


def plot_convergence() -> None:
    """Plot convergence curves in paper style matching plot_results.py"""
    lstm_search_dir = RESULTS_ROOT / "lstm_search"
    if not lstm_search_dir.exists():
        print("No LSTM search results found for convergence plots.")
        return
    
    # Find search iteration CSV files
    csv_files = list(lstm_search_dir.glob("search_iterations_*.csv"))
    if not csv_files:
        print("No search iteration CSV files found.")
        return
    
    # Paper-style convergence plot
    fig = plt.figure(figsize=(14, 8), dpi=150)
    markers = ['o-', 'o-']
    fillstyles = ['top', 'bottom']
    
    alg_rename_itr = {
        'Mod_FireflyAlgorithm': 'Modified Fire Fly',
        'FireflyAlgorithm': 'Fire Fly'
    }
    
    for counter, csv_file in enumerate(sorted(csv_files)):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
                
            # Extract algorithm name from filename
            filename = csv_file.stem
            if "Mod_FireflyAlgorithm" in filename:
                alg_key = "Mod_FireflyAlgorithm"
            elif "FireflyAlgorithm" in filename:
                alg_key = "FireflyAlgorithm"
            else:
                continue
            
            label = alg_rename_itr.get(alg_key, alg_key)
            
            # Plot validation RMSE over evaluations
            if "VAL_RMSE_orig" in df.columns and "eval_num" in df.columns:
                plt.step(df["eval_num"], df["VAL_RMSE_orig"], markers[counter % len(markers)],
                        markersize=10, dash_joinstyle='bevel', fillstyle=fillstyles[counter % len(fillstyles)],
                        color=colors[counter + 4], label=label, linewidth=3)
            else:
                print(f"Warning: Missing columns in {csv_file.name}")
                
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    plt.xlabel('Iteration', fontsize=fs)
    plt.ylabel('RMSE (kWh)', fontsize=fs)
    plt.legend(prop={'size': fs}, loc='best')
    plt.title('Convergence Graph for the best firefly solution in each iteration vs the iteration number', fontsize=fs)
    plt.gca().grid(True)
    plt.savefig(FIGURES_DIR / f'Conv_eval_comparison.{fromatt}', bbox_inches='tight', format=fromatt)


def write_txt(txt, fname):
    """Write text to file matching plot_results.py style"""
    f = open(fname, "w")
    f.write(txt)
    f.close()

def export_metrics_table(df: pd.DataFrame) -> None:
    """Export metrics table in paper style matching plot_results.py"""
    if df.empty:
        return
    
    # Convert to paper format
    df_paper = df.copy()
    df_paper = df_paper.round(4)
    
    # Rename columns to match paper style
    column_mapping = {
        'RMSE': 'RMSE (kWh)',
        'MAE': 'MAE (kWh)', 
        'MAPE': 'MAPE (%)',
        'R2': 'R square score'
    }
    df_paper = df_paper.rename(columns=column_mapping)
    
    # Rename model names
    df_paper['Model'] = df_paper['Model'].map(alg_rename).fillna(df_paper['Model'])
    
    print(df_paper)
    
    # Save CSV
    df_paper.to_csv(FIGURES_DIR / "results_table.csv", index=False)
    
    # Save LaTeX
    try:
        latex_txt = df_paper.style.to_latex()
        write_txt(latex_txt, FIGURES_DIR / "results_table_latex.txt")
    except Exception:
        pass


def export_complexity_table(df: pd.DataFrame) -> None:
    """Export complexity analysis table in paper style"""
    if df.empty:
        return
    
    # Convert to paper format
    df_paper = df.copy()
    df_paper = df_paper.round(2)
    
    # Rename model names
    df_paper['Model'] = df_paper['Model'].map(alg_rename).fillna(df_paper['Model'])
    
    print("\nComplexity Analysis:")
    print(df_paper)
    
    # Save CSV
    df_paper.to_csv(FIGURES_DIR / "complexity_analysis.csv", index=False)
    
    # Save LaTeX
    try:
        latex_txt = df_paper.style.to_latex()
        write_txt(latex_txt, FIGURES_DIR / "complexity_analysis_latex.txt")
    except Exception:
        pass


def plot_complexity_comparison(df: pd.DataFrame) -> None:
    """Create complexity comparison plots."""
    if df.empty:
        print("No complexity data found for plotting.")
        return
    
    # Plot training time comparison
    if "Training Time (min)" in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
        
        # Training time bar plot
        sns.barplot(data=df, x="Model", y="Training Time (min)", hue="Model", ax=ax1, palette="viridis")
        ax1.set_title("Training Time Comparison")
        ax1.set_xlabel("")
        ax1.set_ylabel("Training Time (minutes)")
        ax1.grid(True, axis="y", alpha=0.2)
        ax1.get_legend().remove()
        for label in ax1.get_xticklabels():
            label.set_rotation(45)
        
        # Testing time bar plot
        if "Testing Time (s)" in df.columns:
            sns.barplot(data=df, x="Model", y="Testing Time (s)", hue="Model", ax=ax2, palette="plasma")
            ax2.set_title("Testing Time Comparison")
            ax2.set_xlabel("")
            ax2.set_ylabel("Testing Time (seconds)")
            ax2.grid(True, axis="y", alpha=0.2)
            ax2.get_legend().remove()
            for label in ax2.get_xticklabels():
                label.set_rotation(45)
        else:
            ax2.axis("off")
        
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "complexity_comparison.png", bbox_inches="tight")
        print(f"Complexity comparison plot saved to {FIGURES_DIR / 'complexity_comparison.png'}")
    
    # Plot computational efficiency (if we have both performance and complexity data)
    try:
        # Load performance metrics for efficiency analysis
        perf_df = build_metrics_table()
        if not perf_df.empty and not df.empty:
            # Merge performance and complexity data
            merged_df = perf_df.merge(df, on="Model", how="inner")
            
            if "RMSE" in merged_df.columns and "Training Time (min)" in merged_df.columns:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=150)
                
                # Scatter plot: RMSE vs Training Time
                scatter = ax.scatter(merged_df["Training Time (min)"], merged_df["RMSE"], 
                                   s=100, alpha=0.7, c=range(len(merged_df)), cmap="tab10")
                
                # Add model labels
                for i, model in enumerate(merged_df["Model"]):
                    ax.annotate(model, (merged_df["Training Time (min)"].iloc[i], merged_df["RMSE"].iloc[i]),
                              xytext=(5, 5), textcoords="offset points", fontsize=8)
                
                ax.set_xlabel("Training Time (minutes)")
                ax.set_ylabel("RMSE (kWh)")
                ax.set_title("Performance vs Computational Complexity")
                ax.grid(True, alpha=0.3)
                
                fig.tight_layout()
                fig.savefig(FIGURES_DIR / "performance_vs_complexity.png", bbox_inches="tight")
                print(f"Performance vs complexity plot saved to {FIGURES_DIR / 'performance_vs_complexity.png'}")
                
    except Exception as e:
        print(f"Could not create performance vs complexity plot: {e}")


def main() -> None:
    # Performance metrics
    metrics_df = build_metrics_table()
    if not metrics_df.empty:
        print("Consolidated metrics:\n", metrics_df)
        export_metrics_table(metrics_df)
        plot_metrics_bar(metrics_df)
    else:
        print("No metrics found under", RESULTS_ROOT)
    
    # Complexity analysis
    complexity_df = build_complexity_table()
    if not complexity_df.empty:
        print("\nComplexity analysis:\n", complexity_df)
        export_complexity_table(complexity_df)
        plot_complexity_comparison(complexity_df)
    else:
        print("No complexity data found under", RESULTS_ROOT)
    
    # Visualization plots
    plot_scatter_per_model()
    plot_timeseries_snippets(window=2000)
    plot_convergence()


if __name__ == "__main__":
    main()



