from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tools.preprocess_data2 import loadDatasetObj


RESULTS_ROOT = Path("ashrae/results/ashrae")
FIGURES_DIR = RESULTS_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    "svr": "SVR",
    "samfor": "SAMFOR",
    "rfr": "RFR",
    "lstm": "LSTM",
}


def read_metrics(model_key: str) -> pd.DataFrame:
    model_dir = RESULTS_ROOT / model_key
    metrics_path = model_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_path)
    df["Model"] = MODEL_DIRS.get(model_key, model_key.upper())
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


def load_predictions(model_key: str) -> Tuple[np.ndarray, np.ndarray] | None:
    obj_path = latest_artifact(model_key)
    if obj_path is None:
        return None
    payload = loadDatasetObj(str(obj_path))
    y_true = np.asarray(payload.get("y_test", []))
    y_pred = np.asarray(payload.get("y_test_pred", []))
    if y_true.size == 0 or y_pred.size == 0:
        return None
    return y_true, y_pred


def build_metrics_table() -> pd.DataFrame:
    tables = [read_metrics(k) for k in MODEL_DIRS]
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame()
    df = pd.concat(tables, ignore_index=True)
    # Keep last row per model (latest run) if duplicates
    if "timestamp" in df.columns:
        df = (
            df.sort_values("timestamp")
            .groupby("Model", as_index=False)
            .tail(1)
        )
    else:
        df = df.groupby("Model", as_index=False).tail(1)
    # Reorder common columns if present
    preferred = ["Model", "RMSE", "MAE", "R2", "MAPE", "RMSLE"]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def plot_metrics_bar(df: pd.DataFrame) -> None:
    metrics = [c for c in ["RMSE", "MAE", "R2", "MAPE"] if c in df.columns]
    if not metrics:
        print("No standard metrics to plot.")
        return
    num = len(metrics)
    fig, axs = plt.subplots(1, num, figsize=(5 * num, 4), dpi=150)
    if num == 1:
        axs = [axs]
    palette = sns.color_palette("magma", n_colors=len(df))
    for ax, metric in zip(axs, metrics):
        sns.barplot(data=df, x="Model", y=metric, ax=ax, palette=palette)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.2)
        for label in ax.get_xticklabels():
            label.set_rotation(20)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "metrics_bar.png", bbox_inches="tight")


def plot_scatter_per_model() -> None:
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
    cols = min(3, len(entries))
    rows = int(np.ceil(len(entries) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), dpi=150)
    axs = np.array(axs).reshape(-1)
    for ax, (label, y_true, y_pred) in zip(axs, entries):
        ax.scatter(y_true, y_pred, s=5, alpha=0.5)
        max_v = float(max(np.max(y_true), np.max(y_pred)))
        ax.plot([0, max_v], [0, max_v], color="gray", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Actual (kWh)")
        ax.set_ylabel("Predicted (kWh)")
        ax.grid(True, alpha=0.2)
    for ax in axs[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "scatter.png", bbox_inches="tight")


def plot_timeseries_snippets(window: int = 2000) -> None:
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
    fig, axs = plt.subplots(len(entries), 1, figsize=(14, 2 * len(entries)), dpi=150)
    axs = np.array(axs).reshape(-1)
    for ax, (label, y_true, y_pred) in zip(axs, entries):
        ax.plot(y_true, label="Actual", linewidth=1.2)
        ax.plot(y_pred, label="Predicted", linewidth=0.9)
        ax.set_title(label)
        ax.set_ylabel("kWh")
        ax.grid(True, alpha=0.2)
    axs[-1].set_xlabel("Time index")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "timeseries.png", bbox_inches="tight")


def export_metrics_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    df.to_csv(FIGURES_DIR / "metrics_consolidated.csv", index=False)
    try:
        with open(FIGURES_DIR / "metrics_consolidated.tex", "w") as f:
            f.write(df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))
    except Exception:
        pass


def main() -> None:
    metrics_df = build_metrics_table()
    if not metrics_df.empty:
        print("Consolidated metrics:\n", metrics_df)
        export_metrics_table(metrics_df)
        plot_metrics_bar(metrics_df)
    else:
        print("No metrics found under", RESULTS_ROOT)
    plot_scatter_per_model()
    plot_timeseries_snippets(window=2000)


if __name__ == "__main__":
    main()



