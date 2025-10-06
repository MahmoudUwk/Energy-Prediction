from __future__ import annotations

from pathlib import Path
import sys


def main():
    # Ensure project root on sys.path so config/tools imports work
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("=" * 80)
    print("SAMFOR ASHRAE DATASET TRAINING")
    print("=" * 80)

    # Import ASHRAE preprocessing
    from preprocessing_ashrae_disjoint import preprocess_ashrae_disjoint_splits, get_ashrae_lstm_data_disjoint
    from models.SAMFOR_trial1 import run_samfor
    from config import SAMFOR_SAMFOR_PARAMS

    # Load ASHRAE data
    print("Loading ASHRAE dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_ashrae_disjoint_splits(
        target_samples=250_000,
        train_fraction=0.4,
        val_fraction=0.2,
        test_fraction=0.4,
    )

    # Get LSTM sequences then flatten for SAMFOR (2D data)
    seq_length = SAMFOR_SAMFOR_PARAMS["sequence_length"]
    X_tr_lstm, y_tr_lstm, X_va_lstm, y_va_lstm, X_te_lstm, y_te_lstm = get_ashrae_lstm_data_disjoint(
        X_train, y_train, X_val, y_val, X_test, y_test, seq_length=seq_length
    )

    # SAMFOR expects 2D data (samples, features) not 3D (samples, timesteps, features)
    # Use last timestep features for SAMFOR
    X_train_flat = X_tr_lstm[:, -1, :]  # Last timestep features
    X_test_flat = X_te_lstm[:, -1, :]   # Last timestep features

    # SAMFOR needs original scale targets (not scaled)
    y_train_orig = scaler.inverse_transform(y_tr_lstm.reshape(-1, 1)).flatten()
    y_test_scaled = y_te_lstm  # Keep scaled for _evaluate_model

    print(f"Data shapes for SAMFOR:")
    print(f"  X_train: {X_train_flat.shape}")
    print(f"  X_test: {X_test_flat.shape}")
    print(f"  y_train: {y_train_orig.shape}")
    print(f"  y_test: {y_test_scaled.shape}")

    # Run SAMFOR
    results = run_samfor(
        X_train=X_train_flat,
        y_train=y_train_orig,
        X_test=X_test_flat,
        y_test=y_test_scaled,
        scaler=scaler,
        seq_length=seq_length,
        save_path_str="results/ashrae"
    )

    print("=" * 80)
    print("SAMFOR TRAINING COMPLETED")
    print("=" * 80)
    print(f"Algorithm: {results['algorithm']}")
    print(f"Metrics: {results['metrics']}")
    print(f"Train time: {results['train_time']:.2f} min")
    print(f"Test time: {results['test_time']:.4f} s")
    print(f"Results saved to: {results['saved_files']}")


if __name__ == "__main__":
    main()


