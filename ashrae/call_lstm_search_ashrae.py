from __future__ import annotations

from pathlib import Path
import sys
import os

# Immediate import-time banner to confirm execution start
print("[CALLER] ashrae.call_lstm_search_ashrae loaded", flush=True)


def main():
    # Force unbuffered stdout to ensure progress prints appear immediately
    os.environ["PYTHONUNBUFFERED"] = "1"
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    print("[CALLER] Starting LSTM hyperparameter search (ashrae.call_lstm_search_ashrae)", flush=True)

    # Ensure project root on sys.path so config/tools imports work
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import and run existing LSTM hyperparameter search script main
    from models.LSTM_hyperpara_search import main as run

    print("[CALLER] Launching models.LSTM_hyperpara_search.main()", flush=True)
    run()


if __name__ == "__main__":
    main()


