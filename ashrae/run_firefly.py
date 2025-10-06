#!/usr/bin/env python3
"""Run LSTM hyperparameter search using standard Firefly Algorithm."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Import the main search function
from ashrae.call_lstm_search_ashrae import run_lstm_search_ashrae

if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING FIREFLY ALGORITHM (FA) HYPERPARAMETER SEARCH")
    print("=" * 80)
    
    results = run_lstm_search_ashrae(
        algorithms=["FireflyAlgorithm"],
        output_suffix="_FA"
    )
    
    print("\n" + "=" * 80)
    print("FIREFLY ALGORITHM SEARCH COMPLETE")
    print("=" * 80)
    print(f"Best parameters: {results['best_params']}")
    print(f"Best score: {results['best_score']:.6f}")
