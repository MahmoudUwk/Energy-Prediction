#!/usr/bin/env python3
"""Run LSTM hyperparameter search using standard Firefly Algorithm."""

from __future__ import annotations

from pathlib import Path

# Import the main search function (relative within package)
from .call_lstm_search_ashrae import run_lstm_search_single


def main() -> None:
    print("=" * 80)
    print("RUNNING FIREFLY ALGORITHM (FA) HYPERPARAMETER SEARCH")
    print("=" * 80)

    results = run_lstm_search_single("FireflyAlgorithm", output_suffix="_FA")

    print("\n" + "=" * 80)
    print("FIREFLY ALGORITHM SEARCH COMPLETE")
    print("=" * 80)
    print(f"Best parameters: {results['best_params']}")
    print(f"Best score: {results['best_score']:.6f}")


if __name__ == "__main__":
    main()
