#!/usr/bin/env python3
"""Run all ASHRAE algorithms sequentially (SVR, RFR, SAMFOR, FA, ModFF)."""

from __future__ import annotations

from pathlib import Path

from .call_svr_ashrae import main as run_svr
from .call_rfr_ashrae import main as run_rfr
from .call_samfor_ashrae import main as run_samfor
from .simple_lstm_ashrae import main as run_simple_lstm
from .call_lstm_search_ashrae import run_lstm_search_single


def main() -> None:
    print("=" * 80)
    print("RUNNING ALL ASHRAE ALGORITHMS")
    print("=" * 80)
    
    
    #print("\n→ Running Simple LSTM...")
    #run_simple_lstm()
    # Baselines
    #print("\n→ Running SVR...")
    #run_svr()

    #print("\n→ Running RFR...")
    #run_rfr()

    print("\n→ Running SAMFOR...")
    run_samfor()



    # LSTM hyperparameter searches
    print("\n→ Running FireflyAlgorithm search...")
    run_lstm_search_single("FireflyAlgorithm", output_suffix="_FA")

    print("\n→ Running Mod_FireflyAlgorithm search...")
    run_lstm_search_single("Mod_FireflyAlgorithm", output_suffix="_ModFF")

    print("\nAll ASHRAE runs completed.")


if __name__ == "__main__":
    main()
