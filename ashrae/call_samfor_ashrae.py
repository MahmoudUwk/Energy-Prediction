from __future__ import annotations

from pathlib import Path
import sys


def main():
    # Ensure project root on sys.path so config/tools imports work
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Import and run existing SAMFOR script main
    from models.SAMFOR_trial1 import main as run

    run()


if __name__ == "__main__":
    main()


