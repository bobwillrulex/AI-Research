from __future__ import annotations

import sys
from pathlib import Path


# Allow `py main.py` from the repository root without requiring package install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_trading_ai.ui import main as ui_main


if __name__ == "__main__":
    ui_main()
