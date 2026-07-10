"""Quick credential / market smoke test."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bot.main import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--check"] + sys.argv[1:]
    raise SystemExit(main())
