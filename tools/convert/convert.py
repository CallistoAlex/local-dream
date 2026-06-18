#!/usr/bin/env python3
"""
Local Dream — model conversion hub (Qualcomm QNN + MediaTek LiteRT).

Run from repo root or this directory:

  python tools/convert/convert.py
  python convert.py

See README.md in this folder.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "convert.py"

if __name__ == "__main__":
    if not LAUNCHER.is_file():
        print(f"Error: root launcher missing: {LAUNCHER}", file=sys.stderr)
        raise SystemExit(1)
    sys.argv[0] = str(LAUNCHER)
    runpy.run_path(str(LAUNCHER), run_name="__main__")
