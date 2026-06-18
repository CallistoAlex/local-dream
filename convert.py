#!/usr/bin/env python3
"""
Local Dream — All-in-one NPU model conversion.

Run from the repository root (no need to cd into tools/npu-convert):

  python convert.py                 # interactive wizard
  python convert.py wizard
  python convert.py setup qualcomm
  python convert.py convert qualcomm sd15 --model-path model.safetensors --model-name MyModel

Supports Qualcomm (QNN) and MediaTek (LiteRT).
Guide: https://ld-guide.chino.icu/conversion/
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOL_DIR = ROOT / "tools" / "npu-convert"


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _ensure_tool_dir() -> None:
    if not (TOOL_DIR / "pyproject.toml").is_file():
        _eprint(f"Error: conversion tool not found at {TOOL_DIR}")
        _eprint("Make sure you run this script from the Local Dream repo root.")
        sys.exit(1)


def _uv_available() -> bool:
    return shutil.which("uv") is not None


def _sync_with_uv() -> bool:
    if not _uv_available():
        return False
    print("→ Syncing conversion tool dependencies (uv)…", flush=True)
    result = subprocess.run(
        ["uv", "sync"],
        cwd=str(TOOL_DIR),
        env=os.environ.copy(),
    )
    return result.returncode == 0


def _run_via_uv(argv: list[str]) -> int:
    cmd = ["uv", "run", "ld-convert", *argv]
    return subprocess.call(cmd, cwd=str(TOOL_DIR), env=os.environ.copy())


def _run_inprocess(argv: list[str]) -> int:
    if str(TOOL_DIR) not in sys.path:
        sys.path.insert(0, str(TOOL_DIR))
    try:
        from ld_convert.cli import main as cli_main
    except ImportError:
        _eprint("Error: ld_convert package not installed.")
        _eprint("Install uv (https://docs.astral.sh/uv/) and re-run, or:")
        _eprint(f"  cd {TOOL_DIR} && pip install -e .")
        return 1
    return cli_main(argv)


def main() -> int:
    _ensure_tool_dir()
    argv = sys.argv[1:]

    if _uv_available():
        if not (TOOL_DIR / ".venv").is_dir():
            if _sync_with_uv() is False:
                return 1
        return _run_via_uv(argv)

    _eprint("Note: uv not found — running with current Python.")
    _eprint("Install uv for an isolated environment: https://docs.astral.sh/uv/")
    return _run_inprocess(argv)


if __name__ == "__main__":
    raise SystemExit(main())
