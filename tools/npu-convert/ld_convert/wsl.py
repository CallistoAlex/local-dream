"""Run ld-convert inside WSL from a Windows host."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from ld_convert.platform import HostOS, QualcommRuntime, detect_platform, host_to_linux_path, in_wsl


def _tool_root() -> Path:
    """Directory containing pyproject.toml for ld-npu-convert."""
    return Path(__file__).resolve().parents[1]


def _translate_argv_for_wsl(argv: list[str]) -> list[str]:
    out: list[str] = []
    path_flags = {
        "--cache-dir",
        "--qnn-sdk-root",
        "--litert-sdk-root",
        "--model-path",
        "--output-dir",
        "--config",
        "--qnn-clip-zip",
        "--clip-zip",
    }
    i = 0
    while i < len(argv):
        arg = argv[i]
        out.append(arg)
        if arg in path_flags and i + 1 < len(argv):
            i += 1
            out.append(host_to_linux_path(Path(argv[i])))
        elif arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
            if key in path_flags:
                out[-1] = f"{key}={host_to_linux_path(Path(val))}"
        i += 1
    return out


def _wsl_bash_cmd(inner: str) -> list[str]:
    return ["wsl", "-e", "bash", "-lc", inner]


def reexec_in_wsl_if_needed(argv: list[str] | None = None) -> None:
    """Re-run the current ld-convert invocation inside WSL (Windows only)."""
    if os.environ.get("LD_CONVERT_IN_WSL") == "1" or in_wsl():
        return

    info = detect_platform()
    if info.qualcomm_runtime != QualcommRuntime.WSL_RELAY:
        return

    argv = argv if argv is not None else sys.argv[1:]
    translated = _translate_argv_for_wsl(argv)
    tool_root = host_to_linux_path(_tool_root())
    args = " ".join(shlex.quote(a) for a in translated)

    inner = (
        "export LD_CONVERT_IN_WSL=1; "
        f"cd {shlex.quote(tool_root)} && "
        f"if command -v uv >/dev/null 2>&1; then "
        f"  uv run ld-convert {args}; "
        f"elif command -v ld-convert >/dev/null 2>&1; then "
        f"  ld-convert {args}; "
        f"else "
        f"  echo 'Install uv in WSL: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2; "
        f"  exit 1; "
        f"fi"
    )

    print("Windows detected — continuing conversion inside WSL…")
    print(f"+ wsl bash -lc '…'")
    raise SystemExit(subprocess.call(_wsl_bash_cmd(inner)))


def ensure_qualcomm_linux_host(*, no_wsl: bool = False) -> None:
    """Ensure we are on a host that can run QNN Linux tools."""
    info = detect_platform()

    if info.can_run_qnn_tools:
        return

    if info.os == HostOS.WINDOWS and info.has_wsl and not no_wsl:
        reexec_in_wsl_if_needed()
        return

    if info.os == HostOS.MACOS:
        raise RuntimeError(
            "Qualcomm QNN conversion requires Linux x86_64 binaries.\n\n"
            "On macOS you can split the pipeline:\n"
            "  1. ld-convert convert qualcomm sd15 ... --stage prepare\n"
            "     (export ONNX + calibration on macOS)\n"
            "  2. Copy ~/.cache/local-dream-npu-convert to a Linux machine or VM\n"
            "  3. ld-convert convert qualcomm sd15 ... --stage qnn\n"
            "  4. ld-convert convert qualcomm sd15 ... --stage package\n\n"
            "Alternatives: UTM/Parallels Linux VM, remote Linux box, or cloud VM."
        )

    raise RuntimeError(
        "Qualcomm QNN conversion requires Linux or WSL2.\n\n"
        "On Windows:\n"
        "  wsl --install\n"
        "  Restart, then re-run — ld-convert will auto-continue inside WSL.\n\n"
        "Pass --no-wsl only if you are already running on a Linux machine "
        "via SSH and want to skip WSL detection."
    )
