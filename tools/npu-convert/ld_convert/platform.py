"""Cross-platform detection and path helpers."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


class HostOS(str, Enum):
    LINUX = "linux"
    MACOS = "macos"
    WINDOWS = "windows"


class QualcommRuntime(str, Enum):
    """Where Qualcomm QNN ELF tools can run."""

    NATIVE_LINUX = "native_linux"
    WSL_RELAY = "wsl_relay"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PlatformInfo:
    os: HostOS
    in_wsl: bool
    has_wsl: bool
    machine: str

    @property
    def display_name(self) -> str:
        if self.in_wsl:
            return "WSL (Linux)"
        return self.os.value

    @property
    def can_run_qnn_tools(self) -> bool:
        """QNN SDK conversion binaries are Linux x86_64 ELF."""
        return self.os == HostOS.LINUX or self.in_wsl

    @property
    def qualcomm_runtime(self) -> QualcommRuntime:
        if self.can_run_qnn_tools:
            return QualcommRuntime.NATIVE_LINUX
        if self.os == HostOS.WINDOWS and self.has_wsl:
            return QualcommRuntime.WSL_RELAY
        return QualcommRuntime.UNAVAILABLE

    @property
    def supports_native_mtk_python(self) -> bool:
        return True


def in_wsl() -> bool:
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/version", encoding="utf-8", errors="ignore") as fh:
            v = fh.read().lower()
        return "microsoft" in v or "wsl" in v
    except OSError:
        return False


def wsl_available() -> bool:
    if platform.system() != "Windows":
        return False
    wsl = shutil.which("wsl")
    if not wsl:
        return False
    try:
        import subprocess

        result = subprocess.run(
            [wsl, "--status"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def detect_platform() -> PlatformInfo:
    system = platform.system()
    if system == "Linux":
        host = HostOS.LINUX
    elif system == "Darwin":
        host = HostOS.MACOS
    elif system == "Windows":
        host = HostOS.WINDOWS
    else:
        host = HostOS.LINUX

    return PlatformInfo(
        os=host,
        in_wsl=in_wsl(),
        has_wsl=wsl_available(),
        machine=platform.machine().lower(),
    )


def venv_python(venv_dir: Path) -> Path:
    """Return python executable inside a uv/pip venv (Windows + Unix)."""
    if sys.platform == "win32":
        candidates = [
            venv_dir / "Scripts" / "python.exe",
            venv_dir / "Scripts" / "python",
        ]
    else:
        candidates = [venv_dir / "bin" / "python3", venv_dir / "bin" / "python"]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(f"venv python not found under {venv_dir}")


def find_bash() -> str:
    for name in ("bash", "bash.exe"):
        path = shutil.which(name)
        if path:
            return path
    info = detect_platform()
    if info.os == HostOS.WINDOWS:
        raise RuntimeError(
            "bash not found. Install WSL2 (recommended: wsl --install) "
            "or Git for Windows (https://git-scm.com/download/win)."
        )
    raise RuntimeError("bash not found.")


def install_hint(cmd: str) -> str:
    info = detect_platform()
    hints: dict[str, dict[HostOS, str]] = {
        "zstd": {
            HostOS.LINUX: "sudo apt-get install zstd",
            HostOS.MACOS: "brew install zstd",
            HostOS.WINDOWS: "winget install Meta.Zstandard",
        },
        "zip": {
            HostOS.LINUX: "sudo apt-get install zip",
            HostOS.MACOS: "brew install zip if needed",
            HostOS.WINDOWS: "not required — ld-convert uses Python zipfile",
        },
        "uv": {
            HostOS.LINUX: "curl -LsSf https://astral.sh/uv/install.sh | sh",
            HostOS.MACOS: "brew install uv",
            HostOS.WINDOWS: "winget install astral-sh.uv",
        },
    }
    by_os = hints.get(cmd, {})
    return by_os.get(info.os, f"Install {cmd} for your platform")


def host_to_linux_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if platform.system() == "Windows" and not in_wsl():
        s = str(resolved)
        if len(s) >= 2 and s[1] == ":":
            drive = s[0].lower()
            rest = s[2:].replace("\\", "/")
            return f"/mnt/{drive}{rest}"
        return s.replace("\\", "/")
    return str(resolved)


def default_cache_dir() -> Path:
    if platform.system() == "Windows" and not in_wsl():
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "local-dream-npu-convert"
    return Path.home() / ".cache" / "local-dream-npu-convert"


Stage = Literal["all", "prepare", "qnn", "package"]
