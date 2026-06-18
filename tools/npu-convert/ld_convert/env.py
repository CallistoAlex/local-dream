"""Environment verification and setup."""

from __future__ import annotations

import shutil
from pathlib import Path

from ld_convert.config import DEFAULT_CACHE, DEFAULT_PYTHON
from ld_convert.download import (
    ensure_qnn_sdk,
    ensure_qualcomm_sd15_bundle,
    ensure_qualcomm_sdxl_bundle,
    patch_qnn_sdk_root,
)
from ld_convert.platform import detect_platform, install_hint, venv_python
from ld_convert.run import require_cmd, run, setup_uv_venv, uv_available


class EnvCheck:
    def __init__(self):
        self.ok: list[str] = []
        self.warn: list[str] = []
        self.fail: list[str] = []

    def add_ok(self, msg: str):
        self.ok.append(msg)

    def add_warn(self, msg: str):
        self.warn.append(msg)

    def add_fail(self, msg: str):
        self.fail.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.fail) == 0


def check_system() -> EnvCheck:
    r = EnvCheck()
    info = detect_platform()
    r.add_ok(f"Platform: {info.display_name} ({info.machine})")

    if info.qualcomm_runtime.value == "wsl_relay":
        r.add_ok("WSL2 available — Qualcomm conversion will run inside WSL")
    elif info.os.value == "macos":
        r.add_ok("macOS — Python/MediaTek steps run natively; Qualcomm QNN needs Linux VM or --stage split")
    elif info.os.value == "windows":
        r.add_warn("Windows without WSL — install WSL2 for Qualcomm conversion (wsl --install)")

    if uv_available():
        r.add_ok("uv: installed")
    else:
        r.add_fail(f"uv: not found — {install_hint('uv')}")

    zstd = shutil.which("zstd") or shutil.which("zstd.exe")
    if zstd:
        r.add_ok(f"zstd: {zstd}")
    else:
        r.add_warn(f"zstd: not found (needed for SD1.5 extra resolutions) — {install_hint('zstd')}")

    return r


def check_qualcomm(
    cache_dir: Path = DEFAULT_CACHE,
    qnn_sdk_root: Path | None = None,
    *,
    no_wsl: bool = False,
) -> EnvCheck:
    r = check_system()
    info = detect_platform()

    if not info.can_run_qnn_tools and info.os.value == "windows" and info.has_wsl:
        r.add_ok("Qualcomm QNN: will relay to WSL at runtime")
    elif not info.can_run_qnn_tools:
        r.add_warn("Qualcomm QNN tools: require Linux/WSL (use --stage prepare on macOS)")

    try:
        bundle = ensure_qualcomm_sd15_bundle(cache_dir)
        r.add_ok(f"SD1.5 bundle: {bundle}")
    except Exception as e:
        r.add_fail(f"SD1.5 bundle download failed: {e}")

    if info.can_run_qnn_tools or (info.os.value == "windows" and info.has_wsl and not no_wsl):
        try:
            sdk = ensure_qnn_sdk(cache_dir, sdk_root=qnn_sdk_root)
            envsetup = sdk / "bin" / "envsetup.sh"
            if envsetup.exists():
                r.add_ok(f"QNN SDK 2.28: {sdk}")
            else:
                r.add_fail(f"QNN SDK invalid (no envsetup.sh): {sdk}")
        except Exception as e:
            r.add_fail(f"QNN SDK: {e}")

    return r


def check_mediatek(litert_sdk_root: Path | None = None) -> EnvCheck:
    r = check_system()
    info = detect_platform()
    r.add_ok(f"MediaTek Python pipeline: supported on {info.display_name}")

    for cmd in ["onnx2tf"]:
        if shutil.which(cmd):
            r.add_ok(f"{cmd}: installed")
        else:
            r.add_warn(f"{cmd}: not found — run: ld-convert setup mediatek")

    if litert_sdk_root and litert_sdk_root.exists():
        r.add_ok(f"LiteRT SDK: {litert_sdk_root}")
    else:
        r.add_warn(
            "LiteRT SDK not configured — pass --litert-sdk-root "
            "(https://github.com/google-ai-edge/LiteRT/releases)"
        )

    compile_bin = shutil.which("litert_compile")
    if compile_bin:
        r.add_ok(f"litert_compile: {compile_bin}")
    else:
        r.add_warn("litert_compile: not on PATH (AOT compile — may be Linux-only)")

    return r


def setup_mediatek(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    python_version: str = DEFAULT_PYTHON,
    recreate: bool = False,
) -> Path:
    """Create uv venv with MediaTek conversion dependencies."""
    require_cmd("uv", install_hint("uv"))
    venv_dir = cache_dir / "venv-mtk"
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        run(["uv", "venv", "-p", python_version, str(venv_dir)])

    python = venv_python(venv_dir)
    req_file = Path(__file__).resolve().parents[1] / "requirements-mtk.txt"
    run(["uv", "pip", "install", "--python", str(python), "-r", str(req_file)])

    return python


def setup_qualcomm(
    *,
    cache_dir: Path = DEFAULT_CACHE,
    qnn_sdk_root: Path | None = None,
    python_version: str = DEFAULT_PYTHON,
    kind: str = "sd15",
) -> tuple[Path, Path, Path]:
    """Download bundles, QNN SDK, create venv. Returns (bundle_dir, qnn_sdk, python)."""
    require_cmd("uv", install_hint("uv"))
    if kind == "sdxl":
        bundle = ensure_qualcomm_sdxl_bundle(cache_dir)
    else:
        bundle = ensure_qualcomm_sd15_bundle(cache_dir)
    sdk = ensure_qnn_sdk(cache_dir, sdk_root=qnn_sdk_root)
    patch_qnn_sdk_root(bundle, sdk)
    python = setup_uv_venv(bundle, python_version)
    return bundle, sdk, python


def print_check(result: EnvCheck) -> None:
    for line in result.ok:
        print(f"  ✓ {line}")
    for line in result.warn:
        print(f"  ! {line}")
    for line in result.fail:
        print(f"  ✗ {line}")
