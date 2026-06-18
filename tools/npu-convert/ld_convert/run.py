"""Subprocess helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from ld_convert.platform import find_bash, install_hint, venv_python


def which(cmd: str) -> str | None:
    return shutil.which(cmd)


def require_cmd(cmd: str, hint: str = "") -> str:
    path = which(cmd)
    if not path:
        msg = f"Required command not found: {cmd}"
        if not hint:
            hint = install_hint(cmd)
        if hint:
            msg += f"\n{hint}"
        raise RuntimeError(msg)
    return path


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    display = " ".join(cmd)
    if len(display) > 120:
        display = display[:117] + "..."
    print(f"+ {display}", flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def run_python(
    python: Path,
    script: Path,
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    run([str(python), str(script), *args], cwd=cwd, env=env)


def run_bash_script(
    script_rel: str,
    args: list[str],
    *,
    cwd: Path,
) -> None:
    """Run an official .sh script with bash (Linux / WSL / Git Bash)."""
    bash = find_bash()
    run([bash, script_rel, *args], cwd=cwd)


def uv_available() -> bool:
    return which("uv") is not None


def setup_uv_venv(
    bundle_dir: Path,
    python_version: str,
    *,
    recreate: bool = False,
) -> Path:
    """Create/sync uv venv inside an official conversion bundle."""
    require_cmd("uv", install_hint("uv"))
    venv_dir = bundle_dir / ".venv"
    if recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)
    if not venv_dir.exists():
        run(["uv", "venv", "-p", python_version], cwd=bundle_dir)
    run(["uv", "sync"], cwd=bundle_dir)
    return venv_python(venv_dir)


def zip_dir(source_dir: Path, zip_path: Path) -> None:
    """Create a zip archive with model files at zip root (Windows / macOS / Linux)."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(source_dir.rglob("*")):
            if file.is_file():
                zf.write(file, file.name)
    print(f"  packaged: {zip_path}")


def zstd_patch(base_file: Path, new_file: Path, patch_out: Path) -> None:
    """Create a zstd patch (requires zstd CLI on all platforms)."""
    zstd = which("zstd") or which("zstd.exe")
    if not zstd:
        raise RuntimeError(
            "zstd is required for extra-resolution patches.\n" + install_hint("zstd")
        )
    run(
        [
            zstd,
            "--patch-from",
            str(base_file),
            str(new_file),
            "-o",
            str(patch_out),
        ]
    )
