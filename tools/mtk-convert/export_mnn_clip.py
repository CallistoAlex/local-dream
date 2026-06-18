#!/usr/bin/env python3
"""Extract MNN CLIP + tokenizer for MediaTek NPU packages.

Sources (pick one):
  --clip-zip     Any Local Dream zip with CLIP files (QNN, CPU/MNN, or litert)
  --native       Run Local Dream native --convert (needs --native-binary + --cvtbase)
  --safetensors  Same as --native when binary + cvtbase are configured via env/flags
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CLIP_FILES_SD15 = ["tokenizer.json", "clip_v2.mnn", "pos_emb.bin", "token_emb.bin"]
CLIP_FILES_SDXL = [
    "tokenizer.json",
    "clip.mnn",
    "clip_2.mnn",
    "pos_emb.bin",
    "token_emb.bin",
    "pos_emb_2.bin",
    "token_emb_2.bin",
]


def _clip_files(sdxl: bool) -> list[str]:
    return CLIP_FILES_SDXL if sdxl else CLIP_FILES_SD15


def extract_from_zip(zip_path: Path, output_dir: Path, sdxl: bool) -> list[str]:
    """Extract CLIP artifacts from any model zip (flat entry names like the app)."""
    files = _clip_files(sdxl)
    output_dir.mkdir(parents=True, exist_ok=True)
    found: list[str] = []
    missing: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        names_in_zip = {n.split("/")[-1]: n for n in zf.namelist()}
        for name in files:
            entry = names_in_zip.get(name)
            if entry:
                zf.extract(entry, output_dir)
                # Flatten if zip had subdirectories
                extracted = output_dir / entry
                target = output_dir / name
                if extracted != target and extracted.exists():
                    extracted.rename(target)
                found.append(name)
                print(f"  extracted {name}")
            else:
                missing.append(name)

    # Clean nested dirs left from zip extract
    for child in list(output_dir.iterdir()):
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)

    if missing:
        print(f"  WARNING: missing in {zip_path.name}: {missing}", file=sys.stderr)
    if not found:
        raise FileNotFoundError(f"No CLIP files found in {zip_path}")
    return found


def export_via_native(
    safetensors: Path,
    output_dir: Path,
    *,
    native_binary: Path,
    cvtbase_dir: Path,
    clip_skip: int = 2,
) -> None:
    """Run stable_diffusion_core --convert to produce MNN CLIP files."""
    if not native_binary.is_file():
        raise FileNotFoundError(f"Native binary not found: {native_binary}")
    if not cvtbase_dir.is_dir():
        raise FileNotFoundError(
            f"cvtbase directory not found: {cvtbase_dir}\n"
            "Extract assets/cvtbase from a Local Dream APK, or set LD_CVTBASE."
        )

    work = output_dir / ".clip_convert_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # Copy cvtbase templates (clip_skip_1.mnn / clip_skip_2.mnn, tokenizer.json, …)
    for item in cvtbase_dir.iterdir():
        dest = work / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    shutil.copy2(safetensors, work / "model.safetensors")

    clip_template = work / ("clip_skip_2.mnn" if clip_skip == 2 else "clip_skip_1.mnn")
    if clip_template.exists():
        shutil.copy2(clip_template, work / "clip_v2.mnn")

    cmd = [str(native_binary.resolve()), "--convert", str(work.resolve())]
    if clip_skip == 2:
        cmd.append("--clip_skip_2")

    print(f"+ {' '.join(cmd)}")
    env = os.environ.copy()
    lib_dir = native_binary.parent
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    result = subprocess.run(cmd, env=env, cwd=str(lib_dir), capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Native CLIP convert failed (exit {result.returncode})")

    finished = work / "finished"
    if not finished.exists():
        raise RuntimeError("Native convert did not produce finished marker")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in CLIP_FILES_SD15:
        src = work / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)
            print(f"  copied {name}")

    shutil.rmtree(work, ignore_errors=True)


def resolve_native_binary(explicit: Path | None) -> Path | None:
    if explicit and explicit.exists():
        return explicit.resolve()
    env = os.environ.get("LD_NATIVE_CONVERT")
    if env and Path(env).exists():
        return Path(env).resolve()
    candidates = [
        Path("libstable_diffusion_core.so"),
        Path("stable_diffusion_core"),
        Path(__file__).resolve().parents[2].parent
        / "app/src/main/cpp/build/host/bin/stable_diffusion_core",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def resolve_cvtbase(explicit: Path | None) -> Path | None:
    if explicit and explicit.is_dir():
        return explicit.resolve()
    env = os.environ.get("LD_CVTBASE")
    if env and Path(env).is_dir():
        return Path(env).resolve()
    local = Path(__file__).parent / "cvtbase"
    if local.is_dir() and any(local.iterdir()):
        return local.resolve()
    return None


def export_clip(
    output_dir: Path,
    *,
    sdxl: bool = False,
    clip_zip: Path | None = None,
    safetensors: Path | None = None,
    native_binary: Path | None = None,
    cvtbase_dir: Path | None = None,
    clip_skip: int = 2,
) -> None:
    if clip_zip:
        extract_from_zip(clip_zip, output_dir, sdxl)
        return

    if safetensors:
        binary = resolve_native_binary(native_binary)
        cvtbase = resolve_cvtbase(cvtbase_dir)
        if binary and cvtbase:
            if sdxl:
                raise NotImplementedError(
                    "Native SDXL CLIP export not yet supported — use --clip-zip"
                )
            export_via_native(
                safetensors,
                output_dir,
                native_binary=binary,
                cvtbase_dir=cvtbase,
                clip_skip=clip_skip,
            )
            return
        raise RuntimeError(
            "MNN CLIP from safetensors requires native binary + cvtbase.\n"
            "Options:\n"
            "  1. --clip-zip PATH   extract from QNN / CPU / litert zip\n"
            "  2. Set LD_NATIVE_CONVERT + LD_CVTBASE, or pass --native-binary + --cvtbase\n"
            "  3. Convert to CPU model in the Local Dream app first, then use that zip"
        )

    raise ValueError("Provide --clip-zip or --safetensors")


def main():
    parser = argparse.ArgumentParser(description="Export MNN CLIP for MTK NPU models")
    parser.add_argument("--clip-zip", "--qnn-zip", dest="clip_zip", help="Source model zip")
    parser.add_argument("--safetensors", help="Checkpoint for native CLIP export")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sdxl", action="store_true")
    parser.add_argument("--clip-skip", type=int, default=2, choices=[1, 2])
    parser.add_argument("--native-binary", type=str, default=None)
    parser.add_argument("--cvtbase", type=str, default=None)
    args = parser.parse_args()

    export_clip(
        Path(args.output_dir),
        sdxl=args.sdxl,
        clip_zip=Path(args.clip_zip) if args.clip_zip else None,
        safetensors=Path(args.safetensors) if args.safetensors else None,
        native_binary=Path(args.native_binary) if args.native_binary else None,
        cvtbase_dir=Path(args.cvtbase) if args.cvtbase else None,
        clip_skip=args.clip_skip,
    )


if __name__ == "__main__":
    main()
