"""MediaTek NPU conversion — CLIP export helpers."""

from __future__ import annotations

from pathlib import Path

from ld_convert.run import run_python

MTK_SCRIPTS = Path(__file__).resolve().parents[2].parent / "mtk-convert"


def export_mnn_clip(
    python: Path,
    output_dir: Path,
    *,
    sdxl: bool,
    clip_zip: Path | None,
    safetensors: Path | None,
    clip_skip: int = 2,
    native_binary: Path | None = None,
    cvtbase_dir: Path | None = None,
) -> None:
    args = ["--output-dir", str(output_dir)]
    if sdxl:
        args.append("--sdxl")
    args.extend(["--clip-skip", str(clip_skip)])
    if clip_zip:
        args.extend(["--clip-zip", str(clip_zip)])
    elif safetensors:
        args.extend(["--safetensors", str(safetensors)])
    if native_binary:
        args.extend(["--native-binary", str(native_binary)])
    if cvtbase_dir:
        args.extend(["--cvtbase", str(cvtbase_dir)])
    run_python(python, MTK_SCRIPTS / "export_mnn_clip.py", args, cwd=MTK_SCRIPTS)
