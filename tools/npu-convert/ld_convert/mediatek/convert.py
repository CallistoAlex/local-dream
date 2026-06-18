"""MediaTek LiteRT NPU conversion orchestration."""

from __future__ import annotations

from pathlib import Path

from ld_convert.mediatek.clip import export_mnn_clip
from ld_convert.config import ConvertConfig
from ld_convert.env import setup_mediatek
from ld_convert.run import run_python

# Reuse step scripts from tools/mtk-convert (same repo)
MTK_SCRIPTS = Path(__file__).resolve().parents[2].parent / "mtk-convert"

SOC_SUFFIX = {
    "MT6991": "d9400",
    "MT6990": "d9500",
}


def _suffix_for_soc(soc: str, override: str | None) -> str:
    if override:
        return override.lower()
    return SOC_SUFFIX.get(soc.upper(), soc.lower())


def _run_step(python: Path, script: str, args: list[str]) -> None:
    run_python(python, MTK_SCRIPTS / script, args, cwd=MTK_SCRIPTS)


def _clip_zip(cfg: ConvertConfig) -> Path | None:
    return cfg.clip_zip or cfg.qnn_clip_zip


def convert_sd15(cfg: ConvertConfig) -> list[Path]:
    if not cfg.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    python = setup_mediatek(cache_dir=cfg.cache_dir, python_version=cfg.python_version)
    suffix = _suffix_for_soc(cfg.mtk_soc, cfg.mtk_suffix)
    resolution = 512

    work_dir = cfg.output_dir / cfg.model_name / "work"
    component_dir = cfg.output_dir / cfg.model_name / "components"
    component_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== MediaTek SD1.5: export ONNX ===")
    _run_step(
        python,
        "export_onnx.py",
        [
            "--safetensors",
            str(cfg.model_path),
            "--output-dir",
            str(work_dir / "onnx"),
            "--resolution",
            str(resolution),
        ],
    )

    print("\n=== MediaTek SD1.5: ONNX → TFLite ===")
    _run_step(
        python,
        "onnx_to_tflite.py",
        [
            "--input-dir",
            str(work_dir / "onnx"),
            "--output-dir",
            str(work_dir / "tflite"),
        ],
    )

    print(f"\n=== MediaTek SD1.5: AOT compile ({cfg.mtk_soc}) ===")
    aot_args = [
        "--input-dir",
        str(work_dir / "tflite"),
        "--output-dir",
        str(component_dir),
        "--soc",
        cfg.mtk_soc,
    ]
    if cfg.litert_sdk_root:
        aot_args.extend(["--litert-root", str(cfg.litert_sdk_root)])
    _run_step(python, "aot_compile.py", aot_args)

    if not cfg.skip_mnn_clip:
        print("\n=== MediaTek SD1.5: MNN CLIP ===")
        export_mnn_clip(
            python,
            component_dir,
            sdxl=False,
            clip_zip=_clip_zip(cfg),
            safetensors=cfg.model_path if not _clip_zip(cfg) else None,
            clip_skip=cfg.clip_skip,
            native_binary=cfg.native_binary,
            cvtbase_dir=cfg.cvtbase_dir,
        )

    print("\n=== MediaTek SD1.5: package ===")
    zip_path = cfg.output_dir / f"{cfg.model_name}_litert_{suffix}.zip"
    _run_step(
        python,
        "package_model.py",
        [
            "--component-dir",
            str(component_dir),
            "--output",
            str(zip_path),
        ],
    )
    return [zip_path]


def convert_sdxl(cfg: ConvertConfig) -> list[Path]:
    if not cfg.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    python = setup_mediatek(cache_dir=cfg.cache_dir, python_version=cfg.python_version)
    suffix = _suffix_for_soc(cfg.mtk_soc, cfg.mtk_suffix)

    work_dir = cfg.output_dir / cfg.model_name / "work"
    component_dir = cfg.output_dir / cfg.model_name / "components"
    component_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== MediaTek SDXL: export ONNX ===")
    _run_step(
        python,
        "export_onnx.py",
        [
            "--safetensors",
            str(cfg.model_path),
            "--output-dir",
            str(work_dir / "onnx"),
            "--sdxl",
        ],
    )

    print("\n=== MediaTek SDXL: ONNX → TFLite ===")
    _run_step(
        python,
        "onnx_to_tflite.py",
        [
            "--input-dir",
            str(work_dir / "onnx"),
            "--output-dir",
            str(work_dir / "tflite"),
        ],
    )

    print(f"\n=== MediaTek SDXL: AOT compile ({cfg.mtk_soc}) ===")
    aot_args = [
        "--input-dir",
        str(work_dir / "tflite"),
        "--output-dir",
        str(component_dir),
        "--soc",
        cfg.mtk_soc,
    ]
    if cfg.litert_sdk_root:
        aot_args.extend(["--litert-root", str(cfg.litert_sdk_root)])
    _run_step(python, "aot_compile.py", aot_args)

    if not cfg.skip_mnn_clip:
        print("\n=== MediaTek SDXL: MNN CLIP ===")
        export_mnn_clip(
            python,
            component_dir,
            sdxl=True,
            clip_zip=_clip_zip(cfg),
            safetensors=cfg.model_path if not _clip_zip(cfg) else None,
            clip_skip=cfg.clip_skip,
            native_binary=cfg.native_binary,
            cvtbase_dir=cfg.cvtbase_dir,
        )

    print("\n=== MediaTek SDXL: package ===")
    zip_path = cfg.output_dir / f"{cfg.model_name}_litert_{suffix}.zip"
    _run_step(
        python,
        "package_model.py",
        [
            "--component-dir",
            str(component_dir),
            "--output",
            str(zip_path),
            "--sdxl",
        ],
    )
    return [zip_path]
