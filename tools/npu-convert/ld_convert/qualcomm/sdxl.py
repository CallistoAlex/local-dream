"""Qualcomm SDXL NPU conversion — mirrors ld-guide export_sdxl.sh workflow."""

from __future__ import annotations

from pathlib import Path

from ld_convert.config import ConvertConfig
from ld_convert.env import setup_qualcomm
from ld_convert.run import run_bash_script, run_python, zip_dir
from ld_convert.wsl import ensure_qualcomm_linux_host, reexec_in_wsl_if_needed


def _prepare(cfg: ConvertConfig, bundle: Path, python: Path) -> None:
    realistic = ["--realistic"] if cfg.realistic else []
    print("\n=== SDXL 1024×1024: prepare + quantize + export ONNX ===")
    run_python(
        python,
        bundle / "prepare_data_sdxl.py",
        [
            "--model_path",
            str(cfg.model_path),
            *realistic,
            "--scheduler",
            cfg.scheduler,
            "--cfg",
            cfg.cfg_range,
            "--step",
            cfg.steps_range,
        ],
        cwd=bundle,
    )
    run_python(python, bundle / "gen_quant_data_sdxl.py", [], cwd=bundle)
    run_python(
        python,
        bundle / "export_onnx_sdxl.py",
        ["--model_path", str(cfg.model_path)],
        cwd=bundle,
    )


def _qnn_convert(cfg: ConvertConfig, bundle: Path) -> None:
    print("\n=== SDXL: QNN convert per SOC tier ===")
    for soc in cfg.sdxl_soc_versions:
        print(f"--- SOC tier: {soc} ---")
        run_bash_script("scripts/convert_all_sdxl.sh", ["--min_soc", soc], cwd=bundle)


def _package(cfg: ConvertConfig, bundle: Path) -> list[Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    zips: list[Path] = []
    for soc in cfg.sdxl_soc_versions:
        src = bundle / "output" / f"qnn_models_sdxl_{soc}"
        if not src.is_dir():
            raise RuntimeError(f"Missing SDXL output for {soc}: {src}")
        (src / "SDXL").touch()
        zip_path = cfg.output_dir / f"{cfg.model_name}_qnn2.28_{soc}.zip"
        zip_dir(src, zip_path)
        zips.append(zip_path)
    return zips


def convert(cfg: ConvertConfig) -> list[Path]:
    if not cfg.model_path.is_file():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    reexec_in_wsl_if_needed()
    stage = cfg.stage

    if stage in ("all", "prepare"):
        bundle, sdk, python = setup_qualcomm(
            cache_dir=cfg.cache_dir,
            qnn_sdk_root=cfg.qnn_sdk_root,
            python_version=cfg.python_version,
            kind="sdxl",
        )
        cfg.qnn_sdk_root = sdk
        _prepare(cfg, bundle, python)
        if stage == "prepare":
            print("\nPrepare stage complete.")
            return []

    if stage in ("all", "qnn"):
        ensure_qualcomm_linux_host(no_wsl=cfg.no_wsl)
        bundle, sdk, python = setup_qualcomm(
            cache_dir=cfg.cache_dir,
            qnn_sdk_root=cfg.qnn_sdk_root,
            python_version=cfg.python_version,
            kind="sdxl",
        )
        cfg.qnn_sdk_root = sdk
        if stage == "qnn" and not (bundle / "unet").exists():
            raise RuntimeError("ONNX not found — run --stage prepare first.")
        if stage == "all":
            _prepare(cfg, bundle, python)
        _qnn_convert(cfg, bundle)
        if stage == "qnn":
            print("\nQNN stage complete.")
            return []

    if stage in ("all", "package"):
        bundle, _, _ = setup_qualcomm(
            cache_dir=cfg.cache_dir,
            qnn_sdk_root=cfg.qnn_sdk_root,
            python_version=cfg.python_version,
            kind="sdxl",
        )
        print("\n=== Packaging ===")
        return _package(cfg, bundle)

    raise ValueError(f"Unknown stage: {stage}")
