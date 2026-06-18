"""Qualcomm SD1.5 NPU conversion — mirrors ld-guide export.sh workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

from ld_convert.config import ConvertConfig, Resolution
from ld_convert.env import setup_qualcomm
from ld_convert.run import run_bash_script, run_python, zstd_patch, zip_dir
from ld_convert.wsl import ensure_qualcomm_linux_host, reexec_in_wsl_if_needed


def _realistic_flag(cfg: ConvertConfig) -> list[str]:
    return ["--realistic"] if cfg.realistic else []


def _prepare_base_512(cfg: ConvertConfig, bundle: Path, python: Path) -> None:
    clip_args = [
        "--model_path",
        str(cfg.model_path),
        "--clip_skip",
        str(cfg.clip_skip),
        *_realistic_flag(cfg),
    ]
    print("\n=== SD1.5 base 512×512: prepare + quantize + export ONNX ===")
    run_python(python, bundle / "prepare_data.py", clip_args, cwd=bundle)
    run_python(python, bundle / "gen_quant_data.py", [], cwd=bundle)
    run_python(
        python,
        bundle / "export_onnx.py",
        ["--model_path", str(cfg.model_path), "--clip_skip", str(cfg.clip_skip)],
        cwd=bundle,
    )


def _qnn_base_512(cfg: ConvertConfig, bundle: Path) -> Path:
    print("\n=== SD1.5 base 512×512: QNN convert per SOC tier ===")
    for soc in cfg.soc_versions:
        print(f"--- SOC tier: {soc} ---")
        run_bash_script("scripts/convert_all.sh", ["--min_soc", soc], cwd=bundle)

    out_512 = bundle / "output_512"
    if out_512.exists():
        shutil.rmtree(out_512)
    shutil.move(str(bundle / "output"), str(out_512))
    return out_512


def _prepare_extra_resolution(
    cfg: ConvertConfig,
    bundle: Path,
    python: Path,
    res: Resolution,
) -> None:
    size = res.label
    print(f"\n=== SD1.5 extra resolution {size}: prepare + export UNet ONNX ===")
    run_python(
        python,
        bundle / "prepare_data.py",
        [
            "--model_path",
            str(cfg.model_path),
            "--clip_skip",
            str(cfg.clip_skip),
            "--height",
            str(res.height),
            "--width",
            str(res.width),
            *_realistic_flag(cfg),
        ],
        cwd=bundle,
    )
    run_python(python, bundle / "gen_quant_data.py", [], cwd=bundle)
    run_python(
        python,
        bundle / "export_onnx_unet_only.py",
        [
            "--model_path",
            str(cfg.model_path),
            "--clip_skip",
            str(cfg.clip_skip),
            "--height",
            str(res.height),
            "--width",
            str(res.width),
        ],
        cwd=bundle,
    )


def _qnn_extra_resolution(
    cfg: ConvertConfig,
    bundle: Path,
    out_512: Path,
    res: Resolution,
) -> None:
    size = res.label
    print(f"\n=== SD1.5 extra resolution {size}: QNN convert ===")
    for soc in cfg.extra_resolution_soc_versions:
        run_bash_script(
            "scripts/convert_all_unet_only.sh",
            ["--min_soc", soc],
            cwd=bundle,
        )

    out_res = bundle / f"output_{size}"
    if out_res.exists():
        shutil.rmtree(out_res)
    shutil.move(str(bundle / "output"), str(out_res))

    for soc in cfg.extra_resolution_soc_versions:
        base_unet = out_512 / f"qnn_models_{soc}" / "unet.bin"
        new_unet = out_res / f"qnn_models_{soc}" / "unet.bin"
        patch_out = out_512 / f"qnn_models_{soc}" / f"{size}.patch"
        if base_unet.exists() and new_unet.exists():
            zstd_patch(base_unet, new_unet, patch_out)
            print(f"  patch: {patch_out.name} for {soc}")


def _package(cfg: ConvertConfig, out_512: Path) -> list[Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    zips: list[Path] = []
    for soc in cfg.soc_versions:
        src = out_512 / f"qnn_models_{soc}"
        if not src.is_dir():
            raise RuntimeError(f"Missing output for SOC tier {soc}: {src}")
        zip_path = cfg.output_dir / f"{cfg.model_name}_qnn2.28_{soc}.zip"
        zip_dir(src, zip_path)
        zips.append(zip_path)
    return zips


def _resolve_out_512(bundle: Path) -> Path:
    out_512 = bundle / "output_512"
    if not out_512.is_dir():
        raise RuntimeError(
            f"Missing {out_512} — run --stage prepare and --stage qnn first."
        )
    return out_512


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
            kind="sd15",
        )
        cfg.qnn_sdk_root = sdk
        _prepare_base_512(cfg, bundle, python)
        for res in cfg.extra_resolutions:
            _prepare_extra_resolution(cfg, bundle, python, res)
        if stage == "prepare":
            print("\nPrepare stage complete (ONNX + calibration data in bundle cache).")
            return []

    if stage in ("all", "qnn"):
        ensure_qualcomm_linux_host(no_wsl=cfg.no_wsl)
        bundle, sdk, python = setup_qualcomm(
            cache_dir=cfg.cache_dir,
            qnn_sdk_root=cfg.qnn_sdk_root,
            python_version=cfg.python_version,
            kind="sd15",
        )
        cfg.qnn_sdk_root = sdk
        if stage == "qnn" and not (bundle / "unet").exists():
            raise RuntimeError("ONNX not found — run --stage prepare first.")
        out_512 = _qnn_base_512(cfg, bundle)
        for res in cfg.extra_resolutions:
            if stage == "qnn":
                _prepare_extra_resolution(cfg, bundle, python, res)
            _qnn_extra_resolution(cfg, bundle, out_512, res)
        if stage == "qnn":
            print("\nQNN stage complete.")
            return []

    if stage in ("all", "package"):
        bundle, _, _ = setup_qualcomm(
            cache_dir=cfg.cache_dir,
            qnn_sdk_root=cfg.qnn_sdk_root,
            python_version=cfg.python_version,
            kind="sd15",
        )
        print("\n=== Packaging ===")
        return _package(cfg, _resolve_out_512(bundle))

    raise ValueError(f"Unknown stage: {stage}")
