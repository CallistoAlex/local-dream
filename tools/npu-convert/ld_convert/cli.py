#!/usr/bin/env python3
"""Unified CLI for Local Dream NPU model conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ld_convert.config import DEFAULT_CACHE, ConvertConfig, Resolution
from ld_convert.env import (
    check_mediatek,
    check_qualcomm,
    check_system,
    print_check,
    setup_mediatek,
    setup_qualcomm,
)
from ld_convert.wsl import reexec_in_wsl_if_needed
from ld_convert.mediatek.convert import convert_sd15, convert_sdxl
from ld_convert.qualcomm import sd15 as qnn_sd15
from ld_convert.qualcomm import sdxl as qnn_sdxl
from ld_convert.wizard import run_wizard


def _parse_resolutions(raw: str | None) -> list[Resolution]:
    if not raw:
        return []
    out: list[Resolution] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        w, h = part.lower().split("x")
        out.append(Resolution(int(w), int(h)))
    return out


def cmd_setup(args: argparse.Namespace) -> int:
    if args.target in ("qualcomm", "all") and not args.no_wsl:
        reexec_in_wsl_if_needed()
    targets = ["qualcomm", "mediatek"] if args.target == "all" else [args.target]
    for t in targets:
        print(f"\n=== Setup: {t} ===")
        if t == "qualcomm":
            bundle, sdk, python = setup_qualcomm(
                cache_dir=args.cache_dir,
                qnn_sdk_root=args.qnn_sdk_root,
                python_version=args.python,
                kind="sd15",
            )
            print(f"  bundle: {bundle}")
            print(f"  QNN SDK: {sdk}")
            print(f"  python: {python}")
            setup_qualcomm(
                cache_dir=args.cache_dir,
                qnn_sdk_root=args.qnn_sdk_root,
                python_version=args.python,
                kind="sdxl",
            )
            print("  SDXL bundle ready")
        else:
            python = setup_mediatek(
                cache_dir=args.cache_dir,
                python_version=args.python,
                recreate=args.recreate,
            )
            print(f"  python: {python}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    targets = ["qualcomm", "mediatek", "system"] if args.target == "all" else [args.target]
    failed = False
    for t in targets:
        print(f"\n=== Check: {t} ===")
        if t == "system":
            result = check_system()
        elif t == "qualcomm":
            result = check_qualcomm(args.cache_dir, args.qnn_sdk_root, no_wsl=args.no_wsl)
        else:
            result = check_mediatek(args.litert_sdk_root)
        print_check(result)
        if not result.passed:
            failed = True
    return 1 if failed else 0


def _build_config(args: argparse.Namespace) -> ConvertConfig:
    return ConvertConfig(
        vendor=args.vendor,
        kind=args.kind,
        model_path=Path(args.model_path),
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        cache_dir=args.cache_dir,
        clip_skip=args.clip_skip,
        realistic=args.realistic,
        soc_versions=args.soc.split(",") if args.soc else ["8gen2", "8gen1", "min"],
        extra_resolutions=_parse_resolutions(args.extra_resolutions),
        extra_resolution_soc_versions=(
            args.extra_soc.split(",") if args.extra_soc else ["8gen2", "8gen1"]
        ),
        scheduler=args.scheduler,
        cfg_range=args.cfg,
        steps_range=args.steps,
        sdxl_soc_versions=args.soc.split(",") if args.soc else ["8gen3"],
        mtk_soc=args.mtk_soc,
        mtk_suffix=args.mtk_suffix,
        skip_mnn_clip=args.skip_mnn,
        clip_zip=Path(args.clip_zip) if getattr(args, "clip_zip", None) else None,
        native_binary=Path(args.native_binary) if args.native_binary else None,
        cvtbase_dir=Path(args.cvtbase) if args.cvtbase else None,
        qnn_sdk_root=args.qnn_sdk_root,
        litert_sdk_root=args.litert_sdk_root,
        python_version=args.python,
        stage=args.stage,
        no_wsl=args.no_wsl,
    )


def cmd_convert(args: argparse.Namespace) -> int:
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f)
        jobs = data if isinstance(data, list) else [data]
        zips: list[Path] = []
        for job in jobs:
            ns = argparse.Namespace(**{**vars(args), **job})
            zips.extend(_run_convert(ns))
        print(f"\nDone. {len(zips)} zip(s) produced.")
        return 0

    zips = _run_convert(args)
    print(f"\nDone. {len(zips)} zip(s) produced:")
    for z in zips:
        print(f"  {z}")
    return 0


def _run_convert(args: argparse.Namespace) -> list[Path]:
    if args.vendor == "qualcomm" and args.stage in ("all", "qnn") and not args.no_wsl:
        reexec_in_wsl_if_needed()
    cfg = _build_config(args)
    if cfg.vendor == "qualcomm":
        if cfg.kind == "sd15":
            return qnn_sd15.convert(cfg)
        return qnn_sdxl.convert(cfg)
    if cfg.kind == "sd15":
        return convert_sd15(cfg)
    return convert_sdxl(cfg)


def cmd_wizard(args: argparse.Namespace) -> int:
    try:
        return run_wizard(cache_dir=args.cache_dir)
    except KeyboardInterrupt:
        print("\n")
        return 130


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ld-convert",
        description=(
            "Automate NPU model conversion for Local Dream "
            "(Qualcomm QNN + MediaTek LiteRT). "
            "Guide: https://ld-guide.chino.icu/conversion/"
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Download/cache directory (default: {DEFAULT_CACHE})",
    )
    parser.add_argument("--python", default="3.10.17", help="Python version for uv venv")
    parser.add_argument(
        "--qnn-sdk-root",
        type=Path,
        default=None,
        help="Existing QNN SDK 2.28 root (skip auto-download)",
    )
    parser.add_argument(
        "--litert-sdk-root",
        type=Path,
        default=None,
        help="LiteRT SDK root for MediaTek AOT",
    )

    sub = parser.add_subparsers(dest="command")

    p_wizard = sub.add_parser(
        "wizard",
        aliases=["guide", "interactive"],
        help="Interactive step-by-step guide (default when no command given)",
    )
    p_wizard.set_defaults(func=cmd_wizard)

    p_setup = sub.add_parser("setup", help="Download bundles/SDKs and create venvs")
    p_setup.add_argument("target", choices=["qualcomm", "mediatek", "all"])
    p_setup.add_argument("--recreate", action="store_true", help="Recreate MTK venv")
    p_setup.add_argument(
        "--no-wsl",
        action="store_true",
        help="Do not auto-relay into WSL on Windows",
    )
    p_setup.set_defaults(func=cmd_setup)

    p_check = sub.add_parser("check", help="Verify environment")
    p_check.add_argument("target", choices=["qualcomm", "mediatek", "system", "all"])
    p_check.add_argument(
        "--no-wsl",
        action="store_true",
        help="Do not assume WSL relay for Qualcomm checks",
    )
    p_check.set_defaults(func=cmd_check)

    p_conv = sub.add_parser("convert", help="Run conversion pipeline")
    p_conv.add_argument("--config", type=Path, help="YAML batch config")
    p_conv.add_argument("vendor", nargs="?", choices=["qualcomm", "mediatek"])
    p_conv.add_argument("kind", nargs="?", choices=["sd15", "sdxl"])
    p_conv.add_argument("--model-path", help="Path to .safetensors checkpoint")
    p_conv.add_argument("--model-name", help="Output model name (zip prefix)")
    p_conv.add_argument("--output-dir", default="./output", help="Output directory")

    p_conv.add_argument(
        "--stage",
        choices=["all", "prepare", "qnn", "package"],
        default="all",
        help=(
            "Pipeline stage (Qualcomm): prepare=ONNX on any OS; "
            "qnn=Linux/WSL only; package=zip"
        ),
    )
    p_conv.add_argument(
        "--no-wsl",
        action="store_true",
        help="Do not auto-relay into WSL on Windows",
    )

    # Qualcomm SD1.5
    p_conv.add_argument("--clip-skip", type=int, default=2, choices=[1, 2],
                        help="CLIP skip (Qualcomm SD1.5 + MediaTek native CLIP)")
    p_conv.add_argument("--realistic", action="store_true")
    p_conv.add_argument(
        "--soc",
        help="SOC tiers comma-separated (SD1.5: min,8gen1,8gen2; SDXL: 8gen3)",
    )
    p_conv.add_argument(
        "--extra-resolutions",
        help="Extra SD1.5 resolutions, e.g. 512x768,768x512",
    )
    p_conv.add_argument(
        "--extra-soc",
        help="SOC tiers for extra SD1.5 resolutions (default: 8gen2,8gen1)",
    )

    # Qualcomm SDXL
    p_conv.add_argument("--scheduler", default="dpm")
    p_conv.add_argument("--cfg", default="5,7")
    p_conv.add_argument("--steps", default="15,30")

    # MediaTek
    p_conv.add_argument("--mtk-soc", default="MT6991")
    p_conv.add_argument("--mtk-suffix", default="d9400")
    p_conv.add_argument("--skip-mnn", action="store_true", help="Skip MNN CLIP export")
    p_conv.add_argument(
        "--clip-zip",
        "--qnn-clip-zip",
        dest="clip_zip",
        help="Extract MNN CLIP from model zip (QNN, CPU/MNN, litert)",
    )
    p_conv.add_argument(
        "--native-binary",
        type=Path,
        help="stable_diffusion_core for native CLIP export (or LD_NATIVE_CONVERT)",
    )
    p_conv.add_argument(
        "--cvtbase",
        type=Path,
        help="cvtbase template dir (or LD_CVTBASE)",
    )

    p_conv.set_defaults(func=cmd_convert)

    args = parser.parse_args(argv)
    if args.command is None:
        return cmd_wizard(args)
    if args.command == "convert" and not args.config:
        if not args.vendor or not args.kind or not args.model_path or not args.model_name:
            parser.error(
                "convert requires --model-path and --model-name, "
                "or --config for batch mode"
            )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
