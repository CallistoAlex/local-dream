"""Interactive step-by-step conversion wizard."""

from __future__ import annotations

import sys
from pathlib import Path

from ld_convert.config import DEFAULT_CACHE, ConvertConfig, Resolution
from ld_convert.env import (
    check_mediatek,
    check_qualcomm,
    check_system,
    print_check,
    setup_mediatek,
    setup_qualcomm,
)
from ld_convert.mediatek.convert import convert_sd15, convert_sdxl
from ld_convert.platform import HostOS, detect_platform
from ld_convert.qualcomm import sd15 as qnn_sd15
from ld_convert.qualcomm import sdxl as qnn_sdxl
from ld_convert.ui import (
    banner,
    error,
    info,
    step_header,
    success,
    warn,
    ask_choice,
    ask_path,
    ask_text,
    ask_yes_no,
    bold,
    dim,
)
from ld_convert.wsl import reexec_in_wsl_if_needed


GUIDE_URL = "https://ld-guide.chino.icu/conversion/"


def _platform_notes(info) -> list[str]:
    notes: list[str] = []
    if info.os == HostOS.WINDOWS and info.qualcomm_runtime.value == "wsl_relay":
        notes.append("Windows detected — Qualcomm QNN steps will run inside WSL2 automatically.")
    elif info.os == HostOS.WINDOWS:
        notes.append("Install WSL2 for Qualcomm: wsl --install")
    if info.os == HostOS.MACOS:
        notes.append("macOS: ONNX export works locally; QNN convert needs Linux or a VM.")
    if info.in_wsl:
        notes.append("Running inside WSL — full Qualcomm pipeline supported.")
    return notes


def _explain_vendor(vendor: str) -> str:
    if vendor == "qualcomm":
        return (
            "Qualcomm (Snapdragon NPU)\n"
            "  • Output: {name}_qnn2.28_{soc}.zip\n"
            "  • SD1.5: builds min / 8gen1 / 8gen2 tiers (~hours of CPU)\n"
            "  • SDXL: 8gen3 only (experimental, very long run)\n"
            "  • Uses official scripts from ld-guide.chino.icu"
        )
    return (
        "MediaTek (Dimensity 9400+ / LiteRT)\n"
        "  • Output: {name}_litert_d9400.zip\n"
        "  • Fixed 512px (SD1.5) or 1024px (SDXL)\n"
        "  • Needs LiteRT AOT compiler for .litert files\n"
        "  • Reuse MNN CLIP from an existing QNN zip when possible"
    )


def _run_convert_cfg(cfg: ConvertConfig) -> list[Path]:
    if cfg.vendor == "qualcomm":
        if cfg.stage in ("all", "qnn") and not cfg.no_wsl:
            reexec_in_wsl_if_needed()
        if cfg.kind == "sd15":
            return qnn_sd15.convert(cfg)
        return qnn_sdxl.convert(cfg)
    if cfg.kind == "sd15":
        return convert_sd15(cfg)
    return convert_sdxl(cfg)


def _collect_model_options(vendor: str, kind: str) -> dict:
    step_header(4, 6, "Choose your model")
    info(
        "You need a Stable Diffusion checkpoint in .safetensors format.\n"
        "The same file you would use in Automatic1111 / ComfyUI."
    )

    model_path = ask_path("Path to .safetensors file")
    default_name = model_path.stem.replace(" ", "")
    model_name = ask_text("Model name (used in output zip filename)", default=default_name)
    output_dir = ask_path("Output folder for zip files", must_exist=False, default="./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    opts: dict = {
        "model_path": model_path,
        "model_name": model_name,
        "output_dir": output_dir,
    }

    if vendor == "qualcomm":
        if kind == "sd15":
            info(
                "SD1.5 Qualcomm options:\n"
                "  • clip_skip=2 is standard for anime models; use 1 for realistic\n"
                "  • SOC tiers: 8gen2, 8gen1, min — one zip per tier\n"
                "  • Extra resolutions (512×768 etc.) need 64 GB+ RAM"
            )
            clip_raw = ask_text("clip_skip", default="2")
            opts["clip_skip"] = int(clip_raw or "2")
            opts["realistic"] = ask_yes_no("Realistic model? (sets --realistic flag)", default=False)
            if ask_yes_no("Use default SOC tiers (8gen2, 8gen1, min)?", default=True):
                opts["soc"] = "8gen2,8gen1,min"
            else:
                opts["soc"] = ask_text("SOC tiers (comma-separated)", default="8gen2,8gen1,min")
            if ask_yes_no("Add extra resolutions (e.g. 512×768)?", default=False):
                opts["extra_resolutions"] = ask_text(
                    "Resolutions (comma-separated WxH)", default="512x768,768x512"
                )
        else:
            info(
                "SDXL Qualcomm options:\n"
                "  • Only 8gen3 tier today\n"
                "  • Experimental — expect 64 GB+ RAM and many hours"
            )
            opts["soc"] = "8gen3"
            opts["scheduler"] = ask_text("Scheduler", default="dpm")
            opts["cfg"] = ask_text("CFG range (min,max)", default="5,7")
            opts["steps"] = ask_text("Steps range (min,max)", default="15,30")
    else:
        info(
            "MediaTek NPU (Dimensity 9400+ / LiteRT):\n"
            "  • Output: {name}_litert_d9400.zip\n"
            "  • UNet/VAE → .litert via LiteRT AOT\n"
            "  • CLIP stays MNN (same as CPU/QNN models)"
        )
        opts["mtk_soc"] = ask_text("Target SOC", default="MT6991")
        opts["mtk_suffix"] = ask_text("Zip suffix", default="d9400")
        opts["clip_skip"] = int(ask_text("clip_skip for CLIP", default="2"))

        clip_source = ask_choice(
            "How should we get MNN CLIP + tokenizer?",
            [
                ("qnn", "From QNN zip (*_qnn2.28_*.zip)"),
                ("cpu", "From CPU model zip (converted in Local Dream app)"),
                ("litert", "From existing litert zip (reuse CLIP only)"),
                ("native", "Native --convert from safetensors (needs binary + cvtbase)"),
                ("skip", "Skip CLIP (already in components dir)"),
            ],
            default=0,
        )
        if clip_source == "qnn":
            opts["clip_zip"] = ask_path("Path to QNN zip")
        elif clip_source == "cpu":
            opts["clip_zip"] = ask_path("Path to CPU/MNN model zip from the app")
        elif clip_source == "litert":
            opts["clip_zip"] = ask_path("Path to existing litert zip")
        elif clip_source == "native":
            info(
                "Requires stable_diffusion_core --convert and cvtbase templates.\n"
                "Set LD_NATIVE_CONVERT / LD_CVTBASE or enter paths below.\n"
                "cvtbase: extract from Local Dream APK → tools/mtk-convert/cvtbase/"
            )
            nb = ask_text("Native binary path (or empty for LD_NATIVE_CONVERT)", default="")
            if nb:
                opts["native_binary"] = Path(nb)
            cb = ask_text("cvtbase directory (or empty for LD_CVTBASE)", default="")
            if cb:
                opts["cvtbase_dir"] = Path(cb)
        else:
            opts["skip_mnn"] = True

    return opts


def _build_cfg(vendor: str, kind: str, stage: str, cache_dir: Path, opts: dict) -> ConvertConfig:
    extra_res: list[Resolution] = []
    if opts.get("extra_resolutions"):
        for part in opts["extra_resolutions"].split(","):
            part = part.strip()
            if part:
                w, h = part.lower().split("x")
                extra_res.append(Resolution(int(w), int(h)))

    return ConvertConfig(
        vendor=vendor,  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        model_path=opts["model_path"],
        model_name=opts["model_name"],
        output_dir=opts["output_dir"],
        cache_dir=cache_dir,
        clip_skip=opts.get("clip_skip", 2),
        realistic=opts.get("realistic", False),
        soc_versions=opts.get("soc", "8gen2,8gen1,min").split(","),
        extra_resolutions=extra_res,
        scheduler=opts.get("scheduler", "dpm"),
        cfg_range=opts.get("cfg", "5,7"),
        steps_range=opts.get("steps", "15,30"),
        sdxl_soc_versions=opts.get("soc", "8gen3").split(","),
        mtk_soc=opts.get("mtk_soc", "MT6991"),
        mtk_suffix=opts.get("mtk_suffix", "d9400"),
        skip_mnn_clip=opts.get("skip_mnn", False),
        clip_zip=opts.get("clip_zip"),
        native_binary=opts.get("native_binary"),
        cvtbase_dir=opts.get("cvtbase_dir"),
        stage=stage,  # type: ignore[arg-type]
    )


def run_wizard(cache_dir: Path | None = None) -> int:
    cache = cache_dir or DEFAULT_CACHE
    info_obj = detect_platform()

    # ── Welcome ──
    banner(
        "Local Dream — NPU Model Conversion Wizard",
        f"Platform: {info_obj.display_name}  •  Guide: {GUIDE_URL}",
    )
    info(
        "This wizard walks you through converting a Stable Diffusion checkpoint\n"
        "into a zip file that Local Dream can load on your phone's NPU.\n\n"
        "What we'll do:\n"
        "  1. Check your system\n"
        "  2. Download tools & SDKs (first time only)\n"
        "  3. Pick chip vendor + model type\n"
        "  4. Configure your checkpoint\n"
        "  5. Run conversion (may take hours)\n"
        "  6. Get import-ready zip file(s)"
    )
    for note in _platform_notes(info_obj):
        warn(note)
    if not ask_yes_no("Start the wizard?", default=True):
        print(dim("\n  Tip: run `ld-convert convert --help` for non-interactive mode.\n"))
        return 0

    # ── Step 1: Check ──
    step_header(1, 6, "Check your environment")
    info(
        "Verifying uv, optional zstd, and vendor-specific tools.\n"
        "Fix any ✗ errors before continuing."
    )
    sys_result = check_system()
    print_check(sys_result)

    vendor = ask_choice(
        "Which NPU do you want to convert for?",
        [
            ("qualcomm", "Qualcomm Snapdragon (QNN)"),
            ("mediatek", "MediaTek Dimensity 9400+ (LiteRT)"),
        ],
    )
    info(_explain_vendor(vendor))

    vendor_result = (
        check_qualcomm(cache) if vendor == "qualcomm" else check_mediatek()
    )
    print()
    print_check(vendor_result)

    if not sys_result.passed:
        error("System check failed — install missing tools first.")
        info(
            f"Install uv: see {GUIDE_URL}\n"
            "Then re-run: python convert.py (from repo root)"
        )
        return 1

    if not vendor_result.passed and vendor == "qualcomm" and info_obj.os == HostOS.MACOS:
        warn("Some Qualcomm checks failed — you can still run --stage prepare on macOS.")
    elif not vendor_result.passed:
        if not ask_yes_no("Checks reported issues. Continue anyway?", default=False):
            return 1

    # ── Step 2: Setup ──
    step_header(2, 6, "Prepare conversion environment")
    info(
        "First-time setup downloads:\n"
        "  • Official conversion script bundles (from ld-guide)\n"
        "  • QNN SDK 2.28 (Qualcomm) or Python venv (MediaTek)\n"
        "  • Creates isolated Python environments with uv\n\n"
        "Downloads are cached — skipped on subsequent runs."
    )
    if ask_yes_no("Run setup now?", default=True):
        if vendor == "qualcomm":
            if info_obj.qualcomm_runtime.value == "wsl_relay":
                reexec_in_wsl_if_needed(["wizard"])
            print()
            info("Setting up Qualcomm SD1.5 bundle + QNN SDK…")
            bundle, sdk, py = setup_qualcomm(cache_dir=cache, kind="sd15")
            success(f"SD1.5 bundle: {bundle}")
            success(f"QNN SDK: {sdk}")
            setup_qualcomm(cache_dir=cache, kind="sdxl")
            success("SDXL bundle ready")
        else:
            py = setup_mediatek(cache_dir=cache)
            success(f"MediaTek venv: {py.parent}")
    else:
        warn("Skipped setup — conversion will fail if bundles are missing.")

    # ── Step 3: Model type ──
    step_header(3, 6, "Select model type")
    kind = ask_choice(
        "Which Stable Diffusion version is your checkpoint?",
        [
            ("sd15", "SD 1.5 — 512×512, most models, faster conversion"),
            ("sdxl", "SDXL — 1024×1024, slower, experimental on NPU"),
        ],
    )

    # ── Step 4: Model options ──
    opts = _collect_model_options(vendor, kind)

    # ── Step 5: Conversion plan ──
    step_header(5, 6, "Review conversion plan")
    needs_split = vendor == "qualcomm" and not info_obj.can_run_qnn_tools
    stages: list[str]

    if needs_split:
        stages = ["prepare"]  # qnn + package run after user confirms Linux step
        info(
            "Your platform cannot run QNN Linux tools directly.\n"
            "We'll run in three stages:\n\n"
            "  Stage A — prepare (now, on this machine)\n"
            "    Export ONNX + calibration data\n\n"
            "  Stage B — qnn (Linux / WSL / VM)\n"
            "    Quantize to QNN binaries — copy cache folder to Linux, then:\n"
            f"    ld-convert convert {vendor} {kind} --stage qnn \\\n"
            f"      --model-path {opts['model_path']} \\\n"
            f"      --model-name {opts['model_name']}\n\n"
            "  Stage C — package\n"
            "    Create final zip(s) for Local Dream"
        )
    elif vendor == "qualcomm":
        stages = ["all"]
        info(
            "Qualcomm full pipeline:\n"
            "  1. prepare_data.py — calibration images + CLIP\n"
            "  2. gen_quant_data.py — quantization dataset\n"
            "  3. export_onnx.py — ONNX export\n"
            "  4. convert_all.sh — QNN quantize per SOC tier\n"
            "  5. zip — {name}_qnn2.28_{soc}.zip\n\n"
            f"Output folder: {opts['output_dir']}\n"
            "⏱  SD1.5: several hours CPU. SDXL: much longer."
        )
    else:
        stages = ["all"]
        info(
            "MediaTek pipeline:\n"
            "  1. Export ONNX (UNet + VAE)\n"
            "  2. ONNX → TFLite (onnx2tf)\n"
            "  3. LiteRT AOT → .litert\n"
            "  4. MNN CLIP (from zip or native convert)\n"
            "  5. Package → {name}_litert_d9400.zip\n\n"
            "Note: litert_compile is usually Linux-only — ONNX export works on Windows."
        )

    print()
    print(f"  {bold('Summary')}")
    print(f"    Vendor:   {vendor}")
    print(f"    Model:    {kind.upper()}")
    print(f"    File:     {opts['model_path']}")
    print(f"    Name:     {opts['model_name']}")
    print(f"    Output:   {opts['output_dir']}")
    print()

    if not ask_yes_no("Start conversion?", default=True):
        print(dim("\n  Cancelled. Re-run `ld-convert wizard` when ready.\n"))
        return 0

    # ── Step 6: Convert ──
    step_header(6, 6, "Run conversion")
    all_zips: list[Path] = []

    for i, stage in enumerate(stages, 1):
        if len(stages) > 1:
            print(bold(f"\n  ── Sub-step {i}/{len(stages)}: stage={stage} ──\n"))
        cfg = _build_cfg(vendor, kind, stage, cache, opts)
        try:
            zips = _run_convert_cfg(cfg)
            all_zips.extend(zips)
            if stage == "prepare" and needs_split:
                success("Prepare stage finished.")
                print()
                info(
                    f"Copy this cache folder to your Linux machine or VM:\n"
                    f"  {cache}\n\n"
                    "On Linux/WSL, run:\n"
                    f"  python convert.py convert {vendor} {kind} --stage qnn \\\n"
                    f"    --model-path {opts['model_path']} \\\n"
                    f"    --model-name {opts['model_name']} \\\n"
                    f"    --output-dir {opts['output_dir']}\n\n"
                    "Then copy the cache back here (if needed) and continue."
                )
                if ask_yes_no("Have you finished the qnn stage on Linux?", default=False):
                    cfg_qnn = _build_cfg(vendor, kind, "qnn", cache, opts)
                    all_zips.extend(_run_convert_cfg(cfg_qnn))
                    cfg_pkg = _build_cfg(vendor, kind, "package", cache, opts)
                    all_zips.extend(_run_convert_cfg(cfg_pkg))
                else:
                    warn("Run the qnn command above, then: ld-convert wizard (or --stage package)")
                    return 0
        except KeyboardInterrupt:
            print()
            warn("Conversion interrupted.")
            return 130
        except Exception as exc:
            error(str(exc))
            return 1

    # ── Done ──
    banner("Conversion complete!", "Import the zip(s) into Local Dream")
    if all_zips:
        for z in all_zips:
            success(str(z))
        info(
            "In Local Dream app:\n"
            "  1. Open model list → Import / custom model\n"
            "  2. Select the zip file above\n"
            "  3. Choose NPU backend when generating"
        )
    elif needs_split:
        info("Prepare stage done. Finish qnn + package on Linux, then import the zip.")
    else:
        warn("No zip files produced — check logs above.")

    print()
    return 0
