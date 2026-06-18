#!/usr/bin/env python3
"""AOT compile TFLite models for MediaTek NPU (MT6991 / Dimensity 9400+).

Requires LiteRT Python package and Bazel-built AOT compiler.
See: https://developers.google.com/edge/litert/next/mediatek
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Target SoC identifiers for LiteRT AOT compilation
SOC_TARGETS = {
    "MT6991": "mt6991",  # Dimensity 9400 / 9400+
    "MT6990": "mt6990",  # Dimensity 9500
}


def aot_compile_tflite(tflite_path: Path, output_path: Path, soc: str, litert_root: Path):
    """Run LiteRT AOT compilation for a single TFLite model."""
    soc_id = SOC_TARGETS.get(soc.upper())
    if not soc_id:
        raise ValueError(f"Unsupported SOC: {soc}. Supported: {list(SOC_TARGETS)}")

    # LiteRT AOT compilation via Python API (when available)
    try:
        from ai_edge_litert import aot_compile as litert_aot  # type: ignore

        litert_aot.compile(
            model_path=str(tflite_path),
            output_path=str(output_path),
            target_soc=soc_id,
            accelerator="npu",
        )
        return
    except ImportError:
        pass

    # Fallback: invoke LiteRT CLI if installed
    compile_cmd = [
        "litert_compile",
        "--input", str(tflite_path),
        "--output", str(output_path),
        "--target_soc", soc_id,
        "--accelerator", "npu",
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"AOT compile failed for {tflite_path.name}. "
            "Install LiteRT SDK and ensure litert_compile is on PATH."
        )


def main():
    parser = argparse.ArgumentParser(description="AOT compile models for MediaTek NPU")
    parser.add_argument("--input-dir", required=True, help="Directory with .tflite files")
    parser.add_argument("--output-dir", required=True, help="Output directory for .litert files")
    parser.add_argument("--soc", default="MT6991", help="Target SOC (default: MT6991)")
    parser.add_argument("--litert-root", default=None, help="LiteRT SDK root path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    litert_root = Path(args.litert_root) if args.litert_root else None

    for tflite in sorted(input_dir.glob("*.tflite")):
        out_name = tflite.stem + ".litert"
        out_path = output_dir / out_name
        print(f"Compiling {tflite.name} -> {out_name} for {args.soc}...")
        aot_compile_tflite(tflite, out_path, args.soc, litert_root)
        print(f"  -> {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print(f"AOT compilation complete: {output_dir}")


if __name__ == "__main__":
    main()
