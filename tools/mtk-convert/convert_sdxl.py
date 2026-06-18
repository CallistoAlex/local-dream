#!/usr/bin/env python3
"""End-to-end SDXL conversion pipeline for MediaTek NPU."""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Convert SDXL model for MTK NPU")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--safetensors", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--soc", default="MT6991")
    parser.add_argument("--skip-mnn", action="store_true")
    parser.add_argument("--clip-zip", "--qnn-zip", dest="clip_zip")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir)
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    SOC_SUFFIX = {"MT6991": "d9400", "MT6990": "d9500"}
    zip_suffix = SOC_SUFFIX.get(args.soc.upper(), args.soc.lower())

    run([
        sys.executable, str(script_dir / "export_onnx.py"),
        "--safetensors", args.safetensors,
        "--output-dir", str(work_dir / "onnx"),
        "--sdxl",
    ])

    run([
        sys.executable, str(script_dir / "onnx_to_tflite.py"),
        "--input-dir", str(work_dir / "onnx"),
        "--output-dir", str(work_dir / "tflite"),
    ])

    run([
        sys.executable, str(script_dir / "aot_compile.py"),
        "--input-dir", str(work_dir / "tflite"),
        "--output-dir", str(output_dir),
        "--soc", args.soc,
    ])

    if not args.skip_mnn:
        clip_args = [
            sys.executable, str(script_dir / "export_mnn_clip.py"),
            "--output-dir", str(output_dir),
            "--sdxl",
        ]
        if args.clip_zip:
            clip_args.extend(["--clip-zip", args.clip_zip])
        else:
            clip_args.extend(["--safetensors", args.safetensors])
        run(clip_args)

    zip_name = f"{args.model_id}_litert_{zip_suffix}.zip"
    run([
        sys.executable, str(script_dir / "package_model.py"),
        "--component-dir", str(output_dir),
        "--output", str(output_dir.parent / zip_name),
        "--sdxl",
    ])


if __name__ == "__main__":
    main()
