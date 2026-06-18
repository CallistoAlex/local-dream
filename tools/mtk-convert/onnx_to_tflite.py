#!/usr/bin/env python3
"""Convert ONNX models to TFLite via onnx2tf."""

import argparse
import subprocess
import sys
from pathlib import Path


def convert_onnx_to_tflite(onnx_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "onnx2tf",
        "-i", str(onnx_path),
        "-o", str(output_dir / onnx_path.stem),
        "-osd",
        "-coion",
    ]
    print(f"+ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # onnx2tf outputs model_float32.tflite; rename to component name
    tflite_candidates = list((output_dir / onnx_path.stem).glob("*.tflite"))
    if not tflite_candidates:
        raise FileNotFoundError(f"No TFLite output for {onnx_path.name}")
    src = tflite_candidates[0]
    dst = output_dir / f"{onnx_path.stem}.tflite"
    src.rename(dst)
    print(f"  -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="ONNX to TFLite conversion")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for onnx in sorted(input_dir.glob("*.onnx")):
        convert_onnx_to_tflite(onnx, output_dir)


if __name__ == "__main__":
    main()
