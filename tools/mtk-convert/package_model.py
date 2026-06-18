#!/usr/bin/env python3
"""Package converted MTK NPU model components into a Local Dream zip."""

import argparse
import zipfile
from pathlib import Path

SD15_FILES = [
    "tokenizer.json",
    "clip_v2.mnn",
    "pos_emb.bin",
    "token_emb.bin",
    "unet.litert",
    "vae_encoder.litert",
    "vae_decoder.litert",
]

SDXL_FILES = [
    "tokenizer.json",
    "clip.mnn",
    "clip_2.mnn",
    "pos_emb.bin",
    "token_emb.bin",
    "pos_emb_2.bin",
    "token_emb_2.bin",
    "unet.litert",
    "vae_encoder.litert",
    "vae_decoder.litert",
]


def package(component_dir: Path, output_zip: Path, sdxl: bool = False):
    required = SDXL_FILES if sdxl else SD15_FILES
    missing = [f for f in required if not (component_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {component_dir}: {missing}")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in required:
            zf.write(component_dir / name, name)
        # Version marker (matches QNN v3 convention)
        zf.writestr("v3", "")

    print(f"Packaged {output_zip} ({output_zip.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Package MTK model for Local Dream")
    parser.add_argument("--component-dir", required=True, help="Directory with model files")
    parser.add_argument("--output", required=True, help="Output zip path")
    parser.add_argument("--sdxl", action="store_true", help="SDXL model layout")
    args = parser.parse_args()

    package(Path(args.component_dir), Path(args.output), args.sdxl)


if __name__ == "__main__":
    main()
