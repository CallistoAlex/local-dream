#!/usr/bin/env python3
"""Export SD1.5 / SDXL UNet and VAE components to ONNX (CLIP not needed here)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel


def _detect_kind(safetensors_path: str) -> str:
    """Guess sd15 vs sdxl from checkpoint tensor names."""
    from safetensors import safe_open

    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
    joined = " ".join(keys[:200])
    # SDXL single-file / Diffusers keys
    if any(
        marker in joined
        for marker in (
            "conditioner.embedders",
            "text_encoders",
            "text_encoder_2",
            "model.diffusion_model.input_blocks.0.0.in_layers.0.weight",
        )
    ):
        return "sdxl"
    if any("cond_stage_model" in k for k in keys[:50]):
        return "sd15"
    # Heuristic: SDXL UNet in_channels often visible in key names
    if any("model.diffusion_model" in k for k in keys):
        return "sdxl"
    return "sd15"


def _load_unet_vae(safetensors: str, sdxl: bool):
    """Load UNet + VAE only — avoids CLIP/transformers compatibility issues."""
    dtype = torch.float32
    if sdxl:
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        unet = UNet2DConditionModel.from_single_file(
            safetensors,
            config=base,
            subfolder="unet",
            torch_dtype=dtype,
        )
        vae = AutoencoderKL.from_single_file(
            safetensors,
            config=base,
            subfolder="vae",
            torch_dtype=dtype,
        )
        return unet, vae

    unet = UNet2DConditionModel.from_single_file(safetensors, torch_dtype=dtype)
    vae = AutoencoderKL.from_single_file(safetensors, torch_dtype=dtype)
    return unet, vae


def export_unet_sd15(unet, output_dir: Path, resolution: int = 512):
    latent = resolution // 8
    dummy_latent = torch.randn(1, 4, latent, latent)
    dummy_timestep = torch.tensor([999], dtype=torch.long)
    dummy_text = torch.randn(1, 77, 768)

    torch.onnx.export(
        unet,
        (dummy_latent, dummy_timestep, dummy_text),
        str(output_dir / "unet.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_pred"],
        dynamic_axes={
            "sample": {0: "batch"},
            "encoder_hidden_states": {0: "batch"},
            "noise_pred": {0: "batch"},
        },
        opset_version=17,
    )


def export_vae(vae: AutoencoderKL, output_dir: Path, resolution: int = 512):
    latent = resolution // 8
    dummy_latent = torch.randn(1, 4, latent, latent)
    dummy_image = torch.randn(1, 3, resolution, resolution)

    class VaeDecoderWrapper(torch.nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model

        def forward(self, latents):
            return self.vae.decode(latents).sample

    class VaeEncoderWrapper(torch.nn.Module):
        def __init__(self, vae_model):
            super().__init__()
            self.vae = vae_model

        def forward(self, pixel_values):
            dist = self.vae.encode(pixel_values).latent_dist
            return dist.mean, dist.std

    torch.onnx.export(
        VaeDecoderWrapper(vae),
        dummy_latent,
        str(output_dir / "vae_decoder.onnx"),
        input_names=["latents"],
        output_names=["sample"],
        opset_version=17,
    )
    torch.onnx.export(
        VaeEncoderWrapper(vae),
        dummy_image,
        str(output_dir / "vae_encoder.onnx"),
        input_names=["pixel_values"],
        output_names=["mean", "std"],
        opset_version=17,
    )


def _export_unet_sdxl(unet, output_dir: Path, resolution: int = 1024):
    latent = resolution // 8
    dummy_latent = torch.randn(1, 4, latent, latent)
    dummy_timestep = torch.tensor([999], dtype=torch.long)
    dummy_text = torch.randn(1, 77, 2048)
    dummy_pooled = torch.randn(1, 1280)
    dummy_time_ids = torch.randn(1, 6)

    class UNetSDXLWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, sample, timestep, encoder_hidden_states, time_ids, text_embeds):
            return self.model(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample

    wrapper = UNetSDXLWrapper(unet)
    torch.onnx.export(
        wrapper,
        (dummy_latent, dummy_timestep, dummy_text, dummy_time_ids, dummy_pooled),
        str(output_dir / "unet.onnx"),
        input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "time_ids",
            "text_embeds",
        ],
        output_names=["noise_pred"],
        opset_version=17,
    )


def main():
    parser = argparse.ArgumentParser(description="Export SD models to ONNX")
    parser.add_argument("--safetensors", required=True, help="Path to model.safetensors")
    parser.add_argument("--output-dir", required=True, help="Output directory for ONNX files")
    parser.add_argument("--sdxl", action="store_true", help="Export SDXL model")
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Detect SD1.5 vs SDXL from checkpoint (overrides wrong --sdxl flag)",
    )
    parser.add_argument("--resolution", type=int, default=512, help="Fixed resolution (512 or 1024)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sdxl = args.sdxl
    if os.path.isfile(args.safetensors):
        detected = _detect_kind(args.safetensors)
        if detected != ("sdxl" if sdxl else "sd15"):
            print(
                f"WARNING: checkpoint looks like {detected.upper()} "
                f"but you selected {'SDXL' if sdxl else 'SD1.5'}. "
                f"Using detected type: {detected.upper()}",
                file=sys.stderr,
            )
            sdxl = detected == "sdxl"

    resolution = 1024 if sdxl else args.resolution

    try:
        unet, vae = _load_unet_vae(args.safetensors, sdxl)
    except Exception as exc:
        # Fallback: full pipeline (older checkpoints)
        print(f"Component load failed ({exc}), trying full pipeline…", file=sys.stderr)
        if sdxl:
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_single_file(
                args.safetensors, torch_dtype=torch.float32
            )
            unet, vae = pipe.unet, pipe.vae
        else:
            if os.path.isfile(args.safetensors):
                pipe = StableDiffusionPipeline.from_single_file(
                    args.safetensors, torch_dtype=torch.float32
                )
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    args.safetensors, torch_dtype=torch.float32
                )
            unet, vae = pipe.unet, pipe.vae

    if sdxl:
        _export_unet_sdxl(unet, output_dir, resolution)
    else:
        export_unet_sd15(unet, output_dir, resolution)

    export_vae(vae, output_dir, resolution)
    print(f"ONNX export complete ({'SDXL' if sdxl else 'SD1.5'} @ {resolution}px): {output_dir}")


if __name__ == "__main__":
    main()
