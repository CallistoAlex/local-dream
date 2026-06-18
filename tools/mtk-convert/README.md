# MediaTek NPU Model Conversion (Dimensity 9400+ / MT6991)

> **Use the root launcher:** [`convert.py`](../../convert.py) at the repo root — `python convert.py`

This directory contains the low-level step scripts used by `ld-convert` for MediaTek
LiteRT AOT conversion. You can also run them directly for debugging.

## Prerequisites

- Ubuntu 22.04 LTS (or WSL)
- Python 3.10+
- [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/releases) with MediaTek dispatch libs
- `litert_compile` on PATH
- Android NDK r28+

## Via ld-convert (recommended)

```bash
cd tools/npu-convert
uv sync

ld-convert setup mediatek
ld-convert check mediatek

ld-convert convert mediatek sd15 \
  --model-path /path/to/AnythingV5.safetensors \
  --model-name AnythingV5 \
  --output-dir ./output \
  --qnn-clip-zip ./AnythingV5_qnn2.28_8gen2.zip
```

## Direct script usage

```bash
cd tools/mtk-convert
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python convert_sd15.py \
  --model-id anythingv5 \
  --safetensors /path/to/AnythingV5.safetensors \
  --output-dir ./output/anythingv5 \
  --soc MT6991
```

## Pipeline

1. **export_onnx.py** — UNet + VAE encoder/decoder
2. **onnx_to_tflite.py** — via onnx2tf
3. **aot_compile.py** — LiteRT AOT → `.litert`
4. **export_mnn_clip.py** — MNN CLIP from zip or native `--convert`
5. **package_model.py** — zip with `v3` marker

### MNN CLIP (`export_mnn_clip.py`)

| Source | Flag |
|--------|------|
| QNN zip | `--clip-zip model_qnn2.28_8gen2.zip` |
| CPU zip (from app) | `--clip-zip MyModel.zip` |
| Native safetensors | `--safetensors model.safetensors` + `LD_NATIVE_CONVERT` + `LD_CVTBASE` |

Place cvtbase templates in `cvtbase/` (extract from Local Dream APK).

## Output naming

Local Dream expects: `{name}_litert_{suffix}.zip`

| SOC | Suffix | Chipset |
|-----|--------|---------|
| MT6991 | d9400 | Dimensity 9400 / 9400+ |
| MT6990 | d9500 | Dimensity 9500 |

## HuggingFace repos

- `xororz/sd-mtk` (SD1.5)
- `xororz/sdxl-mtk` (SDXL)
