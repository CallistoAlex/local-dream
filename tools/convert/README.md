# Local Dream — Model Conversion (tools)

Convert SD 1.5 / SDXL checkpoints to NPU zip files for the Android app.

| Target | Output | Chipsets |
|--------|--------|----------|
| **Qualcomm** | `{name}_qnn2.28_{soc}.zip` | Snapdragon 8 Gen 1–3+ |
| **MediaTek** | `{name}_litert_d9400.zip` | Dimensity 9400+ (MT6991) |

## Quick start

```bash
# From repo root (recommended)
python convert.py

# Or from this folder
python tools/convert/convert.py
```

Interactive wizard walks through setup → model selection → conversion.

## MediaTek NPU conversion

Pipeline for Dimensity 9400+:

1. **Export ONNX** — UNet + VAE from `.safetensors`
2. **ONNX → TFLite** — via `onnx2tf`
3. **LiteRT AOT** — `.litert` for NeuroPilot NPU
4. **MNN CLIP** — tokenizer + CLIP (shared with CPU/QNN models)
5. **Package** — `{name}_litert_d9400.zip` with `v3` marker

### MNN CLIP sources

MediaTek models need MNN CLIP files. Pick one:

| Source | Flag | When to use |
|--------|------|-------------|
| QNN zip | `--clip-zip model_qnn2.28_8gen2.zip` | You already converted for Qualcomm |
| CPU zip | `--clip-zip MyModel.zip` | Converted in Local Dream app (CPU tab) |
| Litert zip | `--clip-zip existing_litert.zip` | Re-build NPU weights, reuse CLIP |
| Native | `--native-binary` + `--cvtbase` | From safetensors on host |

```bash
python convert.py convert mediatek sd15 \
  --model-path /path/to/model.safetensors \
  --model-name MyModel \
  --output-dir ./output \
  --clip-zip ./MyModel.zip

# Or with QNN zip for CLIP
python convert.py convert mediatek sd15 \
  --model-path model.safetensors \
  --model-name MyModel \
  --clip-zip ./MyModel_qnn2.28_8gen2.zip
```

### Prerequisites (MediaTek)

- `uv` + Python 3.10+
- [LiteRT SDK](https://github.com/google-ai-edge/LiteRT/releases) + `litert_compile`
- `onnx2tf` (installed by `python convert.py setup mediatek`)

### cvtbase (native CLIP only)

For `--native-binary` CLIP export, copy templates from a Local Dream APK:

```bash
unzip -j LocalDream.apk 'assets/cvtbase/*' -d tools/mtk-convert/cvtbase/
export LD_CVTBASE=$PWD/tools/mtk-convert/cvtbase
export LD_NATIVE_CONVERT=/path/to/stable_diffusion_core
```

## Qualcomm NPU conversion

See [ld-guide.chino.icu/conversion](https://ld-guide.chino.icu/conversion/) and `tools/npu-convert/README.md`.

```bash
python convert.py setup qualcomm
python convert.py convert qualcomm sd15 \
  --model-path model.safetensors \
  --model-name MyModel
```

## Layout

```
tools/convert/          ← this hub (README + launcher)
tools/npu-convert/      ← Python package (ld-convert)
tools/mtk-convert/      ← MediaTek step scripts + cvtbase/
convert.py              ← root launcher
```

## Import into app

1. Copy the output zip to your phone
2. Local Dream → NPU Models → Add Custom NPU Model
3. Select the zip (`.bin` for Qualcomm, `.litert` for MediaTek)

HuggingFace collections:

- Qualcomm: `xororz/sd-qnn`, `xororz/sdxl-qnn`
- MediaTek: `xororz/sd-mtk`, `xororz/sdxl-mtk`
