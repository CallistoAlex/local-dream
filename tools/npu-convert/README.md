# Local Dream NPU Convert

Unified Python automation for converting Stable Diffusion checkpoints to NPU assets
used by [Local Dream](https://github.com/xororz/local-dream).

**Run from the repository root (recommended):**

```bash
python convert.py              # interactive wizard
./convert.sh                   # Linux / macOS
convert.bat                    # Windows
```

The implementation lives in `tools/npu-convert/`; the root launchers delegate here automatically.

Adapted from the official guide: [ld-guide.chino.icu/conversion](https://ld-guide.chino.icu/conversion/)

| Vendor | Runtime in app | Conversion SDK | Output |
|--------|----------------|----------------|--------|
| Qualcomm | QNN 2.39 | QNN 2.28 (pinned) | `{name}_qnn2.28_{soc}.zip` |
| MediaTek | LiteRT | LiteRT AOT | `{name}_litert_{suffix}.zip` |

## Requirements

- **Windows 10/11**, **macOS 12+**, or **Linux** (native or WSL2)
- [uv](https://docs.astral.sh/uv/) package manager
- `zstd` (Qualcomm SD1.5 extra-resolution patches only)
- **Qualcomm:** QNN SDK 2.28.0.241029 — Linux/WSL required for QNN convert step
- **MediaTek:** LiteRT SDK + `litert_compile` (platform-dependent)

### Platform matrix

| Step | Linux | Windows (WSL2) | Windows (native) | macOS |
|------|-------|----------------|------------------|-------|
| Setup / download bundles | ✅ | ✅ auto-relay | ✅ cache only | ✅ |
| Qualcomm `--stage prepare` | ✅ | ✅ | ✅ | ✅ |
| Qualcomm `--stage qnn` | ✅ | ✅ auto-relay | ❌ need WSL | ❌ use VM / split stages |
| Qualcomm `--stage package` | ✅ | ✅ | ✅ | ✅ |
| MediaTek ONNX → TFLite | ✅ | ✅ | ✅ | ✅ |
| MediaTek AOT (`litert_compile`) | ✅ | varies | varies | varies |

On **Windows**, install WSL2 (`wsl --install`) — `ld-convert` automatically continues inside WSL for Qualcomm setup and QNN conversion.

On **macOS**, run `--stage prepare` locally, then finish `--stage qnn` on a Linux machine or VM.

### Hardware (from guide)

| Workflow | RAM + swap | Disk |
|----------|--------------|------|
| SD1.5 @ 512×512 | ~20 GB | ~30 GB |
| SD1.5 extra resolutions | 64 GB+ | 60 GB+ |
| SDXL @ 1024×1024 | 64 GB+ | 60 GB+ |

## Install

```bash
cd tools/npu-convert
uv sync
uv run ld-convert --help
```

Or install globally:

```bash
uv tool install -e tools/npu-convert
ld-convert check all
```

## Quick Start

### Interactive wizard (recommended)

```bash
# From repo root
python convert.py

# Or from this directory
cd tools/npu-convert
uv sync
uv run ld-convert
```

The wizard walks you through six steps:

1. **Check environment** — uv, zstd, platform notes
2. **Setup** — download bundles & SDKs (first run only)
3. **Model type** — SD1.5 or SDXL
4. **Configure checkpoint** — path, name, SOC / MTK options
5. **Review plan** — what each pipeline stage does
6. **Convert** — runs the pipeline (hours for full Qualcomm)

Aliases: `ld-convert guide`, `ld-convert interactive`

### Manual commands

```bash
# Download official script bundles + QNN SDK + venvs
ld-convert setup qualcomm

# MediaTek venv (torch, diffusers, onnx2tf)
ld-convert setup mediatek

# Both
ld-convert setup all
```

### 2. Verify

```bash
ld-convert check qualcomm
ld-convert check mediatek
```

### 3. Convert

**Qualcomm SD1.5** (matches [SD1.5 guide](https://ld-guide.chino.icu/conversion/sd15/)):

```bash
ld-convert convert qualcomm sd15 \
  --model-path /path/to/model.safetensors \
  --model-name MyModel \
  --output-dir ./output \
  --clip-skip 2 \
  --soc 8gen2,8gen1,min
```

With extra resolutions (512×768 patch zips):

```bash
ld-convert convert qualcomm sd15 \
  --model-path /path/to/model.safetensors \
  --model-name MyModel \
  --extra-resolutions 512x768,768x512 \
  --extra-soc 8gen2,8gen1
```

**Qualcomm SDXL** ([SDXL guide](https://ld-guide.chino.icu/conversion/sdxl/)):

```bash
ld-convert convert qualcomm sdxl \
  --model-path /path/to/sdxl.safetensors \
  --model-name MySdxl \
  --soc 8gen3 \
  --scheduler dpm \
  --cfg 5,7 \
  --steps 15,30
```

**MediaTek SD1.5** (Dimensity 9400+ / MT6991):

```bash
ld-convert convert mediatek sd15 \
  --model-path /path/to/model.safetensors \
  --model-name MyModel \
  --output-dir ./output \
  --mtk-soc MT6991 \
  --mtk-suffix d9400 \
  --qnn-clip-zip ./MyModel_qnn2.28_8gen2.zip
```

> **Tip:** Reuse MNN CLIP from an existing QNN zip via `--qnn-clip-zip` — native MNN export needs the Local Dream `--convert` binary.

**MediaTek SDXL:**

```bash
ld-convert convert mediatek sdxl \
  --model-path /path/to/sdxl.safetensors \
  --model-name MySdxl \
  --qnn-clip-zip ./MySdxl_qnn2.28_8gen3.zip
```

### Batch config (YAML)

```bash
ld-convert convert --config examples/batch.yaml
```

See `examples/batch.yaml` for a multi-model example.

## What it automates

### Qualcomm

1. Downloads official bundles:
   - SD1.5: `npuconvertv2.zip`
   - SDXL: `convertsdxl.zip`
2. Downloads or uses existing QNN SDK 2.28
3. **Patches** hardcoded `QNN_SDK_ROOT=/data/qairt/2.28.0.241029` in shell scripts
4. Creates `uv` venv with Python 3.10.17 inside each bundle
5. Runs the same steps as the guide's `export.sh` / `export_sdxl.sh`
6. Packages `{name}_qnn2.28_{soc}.zip` (SDXL includes `SDXL` marker file)

### MediaTek

Orchestrates the pipeline in `tools/mtk-convert/`:

1. Export ONNX (UNet + VAE)
2. ONNX → TFLite (`onnx2tf`)
3. LiteRT AOT compile → `.litert`
4. MNN CLIP (from QNN zip or native)
5. Package `{name}_litert_{suffix}.zip` with `v3` marker

## Cache layout

Default cache locations:

| OS | Path |
|----|------|
| Linux / macOS / WSL | `~/.cache/local-dream-npu-convert/` |
| Windows (native) | `%LOCALAPPDATA%\local-dream-npu-convert\` |

Override with `--cache-dir`.

## Notes

- **Two QNN versions:** Conversion uses 2.28; the Android app ships 2.39 runtime. Do not mix SDK versions in one run.
- **SD1.5 MTK:** Fixed 512×512 only (no zstd resolution patches like QNN).
- **SDXL:** Experimental on both vendors per the guide.

## Related

- `tools/mtk-convert/` — low-level MediaTek step scripts (used by this tool)
- [Device QA checklist](../docs/device-qa-mt6991.md) — MT6991 testing
