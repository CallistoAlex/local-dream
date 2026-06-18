# Device QA Checklist — MediaTek Dimensity 9400+ (MT6991)

Run on a physical device with Dimensity 9400/9400+ before release.

## Prerequisites

```bash
adb shell getprop ro.soc.model          # expect MT6991
adb shell getprop ro.board.platform       # expect mt6991
adb shell getprop ro.soc.manufacturer    # expect MediaTek
```

## SD1.5 NPU (sd15mtk)

- [ ] Download Anything V5 NPU model (auto-selects `*_litert_d9400.zip`)
- [ ] txt2img 512×512 — image generates without crash
- [ ] img2img — VAE encoder loads and produces output
- [ ] inpaint with mask — completes successfully
- [ ] Compare output quality vs CPU model (anythingv5cpu)
- [ ] Measure generation time vs CPU baseline

## SDXL NPU (sdxlmtk)

- [ ] Download Illustrious v16 or CyberRealistic v10 MTK model
- [ ] txt2img 1024×1024 — completes without OOM
- [ ] lowram mode (Settings → SDXL low RAM) — works on 12GB RAM devices
- [ ] img2img at 1024×1024
- [ ] Aspect-ratio padded inpaint (non-1:1 ratio)

## Custom Model Import

- [ ] Import custom NPU zip with `.litert` files
- [ ] Verify `mtkcustom` marker created in model directory
- [ ] Custom model appears in NPU tab and runs

## Runtime

- [ ] `mtklibs` copied to `filesDir/runtime_libs/` (not qnnlibs)
- [ ] Backend starts with `--type sd15mtk` or `--type sdxlmtk`
- [ ] No `DSP_LIBRARY_PATH` set for MTK backend (check logcat)
- [ ] App survives backend restart / model switch

## Known Issues to Watch

- Neuron adapter libs missing on some OEM builds → check logcat for
  `libneuronusdk_adapter.mtk.so` load failures
- First inference slow (AOT cache warm-up) — acceptable
- SDXL may require lowram on devices with <16GB RAM

## Benchmark Template

| Test | CPU (s) | MTK NPU (s) | Notes |
|------|---------|-------------|-------|
| SD1.5 txt2img 512, 20 steps | | | |
| SDXL txt2img 1024, 20 steps | | | |
