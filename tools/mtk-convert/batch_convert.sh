#!/usr/bin/env bash
# Batch convert all Local Dream models for MediaTek NPU (MT6991).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-./output}"
SOC="${SOC:-MT6991}"

convert_sd15() {
  local id="$1" safetensors="$2"
  echo "=== Converting SD1.5: $id ==="
  python3 "$SCRIPT_DIR/convert_sd15.py" \
    --model-id "$id" \
    --safetensors "$safetensors" \
    --output-dir "$OUTPUT_BASE/$id" \
    --soc "$SOC"
}

convert_sdxl() {
  local id="$1" safetensors="$2"
  echo "=== Converting SDXL: $id ==="
  python3 "$SCRIPT_DIR/convert_sdxl.py" \
    --model-id "$id" \
    --safetensors "$safetensors" \
    --output-dir "$OUTPUT_BASE/$id" \
    --soc "$SOC"
}

case "${1:-all}" in
  sd15)
    convert_sd15 anythingv5    "${SAFETENSORS_ANYTHINGV5:-AnythingV5.safetensors}"
    convert_sd15 qteamix       "${SAFETENSORS_QTEAMIX:-QteaMix.safetensors}"
    convert_sd15 cuteyukimix   "${SAFETENSORS_CUTEYUKIMIX:-CuteYukiMix.safetensors}"
    convert_sd15 absolutereality "${SAFETENSORS_ABSOLUTEREALITY:-AbsoluteReality.safetensors}"
    convert_sd15 chilloutmix   "${SAFETENSORS_CHILLOUTMIX:-ChilloutMix.safetensors}"
    ;;
  sdxl)
    convert_sdxl illustrious_v16      "${SAFETENSORS_ILLUSTRIOUS:-illustrious_v16.safetensors}"
    convert_sdxl cyber_realistic_v10  "${SAFETENSORS_CYBERREALISTIC:-cyber_realistic_v10.safetensors}"
    ;;
  all)
    "$0" sd15
    "$0" sdxl
    ;;
  *)
    echo "Usage: $0 {sd15|sdxl|all}"
    exit 1
    ;;
esac

echo "Batch conversion complete. Upload zips from $OUTPUT_BASE to HuggingFace:"
echo "  xororz/sd-mtk   (SD1.5)"
echo "  xororz/sdxl-mtk (SDXL)"
