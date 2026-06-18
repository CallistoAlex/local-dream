#!/usr/bin/env bash
# Upload converted MTK model zips to HuggingFace repos.
# Requires: huggingface-cli logged in with write access to xororz/sd-mtk and xororz/sdxl-mtk
set -euo pipefail

OUTPUT_BASE="${1:-./output}"

upload_sd15() {
  local repo="xororz/sd-mtk"
  for zip in "$OUTPUT_BASE"/*_litert_mt6991.zip; do
    [ -f "$zip" ] || continue
    echo "Uploading $(basename "$zip") -> $repo"
    huggingface-cli upload "$repo" "$zip" "$(basename "$zip")"
  done
}

upload_sdxl() {
  local repo="xororz/sdxl-mtk"
  for zip in "$OUTPUT_BASE"/*_litert_mt6991.zip; do
    [ -f "$zip" ] || continue
    echo "Uploading $(basename "$zip") -> $repo"
    huggingface-cli upload "$repo" "$zip" "$(basename "$zip")"
  done
}

echo "Create HuggingFace repos if they do not exist:"
echo "  huggingface-cli repo create sd-mtk --type model"
echo "  huggingface-cli repo create sdxl-mtk --type model"
echo ""

case "${2:-all}" in
  sd15) upload_sd15 ;;
  sdxl) upload_sdxl ;;
  all)
    upload_sd15
    upload_sdxl
    ;;
  *)
    echo "Usage: $0 [output_dir] {sd15|sdxl|all}"
    exit 1
    ;;
esac

echo "Upload complete."
