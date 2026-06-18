#!/usr/bin/env bash
# Local Dream — All-in-one NPU model conversion (repo root)
# Usage: ./convert.sh [wizard|setup|check|convert ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$ROOT/tools/npu-convert"

if [[ ! -f "$TOOL/pyproject.toml" ]]; then
  echo "Error: tools/npu-convert not found" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --directory "$TOOL" ld-convert "$@"
fi

if [[ -x "$TOOL/.venv/bin/ld-convert" ]]; then
  exec "$TOOL/.venv/bin/ld-convert" "$@"
fi

exec python3 "$ROOT/convert.py" "$@"
