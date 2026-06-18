"""Configuration defaults adapted from https://ld-guide.chino.icu/conversion/"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Official script bundles (ld-guide.chino.icu)
QUALCOMM_SD15_BUNDLE_URL = "https://chino.icu/local-dream/npuconvertv2.zip"
QUALCOMM_SDXL_BUNDLE_URL = "https://chino.icu/local-dream/convertsdxl.zip"

# QNN SDK 2.28 — required for conversion (app runtime uses 2.39)
QNN_SDK_URL = (
    "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/"
    "software/qualcomm_neural_processing_sdk/v2.28.0.241029.zip"
)
QNN_SDK_VERSION = "2.28.0.241029"

# LiteRT release for MediaTek AOT (user may override)
LITERT_RELEASE_URL = "https://github.com/google-ai-edge/LiteRT/releases/latest"

from ld_convert.platform import Stage, default_cache_dir

DEFAULT_CACHE = default_cache_dir()
DEFAULT_PYTHON = "3.10.17"

Vendor = Literal["qualcomm", "mediatek"]
ModelKind = Literal["sd15", "sdxl"]


@dataclass
class Resolution:
    width: int
    height: int

    @property
    def label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass
class ConvertConfig:
    vendor: Vendor
    kind: ModelKind
    model_path: Path
    model_name: str
    output_dir: Path
    cache_dir: Path = DEFAULT_CACHE

    # Qualcomm SD1.5
    clip_skip: int = 2
    realistic: bool = False
    soc_versions: list[str] = field(default_factory=lambda: ["8gen2", "8gen1", "min"])
    extra_resolutions: list[Resolution] = field(default_factory=list)
    extra_resolution_soc_versions: list[str] = field(
        default_factory=lambda: ["8gen2", "8gen1"]
    )

    # Qualcomm SDXL
    scheduler: str = "dpm"
    cfg_range: str = "5,7"
    steps_range: str = "15,30"
    sdxl_soc_versions: list[str] = field(default_factory=lambda: ["8gen3"])

    # MediaTek
    mtk_soc: str = "MT6991"
    mtk_suffix: str = "d9400"
    skip_mnn_clip: bool = False
    qnn_clip_zip: Path | None = None  # legacy alias
    clip_zip: Path | None = None  # QNN, CPU/MNN, or litert zip
    native_binary: Path | None = None
    cvtbase_dir: Path | None = None

    # Paths (auto-filled by setup)
    qnn_sdk_root: Path | None = None
    litert_sdk_root: Path | None = None
    python_version: str = DEFAULT_PYTHON
    stage: Stage = "all"
    no_wsl: bool = False

    def output_zip_pattern(self) -> str:
        if self.vendor == "qualcomm":
            return f"{self.model_name}_qnn2.28_{{soc}}.zip"
        return f"{self.model_name}_litert_{self.mtk_suffix.lower()}.zip"
