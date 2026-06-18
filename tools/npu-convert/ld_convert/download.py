"""Download and cache conversion assets."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

from ld_convert.config import (
    DEFAULT_CACHE,
    QUALCOMM_SD15_BUNDLE_URL,
    QUALCOMM_SDXL_BUNDLE_URL,
    QNN_SDK_URL,
    QNN_SDK_VERSION,
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}")
        return dest

    print(f"  downloading: {url}")
    print(f"  -> {dest}")

    class ProgressBar:
        def __init__(self):
            self.bar = None

        def __call__(self, block_num, block_size, total_size):
            if self.bar is None and total_size > 0:
                self.bar = tqdm(total=total_size, unit="B", unit_scale=True)
            if self.bar:
                self.bar.update(block_size)

    tmp = dest.with_suffix(dest.suffix + ".part")
    urlretrieve(url, tmp, reporthook=ProgressBar())
    tmp.rename(dest)
    return dest


def _extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    marker = dest_dir / ".extracted"
    if marker.exists():
        return dest_dir
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    # Bundles contain a single top-level folder (npuconvertv2/ or convertsdxl/)
    children = [p for p in dest_dir.iterdir() if p.is_dir()]
    if len(children) == 1:
        root = children[0]
    else:
        root = dest_dir
    marker.write_text(str(root))
    return root


def ensure_qualcomm_sd15_bundle(cache_dir: Path = DEFAULT_CACHE) -> Path:
    cache = cache_dir / "bundles"
    zip_path = cache / "npuconvertv2.zip"
    _download(QUALCOMM_SD15_BUNDLE_URL, zip_path)
    return _extract_zip(zip_path, cache / "npuconvertv2")


def ensure_qualcomm_sdxl_bundle(cache_dir: Path = DEFAULT_CACHE) -> Path:
    cache = cache_dir / "bundles"
    zip_path = cache / "convertsdxl.zip"
    _download(QUALCOMM_SDXL_BUNDLE_URL, zip_path)
    return _extract_zip(zip_path, cache / "convertsdxl")


def ensure_qnn_sdk(
    cache_dir: Path = DEFAULT_CACHE,
    *,
    sdk_root: Path | None = None,
) -> Path:
    if sdk_root and sdk_root.exists():
        print(f"  using QNN SDK: {sdk_root}")
        return sdk_root.resolve()

    cache = cache_dir / "sdk"
    marker = cache / "qnn_sdk.path"
    if marker.exists():
        stored = Path(marker.read_text(encoding="utf-8").strip())
        if stored.exists():
            return stored.resolve()

    zip_path = cache / f"qnn_{QNN_SDK_VERSION}.zip"
    extract_root = cache / f"qairt_{QNN_SDK_VERSION}"

    if not extract_root.exists():
        _download(QNN_SDK_URL, zip_path)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_root)

    candidates = list(extract_root.rglob("bin/envsetup.sh"))
    if not candidates:
        raise RuntimeError(f"envsetup.sh not found in extracted QNN SDK: {extract_root}")
    sdk_path = candidates[0].parent.parent.resolve()

    cache.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(sdk_path), encoding="utf-8")

    link = cache / "qnn_sdk"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(sdk_path, target_is_directory=True)
    except OSError:
        pass

    return sdk_path


def patch_qnn_sdk_root(bundle_dir: Path, qnn_sdk_root: Path) -> None:
    """Replace hardcoded QNN_SDK_ROOT in shell scripts."""
    qnn_str = str(qnn_sdk_root.resolve())
    patterns = [
        "QNN_SDK_ROOT=/data/qairt/2.28.0.241029",
        "QNN_SDK_ROOT=/data/qairt/2.39.0.250926",
    ]
    for sh in bundle_dir.rglob("*.sh"):
        text = sh.read_text()
        original = text
        for pat in patterns:
            text = text.replace(pat, f"QNN_SDK_ROOT={qnn_str}")
        if text != original:
            sh.write_text(text)
            print(f"  patched QNN_SDK_ROOT in {sh.relative_to(bundle_dir)}")
