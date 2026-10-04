"""Image loading and illumination correction (vendored from polaron_qc.io)."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom

from . import config


@dataclass
class BSEImage:
    image_id: str          # e.g. img_0grcilhi
    batch: str             # e.g. Batch_3
    path: str
    gray: np.ndarray       # float32, flat-field corrected, ~N(1, .)
    raw: np.ndarray        # original uint8 grayscale
    pixel_um: float = config.PIXEL_UM
    channels: dict = field(default_factory=dict)   # detector -> path


def _pixel_size_um(path: str) -> float:
    """Pixel pitch from TIFF resolution tags (falls back to config)."""
    try:
        with tifffile.TiffFile(path) as t:
            tags = t.pages[0].tags
            if "XResolution" in tags and "ResolutionUnit" in tags:
                num, den = tags["XResolution"].value
                px_per_unit = num / den
                unit = tags["ResolutionUnit"].value
                if unit == 2:      # inch
                    return 25_400.0 / px_per_unit
                if unit == 3:      # cm
                    return 10_000.0 / px_per_unit
    except Exception:
        pass
    return config.PIXEL_UM


def _flat_field(raw: np.ndarray) -> np.ndarray:
    """Divide by a large-scale Gaussian background (estimated on a
    16x-downsampled image — identical result, ~1000x faster)."""
    g = 16
    small = raw[::g, ::g]
    bg_s = gaussian_filter(small, config.FLATFIELD_SIGMA_PX / g,
                           mode="reflect")
    bg = zoom(bg_s, (raw.shape[0] / small.shape[0],
                     raw.shape[1] / small.shape[1]), order=1)
    bg = bg[:raw.shape[0], :raw.shape[1]]
    flat = raw / np.maximum(bg, 1.0)
    flat /= np.median(flat)          # recentre so bulk ~ 1.0
    return flat


def load_bse(path: str, batch: str, downsample: int = 1) -> BSEImage:
    """Load a BSE TIFF, drop colour + edge artefact, flat-field correct."""
    img = tifffile.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    # last columns can carry annotation artefacts
    img = np.ascontiguousarray(img[:, :-2])
    if downsample > 1:
        img = img[::downsample, ::downsample]
    raw = img.astype(np.float32)
    flat = _flat_field(raw)

    image_id = os.path.basename(path).replace("_BSE.tif", "")
    channels = {}
    for det in ("Inlens", "ETD", "SE"):
        p = path.replace("_BSE.tif", f"_{det}.tif")
        if os.path.exists(p):
            channels[det] = p
    return BSEImage(image_id=image_id, batch=batch, path=path,
                    gray=flat, raw=raw,
                    pixel_um=config.PIXEL_UM * downsample,
                    channels=channels)


def discover_batches(root: str, batches: list[str] | None = None
                     ) -> dict[str, list[str]]:
    """Map batch dir -> sorted list of BSE paths."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "Batch_*"))):
        name = os.path.basename(d)
        if os.path.isdir(d) and (batches is None or name in batches):
            files = sorted(glob.glob(os.path.join(d, "*_BSE.tif")))
            if files:
                out[name] = files
    return out
