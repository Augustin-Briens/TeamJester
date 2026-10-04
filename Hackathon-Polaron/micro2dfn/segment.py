"""3-phase BSE segmentation with one shared, frozen threshold pair.

Phases: 0 = pore (dark), 1 = bulk graphite+binder (mid), 2 = silicon (bright).

The threshold pair is fitted ONCE on the baseline batch's flat-fielded
pixels (default Batch_3, the approved reference), saved to recipe.json,
and reused verbatim on every later run — a new batch can never
recalibrate the ruler applied to existing batches. Run provenance is
recorded in run_manifest.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from skimage.measure import label

from . import config
from .io import BSEImage


def fit_thresholds(images: list[BSEImage], seed: int = 0
                   ) -> tuple[float, float]:
    """Fit (t_pore, t_si) on pooled flat-fielded pixels of the given
    images — run_dfn passes the baseline batch only (frozen recipe)."""
    rng = np.random.default_rng(seed)
    per_img = max(10_000, config.SEG_SUBSAMPLE // max(1, len(images)))
    samples = []
    for im in images:
        sm = gaussian_filter(im.gray, config.SEG_SMOOTH_SIGMA_PX)
        flat = sm.ravel()
        idx = rng.choice(flat.size, min(per_img, flat.size), replace=False)
        samples.append(flat[idx])
    pooled = np.concatenate(samples)
    t1, t2 = threshold_multiotsu(pooled, classes=3, nbins=256)
    return float(t1), float(t2)


def save_thresholds(path: str, t_pore: float, t_si: float,
                    n_images: int) -> None:
    with open(path, "w") as fh:
        json.dump({"t_pore": t_pore, "t_si": t_si,
                   "fitted_on_n_images": n_images}, fh, indent=2)


def load_thresholds(path: str) -> tuple[float, float]:
    with open(path) as fh:
        d = json.load(fh)
    return float(d["t_pore"]), float(d["t_si"])


def segment(im: BSEImage, t_pore: float, t_si: float) -> np.ndarray:
    """Apply shared thresholds -> uint8 label map {0,1,2}.

    Sub-resolution objects (Si < MIN_BRIGHT_PX, pores < MIN_FEATURE_PX)
    are almost always threshold speckle, not material; they are merged
    into bulk. Consistent across images -> unbiased batch comparison.
    """
    sm = gaussian_filter(im.gray, config.SEG_SMOOTH_SIGMA_PX)
    seg = np.ones(sm.shape, dtype=np.uint8) * config.BULK
    seg[sm < t_pore] = config.PORE
    seg[sm >= t_si] = config.SI
    return _remove_specks(seg)


def _remove_specks(seg: np.ndarray) -> np.ndarray:
    for phase, min_px in ((config.SI, config.MIN_BRIGHT_PX),
                          (config.PORE, config.MIN_FEATURE_PX)):
        lab = label(seg == phase)
        counts = np.bincount(lab.ravel())
        drop = np.where(counts < min_px)[0]
        drop = drop[drop != 0]
        if len(drop):
            seg[np.isin(lab, drop)] = config.BULK
    return seg


def split_si(si_mask: np.ndarray) -> np.ndarray:
    """Label Si objects, splitting touching ones by watershed."""
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    dist = distance_transform_edt(si_mask)
    coords = peak_local_max(dist, min_distance=config.WATERSHED_MIN_DIST_PX,
                            labels=si_mask)
    markers = np.zeros(si_mask.shape, np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    if len(coords) == 0:
        return label(si_mask)
    return watershed(-dist, markers, mask=si_mask)
