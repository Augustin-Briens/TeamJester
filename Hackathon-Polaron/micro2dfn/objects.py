"""Object-level bright-phase classification: si_particle vs bright_fine.

Low-contrast photos over-count silicon (carbon-binder regions and particle
edges get tinted bright). The fix: measure each bright object's interior
flat-field intensity and shape, calibrate the interior-brightness cutoff
on the set of large, clearly-particle objects in the baseline batch
only (frozen recipe — see run_dfn.py), then split every object into
si_particle or bright_fine.

This is the known Batch_1 confound (img_4ih2ggld, img_5n1q8atc) handled
as data, not silently absorbed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops

from . import config
from .io import BSEImage
from .segment import split_si


def extract_objects(im: BSEImage, si_mask: np.ndarray) -> pd.DataFrame:
    """One row per watershed-split bright object."""
    lab = split_si(si_mask)
    border = np.zeros(si_mask.shape, bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    rows = []
    for r in regionprops(lab):
        if r.area < config.MIN_BRIGHT_PX:
            continue
        coords = r.coords
        eroded = binary_erosion(lab == r.label,
                                iterations=config.OBJECT_ERODE_PX)
        interior = im.gray[eroded] if eroded.any() else im.gray[
            tuple(coords.T)]
        eq_diam = np.sqrt(4 * r.area / np.pi) * im.pixel_um
        rows.append({
            "batch": im.batch, "image_id": im.image_id,
            "label": int(r.label),
            "area_px": int(r.area),
            "area_um2": float(r.area * im.pixel_um ** 2),
            "eq_diam_um": float(eq_diam),
            "interior_median": float(np.median(interior)),
            "interior_std": float(interior.std()),
            "solidity": float(r.solidity),
            "aspect": float(r.axis_major_length
                            / max(r.axis_minor_length, 1e-9)),
            "touches_border": bool(border[tuple(coords.T)].any()),
            "centroid_y_um": float(r.centroid[0] * im.pixel_um),
            "centroid_x_um": float(r.centroid[1] * im.pixel_um),
        })
    return pd.DataFrame(rows)


def calibrate_t_core(objects: pd.DataFrame) -> float:
    """Interior-brightness cutoff from the pooled calibration set:
    large (>=2 um), compact (solidity>=0.85) objects — clearly particles.
    t_core = their 5th-percentile interior intensity."""
    calib = objects[(objects["eq_diam_um"] >= config.SI_CALIB_MIN_DIAM_UM)
                    & (objects["solidity"] >= config.SI_CALIB_MIN_SOLIDITY)]
    if len(calib) < 20:
        # degenerate fallback: pooled 25th percentile of all interiors
        return float(np.percentile(objects["interior_median"], 25))
    return float(np.percentile(calib["interior_median"],
                               config.SI_CORE_PERCENTILE))


def classify_objects(objects: pd.DataFrame, t_core: float
                     ) -> pd.DataFrame:
    """Add a `kind` column: si_particle or bright_fine."""
    objects = objects.copy()
    objects["kind"] = np.where(
        (objects["interior_median"] >= t_core)
        & (objects["solidity"] >= config.SI_MIN_SOLIDITY),
        "si_particle", "bright_fine")
    return objects


def particle_mask(im: BSEImage, si_mask: np.ndarray,
                  objects: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild (si_particle_mask, bright_fine_mask) from classified objects."""
    lab = split_si(si_mask)
    part = np.zeros(si_mask.shape, bool)
    fine = np.zeros(si_mask.shape, bool)
    sub = objects[objects["image_id"] == im.image_id]
    for _, row in sub.iterrows():
        m = lab == int(row["label"])
        (part if row["kind"] == "si_particle" else fine)[m] = True
    return part, fine
