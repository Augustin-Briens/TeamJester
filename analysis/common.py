"""Shared constants, IO and helpers for the baseline/delta analysis.

All quantities in physical units use um (micrometres); the pixel size is read
from the TIFF resolution tags (25.4e6 um per inch / dpi). The images are RGB
TIFFs whose channels are equal everywhere except a 1-4 px coloured marker
stripe on the left/right edge, which is cropped off.

ASSUMPTION (applies to every output of this pipeline):
    The bright particles in BSE are assumed to be silicon-based and the grey
    bulk is assumed to be graphite; black is pore or crack. This comes from
    image appearance and is UNCONFIRMED by composition measurement.
"""
import os
import glob
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

# ---------------------------------------------------------------------------
# Paths / batches
# ---------------------------------------------------------------------------
DATA_GLOB = "Hackathon-Polaron/Batch_*"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs")
MASK_DIR = os.path.join(OUT, "masks")
FIG_DIR = os.path.join(OUT, "figures")
OVL_DIR = os.path.join(OUT, "overlays")
PC_DIR = os.path.join(OUT, "pointcount")
TABLE_DIR = os.path.join(OUT, "tables")
PDF_DIR = os.path.join(OUT, "reports")
XLSX_DIR = os.path.join(OUT, "xlsx")
for d in (OUT, MASK_DIR, FIG_DIR, OVL_DIR, PC_DIR, TABLE_DIR, PDF_DIR, XLSX_DIR):
    os.makedirs(d, exist_ok=True)

BATCHES = ["Batch_1", "Batch_2", "Batch_3"]
BASELINE = "Batch_3"

# fixed colours per batch (colourblind-safe Okabe-Ito) used in ALL figures
BATCH_COLORS = {"Batch_1": "#E69F00", "Batch_2": "#009E73", "Batch_3": "#0072B2"}
# segmentation overlay colours
CLASS_COLORS = {"pore": (0, 0, 0), "graphite": (110, 110, 110),
                "silicon": (255, 80, 0)}
EDGE_CROP_FRAC = 0.05          # crop columns/rows with >5% non-grey pixels

# ---------------------------------------------------------------------------
# Segmentation settings (identical for every image in every batch)
# ---------------------------------------------------------------------------
P_LO, P_HI = 1.0, 99.5          # normalisation percentiles
GAUSS_SIGMA = 1.5               # px, pre-threshold smoothing
HIST_BINS, HIST_SMOOTH = 1024, 3
PORE_GUARD = 60                 # bins: bulk peak search start
BRIGHT_MIN_SEP = 25             # bins: bulk peak -> bright peak min separation
BRIGHT_MIN_PROM = 0.002         # fraction of pixels: bright-peak prominence
OPEN_RADIUS = 2                 # px, binary opening radius (bright mask)
MIN_AREA_PX = 150               # px, minimum bright object
MIN_PORE_PX = 24                # px, minimum pore object
WS_MIN_DIST, WS_MIN_DEPTH = 12, 4  # watershed min peak distance / depth (px)
INLENS_DIL = 2                  # px, Inlens crack refinement reach
SENS_SHIFTS = (-0.10, 0.0, 0.10)   # +/-10% threshold sensitivity shifts


# ---------------------------------------------------------------------------
# Image enumeration and loading
# ---------------------------------------------------------------------------
def find_locations(root=REPO):
    """Sorted list of dicts {batch, image_id, bse, inlens, topo, topo_kind}.
    topo_kind is 'ETD' or 'SE' (some locations ship SE instead of ETD);
    topo is None when neither exists.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(root, DATA_GLOB, "*_BSE.tif"))):
        batch = os.path.basename(os.path.dirname(path))
        iid = os.path.basename(path).split("img_")[1].split("_")[0]
        base = os.path.join(os.path.dirname(path), f"img_{iid}")
        rec = dict(batch=batch, image_id=iid,
                   bse=path,
                   inlens=base + "_Inlens.tif",
                   topo=None, topo_kind=None)
        for kind in ("ETD", "SE"):
            p = base + f"_{kind}.tif"
            if os.path.exists(p):
                rec.update(topo=p, topo_kind=kind)
                break
        if not os.path.exists(rec["inlens"]):
            rec["inlens"] = None
        out.append(rec)
    return out


def pixel_size_um(path):
    """um per pixel from TIFF XResolution/YResolution (pixels per inch)."""
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        xr = page.tags["XResolution"].value
        yr = page.tags["YResolution"].value
        dpi_x, dpi_y = xr[0] / xr[1], yr[0] / yr[1]
        unit = str(page.tags["ResolutionUnit"].value)
    if "INCH" not in unit.upper():
        return np.nan, f"unrecognised ResolutionUnit {unit}"
    return float(25.4e3 / ((dpi_x + dpi_y) / 2.0)), "TIFF XResolution/YResolution"


def load_gray(path):
    """Load as float32 grey. RGB channels are identical apart from edge
    marker stripes, which are cropped away column/row-wise (>5% non-grey).
    Returns (grey, crop_note)."""
    im = tifffile.imread(path)
    if im.ndim == 3:
        d = (np.abs(im[..., 0].astype(np.int16) - im[..., 1]) +
             np.abs(im[..., 0].astype(np.int16) - im[..., 2])) > 4
        g = np.median(im.astype(np.float32), axis=2)
    else:
        d = np.zeros(im.shape, bool)
        g = im.astype(np.float32)
    h, w = g.shape
    col_bad = d.mean(axis=0) > EDGE_CROP_FRAC
    row_bad = d.mean(axis=1) > EDGE_CROP_FRAC
    x0 = int(np.argmax(~col_bad)) if col_bad.any() else 0
    x1 = w - int(np.argmax(~col_bad[::-1])) if col_bad.any() else w
    y0 = int(np.argmax(~row_bad)) if row_bad.any() else 0
    y1 = h - int(np.argmax(~row_bad[::-1])) if row_bad.any() else h
    note = f"cropped [{y0}:{y1}, {x0}:{x1}]" if (x0 or y0 or x1 != w or y1 != h) else "none"
    return g[y0:y1, x0:x1], note


def normalise(g):
    """Percentile-normalised smoothed image s in [0,1]."""
    p1, p995 = np.percentile(g, [P_LO, P_HI])
    n = np.clip((g - p1) / max(p995 - p1, 1e-9), 0, 1)
    return ndi.gaussian_filter(n, GAUSS_SIGMA), p1, p995


def mask_path(batch, iid):
    return os.path.join(MASK_DIR, f"{batch}_{iid}_masks.npz")


def save_masks(batch, iid, **arrays):
    np.savez_compressed(mask_path(batch, iid), **arrays)


def load_masks(batch, iid):
    return np.load(mask_path(batch, iid))


def batch_of(df):
    return df["batch"].isin(BATCHES)


def bootstrap_ci_mean(x, n=2000, alpha=0.05, rng=None):
    """Bootstrap CI on the mean, resampling images."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(axis=1)
    return tuple(np.percentile(m, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def bootstrap_ci_diff(x, y, n=2000, alpha=0.05, rng=None):
    """Bootstrap CI on mean(x)-mean(y)."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan
    mx = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    my = y[rng.integers(0, len(y), (n, len(y)))].mean(axis=1)
    d = mx - my
    return tuple(np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def log(msg):
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
