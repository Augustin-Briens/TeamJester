"""Build per-batch 3-phase training masks with micro2dfn's frozen recipe.

Phase masks are produced by the SAME loader + flat-field + shared
threshold pair as the markers pipeline, then majority-vote downsampled
to the reconstruction voxel size.  Phase definition is therefore
identical to every other deliverable — only the grid spacing changes.
"""
from __future__ import annotations

import io as _io
import json
import os
import time
import zipfile

import numpy as np

from . import config


def batch_masks(root: str, batches: list[str], thresholds: tuple,
                downsample: int = config.DOWNSAMPLE
                ) -> dict[str, dict[str, np.ndarray]]:
    """{batch: {image_id: uint8 label map}} at the reconstruction voxel
    size.  Segmentation runs at full resolution with micro2dfn's frozen
    thresholds; labels are then block-reduced by majority vote (phase
    fractions preserved better than re-segmenting a downsampled image).
    """
    from micro2dfn import io, segment   # local-only dep (TIFF stack)
    t_pore, t_si = thresholds
    out = {}
    for b in batches:
        out[b] = {}
        paths = io.discover_batches(root, [b]).get(b, [])
        for p in paths:
            im = io.load_bse(p, b, downsample=1)
            seg = segment.segment(im, t_pore, t_si)
            out[b][im.image_id] = _reduce_labels(seg, downsample)
    return out


def _reduce_labels(seg: np.ndarray, g: int) -> np.ndarray:
    """Majority-vote downsample of a label map by g x g blocks."""
    if g <= 1:
        return seg
    h, w = seg.shape
    H, W = h // g, w // g
    seg = seg[:H * g, :W * g]
    flat = seg.reshape(H, g, W, g).transpose(0, 2, 1, 3).reshape(H, W, -1)
    counts = np.stack([np.sum(flat == k, axis=-1)
                       for k in range(3)], axis=-1)
    return counts.argmax(axis=-1).astype(np.uint8)


def ruler_tag(prov: dict) -> str:
    """Short hash identifying the measuring ruler — stamped into mask
    caches and volume filenames so ensembles built under different
    rulers can never be silently mixed."""
    import hashlib
    key = f"{prov['t_pore']:.6f}|{prov['t_si']:.6f}|{prov['fitted_on']}"
    return hashlib.md5(key.encode()).hexdigest()[:6]


def load_or_fit_recipe(root: str, out_dir: str,
                       baseline: str = "Batch_3") -> tuple[float, float,
                                                           dict]:
    """Frozen threshold pair under reference-only calibration.

    Priority: micro2dfn's baseline-fitted recipe.json if it exists
    (identical ruler as run_dfn.py), else recon3d's own saved recipe,
    else fit thresholds on the BASELINE batch alone and save.  The
    pooled-all-images thresholds.json from the exploratory era is
    deliberately NOT reused — the ruler must not learn from the
    incoming batches.

    Returns (t_pore, t_si, provenance-dict)."""
    from micro2dfn import io, segment   # local-only dep
    candidates = [
        os.path.join("dfn_output", "recipe.json"),
        os.path.join(out_dir, "recipe.json"),
    ]
    for c in candidates:
        if c and os.path.exists(c):
            with open(c) as fh:
                rec = json.load(fh)
            prov = dict(t_pore=float(rec["t_pore"]),
                        t_si=float(rec["t_si"]),
                        fitted_on=rec.get("fitted_on", "?"),
                        n_fit_images=rec.get("n_fit_images"),
                        source=c,
                        created_utc=rec.get("created_utc"))
            return prov["t_pore"], prov["t_si"], prov

    paths = io.discover_batches(root, [baseline]).get(baseline, [])
    if not paths:
        raise FileNotFoundError(
            f"baseline batch {baseline} not found under {root} — "
            "cannot fit the reference ruler")
    images = [io.load_bse(p, baseline, downsample=1) for p in paths]
    t_pore, t_si = segment.fit_thresholds(images)
    rec = {"recipe_version": "recon3d-1.1",
           "calibration": "reference-only",
           "fitted_on": baseline,
           "n_fit_images": len(images),
           "t_pore": float(t_pore), "t_si": float(t_si),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "recipe.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    prov = dict(rec, source=os.path.join(out_dir, "recipe.json"))
    return t_pore, t_si, prov


# ---------------------------------------------------------------------------
# npz serialisation for shipping masks to Modal workers
# ---------------------------------------------------------------------------
def pack_masks(masks: dict[str, dict[str, np.ndarray]]) -> bytes:
    buf = _io.BytesIO()
    flat = {}
    for b, imgs in masks.items():
        for iid, m in imgs.items():
            flat[f"{b}::{iid}"] = m
    np.savez_compressed(buf, **flat)
    return buf.getvalue()


def unpack_masks(blob: bytes) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with zipfile.ZipFile(_io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            key = name[:-4] if name.endswith(".npy") else name
            with zf.open(name) as fh:
                arr = np.load(fh)
            b, iid = key.split("::", 1)
            out.setdefault(b, {})[iid] = arr
    return out


def save_manifest(masks: dict[str, dict[str, np.ndarray]],
                  path: str, prov: dict | None = None) -> None:
    man = {"ruler": prov or {}}
    man["batches"] = {
        b: {iid: list(map(int, m.shape)) for iid, m in imgs.items()}
        for b, imgs in masks.items()}
    man["phase_frac_mean"] = {
        b: {k: float(np.mean([(m == ph).mean()
                              for m in imgs.values()]))
            for k, ph in (("pore", 0), ("bulk", 1), ("bright", 2))}
        for b, imgs in masks.items()}
    with open(path, "w") as fh:
        json.dump(man, fh, indent=1)
