"""Spatial heterogeneity analysis — the defect-risk signals.

Per image, from the cached 4-class segmentations (frozen recipe):

  si_hotspot_ratio   max/median Si density over a 6x6 patch grid —
                     an "enormous local cluster" detector
  si_maxpatch_frac   share of image where the densest patch sits (Si
                     fraction of densest patch)
  si_topbot_ratio    Si fraction top band / bottom band (frame
                     gradient — foil not in frame, so 'top'/'bottom'
                     are conventions; physical axis unconfirmed)
  pore_topbot_ratio  same for pores
  si_lr_asym         |left-right| Si fraction asymmetry
  largest_si_cluster largest connected Si region / total Si area —
                     a single continuous agglomerate score
  cracklike_frac     pore area share in objects with aspect>4 AND
                     near-frame-vertical axis (crack candidates —
                     frame direction is NOT confirmed through-plane)
  si_near_pore_frac  Si fraction within ~0.15 um of a pore (breathing
                     room hypothesis — Si that can swell into voids)

DIAGNOSTIC/EXPLORATORY layer only — boundary-sensitive, and none
survive within-session blocking (see marker_session_partition.csv).

Outputs: spatial_features.csv + fig_spatial.png + spatial_report.md
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, label as ndi_label
from skimage.measure import regionprops

from vcompare import extract, config

BANDS = 6
GRID = 6
NEAR_PORE_UM = 0.15


def band_frac(mask: np.ndarray) -> np.ndarray:
    h = mask.shape[0]
    edges = np.linspace(0, h, BANDS + 1).astype(int)
    return np.array([mask[edges[i]:edges[i + 1]].mean()
                     for i in range(BANDS)])


def grid_stats(mask: np.ndarray, n: int = GRID) -> tuple[float, float]:
    """(max patch frac / median patch frac, densest patch frac)."""
    h, w = mask.shape
    fr = []
    for i in range(n):
        for j in range(n):
            fr.append(mask[i * h // n:(i + 1) * h // n,
                           j * w // n:(j + 1) * w // n].mean())
    fr = np.asarray(fr)
    med = np.median(fr)
    return float(fr.max() / max(med, 1e-9)), float(fr.max())


def spatial_features(image_id: str, seg: np.ndarray, pixel_um: float,
                     batch: str) -> dict:
    si = seg == config.SI_CAND
    pore = seg == config.PORE
    f = dict(image_id=image_id, batch=batch)

    # depth gradients (row 0 = top of frame)
    sb, pb = band_frac(si), band_frac(pore)
    f["si_topbot_ratio"] = float(sb[0] / max(sb[-1], 1e-9))
    f["pore_topbot_ratio"] = float(pb[0] / max(pb[-1], 1e-9))
    f["si_depth_slope"] = float(np.polyfit(np.arange(BANDS), sb, 1)[0])
    f["pore_depth_slope"] = float(np.polyfit(np.arange(BANDS), pb, 1)[0])

    # hotspot + asymmetry
    f["si_hotspot_ratio"], f["si_maxpatch_frac"] = grid_stats(si)
    h, w = si.shape
    f["si_lr_asym"] = float(abs(si[:, :w // 2].mean()
                                - si[:, w // 2:].mean()))
    f["pore_hotspot_ratio"], _ = grid_stats(pore)

    # largest connected Si agglomerate (4-conn watershed object = particle;
    # use the RAW mask connectivity for the agglomerate measure)
    lab, n = ndi_label(si)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        f["largest_si_cluster_frac"] = float(sizes.max()
                                             / max(si.sum(), 1))
    else:
        f["largest_si_cluster_frac"] = 0.0

    # crack-like pores: elongated (aspect>4) AND near-frame-vertical
    # (>60 deg) — frame-vertical only; physical axis unconfirmed
    plab, _np = ndi_label(pore)
    crack = np.zeros_like(pore)
    for r in regionprops(plab):
        if r.area < 20:
            continue
        asp = r.axis_major_length / max(r.axis_minor_length, 1e-9)
        if asp > 4 and abs(np.degrees(r.orientation)) > 60:
            crack[tuple(r.coords.T)] = True
    f["cracklike_frac"] = float(crack.sum() / max(pore.sum(), 1))

    # Si within ~0.15 um of a pore
    d = max(1, round(NEAR_PORE_UM / pixel_um))
    near = binary_dilation(pore, iterations=d)
    f["si_near_pore_frac"] = float((si & near).sum() / max(si.sum(), 1))
    return f


def run(cache_dirs=("validated_comparison/cache",
                    "new_image_assignment/cache"),
        out="spatial_features.csv"):
    rows = []
    for cd in cache_dirs:
        for c in sorted(glob.glob(os.path.join(cd, "*_seg.npz"))):
            iid = os.path.basename(c).replace("_seg.npz", "")
            meta = json.load(open(c.replace("_seg.npz", "_meta.json")))
            seg = np.load(c)["seg"]
            rows.append(spatial_features(iid, seg, meta["pixel_um"],
                                         meta["batch"]))
            print(f"  {iid} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


def report(df: pd.DataFrame, out_md="spatial_report.md",
           out_fig="fig_spatial.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = [c for c in df.columns if c not in ("image_id", "batch")]
    L = ["# Spatial heterogeneity — the defect-risk signals\n",
         "All values from the frozen vcompare recipe — "
         "**diagnostic/exploratory layer**: these markers depend on "
         "where object boundaries land and none survive acquisition "
         "blocking, so they are not batch markers (see "
         "`marker_session_partition.csv`). `top`/`bottom` are frame "
         "edges; the foil is not in frame and the "
         "direction-to-physical-axis mapping is unconfirmed (the "
         "earlier FFT-based orientation claim was withdrawn — see "
         "`archive/ARCHIVE.md`). Batch medians [min-max]:\n",
         "| signal | Batch_1 | Batch_2 | Batch_3 | New |",
         "|---|---|---|---|---|"]
    med = {}
    for c in cols:
        line = [f"| {c} |"]
        for b in ("Batch_1", "Batch_2", "Batch_3", "New_Images_Batch"):
            v = df[df.batch == b][c]
            med.setdefault(b, {})[c] = v.median()
            line.append(f"{v.median():.3f} [{v.min():.3f},"
                        f"{v.max():.3f}] |" if len(v) else "n/a |")
        L.append(" ".join(line))
    L.append("\n## Reading the signals\n")
    L.append("- `si_hotspot_ratio` high = one patch carries much more Si "
             "than typical -> uneven loading, local swelling stress.")
    L.append("- `largest_si_cluster_frac` = share of Si that is ONE "
             "connected region — the 'enormous continuous cluster' case.")
    L.append("- `si_topbot_ratio` != 1 = Si segregated top-to-bottom "
             "in the frame (drying/calendering hypothesis only — "
             "physical direction unconfirmed).")
    L.append("- `cracklike_frac` = share of pore area in thin "
             "frame-vertical objects — crack-like rather than round "
             "voids. Caveat: streaking can also be a sectioning "
             "artefact, and it is session-sensitive.")
    L.append("- `si_near_pore_frac` = Si within 0.15 µm of a resolved "
             "pore in this 2-D slice — a measured blind spot "
             "(sub-resolution pores dominate real porosity), not a "
             "batch marker.")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, c, t in zip(axes,
                        ("si_hotspot_ratio", "si_topbot_ratio",
                         "cracklike_frac"),
                        ("Si hotspot ratio (max/med patch)",
                         "Si top/bottom depth ratio",
                         "Crack-like pore share")):
        for i, b in enumerate(("Batch_1", "Batch_2", "Batch_3",
                               "New_Images_Batch")):
            v = df[df.batch == b][c].dropna()
            ax.boxplot([v], positions=[i], widths=0.5,
                       showfliers=False)
            ax.scatter(np.full(len(v), i) +
                       np.linspace(-0.1, 0.1, len(v)), v,
                       s=10, alpha=0.7)
            ax.set_xticks(range(4), ["B1", "B2", "B3", "new"])
            ax.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    with open(out_md, "w") as fh:
        fh.write("\n".join(L))
    return out_md, out_fig


if __name__ == "__main__":
    df = run()
    print("\nbatch medians:")
    print(df.groupby("batch").median(numeric_only=True).round(3)
          .to_string())
    report(df)
    print("-> spatial_report.md, fig_spatial.png")
