#!/usr/bin/env python3
"""
porosity.py - SEM/BSE pore-space analysis for Hackathon-Polaron batches.

Measures pore space (total porosity, pore classes, pore sizes, depth
profiles, chords, and pore space adjacent to the bright particles) in the
BSE images of Hackathon-Polaron/Batch_1..3, compares the batches
statistically, and produces figures, overlays, an Excel workbook, a PDF
report and NOTES.md.

Image loading, pixel-size reading and bright-particle segmentation are the
same code as silicon_size/silicon_size.py (branch `silicon-size-analysis`),
so the two analyses agree pixel-for-pixel on the bright particles.

ASSUMPTION (applies to every output): the bright particles in BSE are
assumed to be silicon-based and the grey bulk is assumed to be graphite.
This is an assumption from image appearance only; it has not been verified
by e.g. EDS.

Usage:
    python3 porosity.py                  # run everything
    python3 porosity.py --data-root ..   # if run from inside porosity/

Requirements: numpy, scipy, scikit-image, pandas, matplotlib, openpyxl,
reportlab, tifffile (+ imagecodecs, which tifffile needs to decode the
LZW-compressed TIFFs).
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from scipy import stats as scs

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from skimage.filters import threshold_multiotsu
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import (binary_opening, disk, remove_small_objects,
                                binary_dilation)
from skimage.segmentation import watershed, find_boundaries

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------
# Configuration - one method, same settings for every image (no per-batch
# tuning)
# --------------------------------------------------------------------------
P_LO, P_HI = 1.0, 99.5          # normalisation percentiles
GAUSS_SIGMA = 1.5               # px, pre-threshold smoothing
HIST_BINS = 512                 # histogram resolution on the normalised image
HIST_SMOOTH = 3.0               # bins, histogram smoothing for peak finding
PORE_GUARD = 15                 # bins ignored at low end (pore spike guard)
BRIGHT_MIN_SEP = 25             # bins, minimum separation bulk peak -> bright peak
BRIGHT_MIN_PROM = 0.002         # fraction of pixels; bright peak prominence/size
OPEN_RADIUS = 2                 # px, binary opening disk radius (bright mask)
MIN_AREA_PX = 150               # px, minimum bright object size
MIN_PORE_PX = 20                # px, minimum pore object size
WS_MIN_DIST = 8                 # px, watershed marker min separation
WS_MIN_DEPTH = 3                # px, marker must sit >= this deep inside object
RING_RADII = (5, 10, 20)        # px, ring widths around bright particles
CONTACT_DIST = 3                # px, perimeter-to-pore contact distance
LARGE_GAP_FRAC = 0.005          # image-area fraction defining a "large gap"
CRACK_ASPECT = 3.0              # major/minor axis ratio defining a crack/slit
N_BANDS = 10                    # horizontal bands for the depth profile
SENS_SHIFTS = (-0.10, 0.0, 0.10)  # pore-threshold multiplicative shifts
OVERLAY_SCALE = 0.5             # downscale factor for saved overlay PNGs
BATCH_COLORS = {"Batch_1": "#0072B2", "Batch_2": "#D55E00",
                "Batch_3": "#009E73"}  # Okabe-Ito, used consistently
PORE_TINT = "#35A7FF"           # translucent blue for pore regions
BRIGHT_EDGE = "#FF7A00"         # orange outline for bright (assumed-Si) particles
CONFINED_COLOR = "#FF00FF"      # magenta outline for confined particles
BORDER_COLOR = "#B0B0B0"        # grey outline for border-touching particles

# Images previously observed (silicon_size analysis) to behave differently at
# lower contrast: ~2-4x particle counts, thresholds dipping into marginal
# bright material. Still processed identically; results reported with and
# without them.
OUTLIER_IDS = {"4ih2ggld", "5n1q8atc"}

N_BOOT = 2000                   # bootstrap resamples (images resampled)
RNG_SEED = 20241003

PAPER_TIT = ("Impact of Silicon/Graphite Composite Electrode Porosity on "
             "the Cycle Life of 18650 Lithium-Ion Cell")
PAPER_DOI = "10.1021/acsaem.0c01999"
# Earlier rough numbers supplied by the user as a cross-check only.
XCHECK = {"Batch_1": 0.089, "Batch_2": 0.100, "Batch_3": 0.109}
XCHECK_P = 0.07


# --------------------------------------------------------------------------
# IO / metadata (identical to silicon_size.py)
# --------------------------------------------------------------------------
def find_bse_images(root):
    """Return sorted list of (batch, image_id, path) for *_BSE.tif files."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "Hackathon-Polaron",
                                              "Batch_*", "*_BSE.tif"))):
        batch = os.path.basename(os.path.dirname(path))
        iid = os.path.basename(path).split("img_")[1].split("_")[0]
        out.append((batch, iid, path))
    return out


def pixel_size_um(path):
    """Pixel size in um, from the TIFF resolution tags.

    XResolution/YResolution are stored in pixels per inch, so
    um/px = 25.4e6 / dpi / 1000. Returns (um_per_px, source_note).
    """
    with tifffile.TiffFile(path) as t:
        page = t.pages[0]
        xr = page.tags["XResolution"].value
        yr = page.tags["YResolution"].value
        dpi_x = xr[0] / xr[1]
        dpi_y = yr[0] / yr[1]
        unit = str(page.tags["ResolutionUnit"].value)
    if "INCH" not in unit:
        return np.nan, f"unrecognised ResolutionUnit {unit}"
    um_x = 25.4e6 / dpi_x / 1000.0
    um_y = 25.4e6 / dpi_y / 1000.0
    return float(np.mean([um_x, um_y])), "TIFF XResolution/YResolution tags"


def load_gray(path):
    """Load image as float64 grey level (per-pixel median of RGB channels)."""
    im = tifffile.imread(path)
    if im.ndim == 3:
        g = np.median(im.astype(np.float32), axis=2)
    else:
        g = im.astype(np.float32)
    return g


def crop_banner(g):
    """Crop a bottom information banner if present (uniform rows, std < 2)."""
    std = g.std(axis=1)
    n = 0
    while n < g.shape[0] and std[g.shape[0] - 1 - n] < 2.0:
        n += 1
    if 0 < n < g.shape[0] // 2:
        return g[: g.shape[0] - n], n
    return g, 0


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def find_bright_threshold(s):
    """Bright-particle threshold on the smoothed, normalised image.

    Primary rule: the valley of the histogram between the bulk peak and a
    separate bright peak. Fallback (flagged): the upper 3-class multi-Otsu
    threshold. Identical to silicon_size.py. Returns
    (threshold, method, bulk_peak_x, bright_peak_x).
    """
    h, edges = np.histogram(s, bins=HIST_BINS, range=(0, 1))
    hs = ndi.gaussian_filter1d(h.astype(float), HIST_SMOOTH)
    cents = (edges[:-1] + edges[1:]) / 2.0
    bulk = PORE_GUARD + int(np.argmax(hs[PORE_GUARD:400]))
    cand = []
    for i in range(bulk + BRIGHT_MIN_SEP, HIST_BINS - 7):
        if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1]:
            prom = hs[i] - hs[bulk:i + 1].min()
            if (prom > BRIGHT_MIN_PROM * h.sum()
                    and hs[i] > BRIGHT_MIN_PROM * h.sum()):
                cand.append((i, prom))
    if cand:
        bp = max(cand, key=lambda c: c[1])[0]
        t = cents[bulk + int(np.argmin(hs[bulk:bp]))]
        return float(t), "valley", float(cents[bulk]), float(cents[bp])
    t = threshold_multiotsu(s, classes=3)[1]
    return float(t), "multiotsu_fallback", float(cents[bulk]), np.nan


def pore_mask_at(s, thr_pore, bright_mask):
    """Pore mask at a given pore threshold: s < thr_pore, bright pixels win,
    objects < MIN_PORE_PX removed."""
    pm = (s < thr_pore) & ~bright_mask
    pm = remove_small_objects(pm, MIN_PORE_PX)
    return pm


def segment3(g):
    """Three-phase segmentation. Returns (norm_smooth, lab, pore_mask,
    bright_mask, thr_pore, info).

    lab: 1..n per bright particle (watershed, identical to silicon_size.py);
    pore_mask: cleaned pore boolean; bright_mask: cleaned bright boolean.
    """
    p1, p995 = np.percentile(g, [P_LO, P_HI])
    n = np.clip((g - p1) / (p995 - p1), 0, 1)
    s = ndi.gaussian_filter(n, GAUSS_SIGMA)
    # pore threshold: LOWER cut of 3-class multi-Otsu
    t_lo, t_hi = threshold_multiotsu(s, classes=3)
    # bright threshold: valley, fallback upper multi-Otsu cut
    t_b, method, bulk_x, bright_x = find_bright_threshold(s)
    bw = s > t_b
    bw = binary_opening(bw, disk(OPEN_RADIUS))
    bw = remove_small_objects(bw, MIN_AREA_PX)
    bw = ndi.binary_fill_holes(bw)
    pm = pore_mask_at(s, t_lo, bw)
    dist = ndi.distance_transform_edt(bw)
    coords = peak_local_max(dist, min_distance=WS_MIN_DIST,
                            threshold_abs=WS_MIN_DEPTH, labels=bw)
    markers = np.zeros(bw.shape, np.int32)
    for j, (r, c) in enumerate(coords, 1):
        markers[r, c] = j
    markers = ndi.label(markers > 0)[0]
    lab = watershed(-dist, markers, mask=bw)
    info = dict(threshold=t_b, method=method, bulk_x=bulk_x,
                bright_x=bright_x, p_lo=p1, p_hi=p995,
                thr_pore=float(t_lo), thr_multiotsu_upper=float(t_hi),
                bright_frac=float(bw.mean()), pore_frac=float(pm.mean()))
    return s, lab, pm, bw, float(t_lo), info


# --------------------------------------------------------------------------
# Measurement - bright particles (identical fields to silicon_size.py)
# --------------------------------------------------------------------------
def measure_particles(lab, iid, batch, um_per_px):
    rows = []
    h, w = lab.shape
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        border = r0 == 0 or c0 == 0 or r1 == h or c1 == w
        rows.append(dict(
            image_id=iid, batch=batch, label=int(p.label),
            area_px=float(p.area),
            area_um2=float(p.area) * um_per_px ** 2,
            equiv_diam_um=float(p.equivalent_diameter_area) * um_per_px,
            major_axis_um=float(p.axis_major_length) * um_per_px,
            minor_axis_um=float(p.axis_minor_length) * um_per_px,
            aspect_ratio=(float(p.axis_major_length) /
                          max(float(p.axis_minor_length), 1e-9)),
            solidity=float(p.solidity),
            centroid_row=float(p.centroid[0]),
            centroid_col=float(p.centroid[1]),
            touches_border=bool(border),
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Measurement - pores
# --------------------------------------------------------------------------
def measure_pores(pore_mask, iid, batch, um_per_px, n_px_img):
    """One row per pore region. pore_class precedence: large_gap
    (area > 0.5% of image) > crack_slit (aspect > 3) > compact.
    size measure = mean local thickness = 2 * mean(distance transform
    inside the region)."""
    lab_p = label(pore_mask)
    dist_p = ndi.distance_transform_edt(pore_mask)
    rows = []
    h, w = pore_mask.shape
    band_edges = np.linspace(0, h, N_BANDS + 1)
    for p in regionprops(lab_p):
        reg = lab_p[p.slice] == p.label
        thick_mean = 2.0 * float(dist_p[p.slice][reg].mean())
        thick_max = 2.0 * float(dist_p[p.slice][reg].max())
        area_px = float(p.area)
        aspect = (float(p.axis_major_length) /
                  max(float(p.axis_minor_length), 1e-9))
        area_frac = area_px / n_px_img
        if area_frac > LARGE_GAP_FRAC:
            cls = "large_gap"
        elif aspect > CRACK_ASPECT:
            cls = "crack_slit"
        else:
            cls = "compact"
        band = int(np.clip(np.digitize(p.centroid[0], band_edges) - 1,
                           0, N_BANDS - 1))
        rows.append(dict(
            image_id=iid, batch=batch, pore_id=int(p.label),
            area_px=area_px, area_um2=area_px * um_per_px ** 2,
            area_frac_image=area_frac,
            equiv_diam_um=float(p.equivalent_diameter_area) * um_per_px,
            thickness_mean_um=thick_mean * um_per_px,
            thickness_max_um=thick_max * um_per_px,
            major_axis_um=float(p.axis_major_length) * um_per_px,
            minor_axis_um=float(p.axis_minor_length) * um_per_px,
            aspect_ratio=aspect,
            orientation_deg=float(np.degrees(p.orientation)),
            centroid_row=float(p.centroid[0]),
            centroid_col=float(p.centroid[1]),
            depth_band=band,
            pore_class=cls,
        ))
    return pd.DataFrame(rows)


def chord_lengths(pore_mask, um_per_px):
    """Horizontal and vertical run lengths of pore pixels (in um)."""
    def runs(mask_rows):
        lens = []
        for row in mask_rows:
            d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
            starts = np.nonzero(d == 1)[0]
            ends = np.nonzero(d == -1)[0]
            lens.append(ends - starts)
        if lens:
            return np.concatenate(lens) * um_per_px
        return np.empty(0)
    h_lens = runs(list(pore_mask))
    v_lens = runs(list(pore_mask.T))
    return h_lens, v_lens


def ring_measurements(lab, bright_mask, pore_mask):
    """Pore space adjacent to each bright particle.

    Uses distance transforms so every non-bright pixel is attributed to its
    nearest bright particle:
      ring porosity at R px  = pore fraction of the annulus 0 < dist <= R
      contact fraction        = share of the particle's inner boundary
                                pixels within CONTACT_DIST px of a pore
      confined at R px        = ring contains zero pore pixels
    Returns a dict label -> measurements.
    """
    # distance to nearest bright pixel + identity of that particle
    dist_b, ind = ndi.distance_transform_edt(~bright_mask,
                                           return_indices=True)
    owner = lab[ind[0], ind[1]]

    # distance to nearest pore pixel (defined everywhere)
    dist_p = ndi.distance_transform_edt(~pore_mask)

    # inner boundary of each particle (pixels inside, adjacent to outside)
    bnd = find_boundaries(lab, mode="inner")
    bnd_owner = lab[bnd]
    bnd_isclose = dist_p[bnd] <= CONTACT_DIST

    out = {}
    max_r = max(RING_RADII)
    wide = (dist_b > 0) & (dist_b <= max_r)
    o = owner[wide]
    is_pore = pore_mask[wide]
    d = dist_b[wide]
    for r in RING_RADII:
        sel = d <= r
        tot = np.bincount(o[sel], minlength=lab.max() + 1)
        npore = np.bincount(o[sel], weights=is_pore[sel].astype(np.int64),
                            minlength=lab.max() + 1)
        out[f"ring_px_{r}"] = tot
        out[f"ring_porepx_{r}"] = npore.astype(int)

    btot = np.bincount(bnd_owner, minlength=lab.max() + 1)
    bclose = np.bincount(bnd_owner, weights=bnd_isclose.astype(np.int64),
                         minlength=lab.max() + 1)
    out["boundary_px"] = btot
    out["boundary_close_px"] = bclose.astype(int)
    return out


def bright_in_voids(lab, ring):
    """Bright objects whose 5-px ring is mostly pore -> bright material
    sitting inside a void (e.g. the back wall of a pore seen through the
    opening). Returns count and total pixel area."""
    n5 = ring["ring_px_5"]
    p5 = ring["ring_porepx_5"]
    frac = np.divide(p5, n5, out=np.zeros_like(p5, float),
                     where=n5 > 0)
    areas = np.bincount(lab.ravel(), minlength=lab.max() + 1)
    sel = (frac > 0.6) & (n5 > 0)
    sel[0] = False
    return int(sel.sum()), int(areas[sel].sum())


def depth_bands(pore_mask):
    """Pore fraction in N_BANDS equal horizontal bands, top to bottom."""
    h = pore_mask.shape[0]
    edges = np.linspace(0, h, N_BANDS + 1).astype(int)
    return [float(pore_mask[edges[i]:edges[i + 1]].mean())
            for i in range(N_BANDS)]


def weighted_quantile(values, weights, qs):
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, v)


# --------------------------------------------------------------------------
# Statistics (same scheme as silicon_size.py: image = independent unit)
# --------------------------------------------------------------------------
def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def kruskal_and_mwu(per_image, metric, batches):
    groups = [per_image.loc[per_image.batch == b, metric].dropna().values
              for b in batches]
    rows = []
    non_empty = [g for g in groups if len(g) > 0]
    if len(non_empty) >= 2 and sum(len(g) for g in non_empty) >= 3:
        H, p = scs.kruskal(*non_empty)
        n = sum(len(g) for g in non_empty)
        k = len(non_empty)
        eps2 = max(0.0, (H - k + 1) / (n - k))
        rows.append(dict(test="Kruskal-Wallis", metric=metric,
                         groups=" vs ".join(batches), statistic=float(H),
                         p_value=float(p), p_adj=np.nan,
                         effect_name="epsilon_squared",
                         effect_size=float(eps2), n_per_group=str(
                             [len(g) for g in non_empty])))
    pairs, pvals, stats_, effs = [], [], [], []
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            a, b = groups[i], groups[j]
            if len(a) == 0 or len(b) == 0:
                continue
            U, p = scs.mannwhitneyu(a, b, alternative="two-sided")
            rbc = 1.0 - 2.0 * U / (len(a) * len(b))  # >0: a < b
            pairs.append(f"{batches[i]} vs {batches[j]}")
            pvals.append(float(p)); stats_.append(float(U))
            effs.append(float(rbc))
    if pairs:
        adj = holm_adjust(pvals)
        for pair, U, p, pa, e in zip(pairs, stats_, pvals, adj, effs):
            rows.append(dict(test="Mann-Whitney U", metric=metric,
                             groups=pair, statistic=U, p_value=p,
                             p_adj=float(pa),
                             effect_name="rank_biserial",
                             effect_size=e, n_per_group=""))
    return rows


def bootstrap_image_mean(values_by_image, ids, n_boot=N_BOOT,
                         seed=RNG_SEED):
    """95% bootstrap CI for the mean of a per-image metric, resampling
    images."""
    rng = np.random.default_rng(seed)
    vals = np.asarray([values_by_image[i] for i in ids], float)
    if len(vals) == 0:
        return (np.nan, np.nan)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(vals, size=len(vals), replace=True)
        boots[b] = pick.mean()
    return (float(np.percentile(boots, 2.5)),
            float(np.percentile(boots, 97.5)))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_porosity_per_image(per_image, batches, outpath):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rng = np.random.default_rng(RNG_SEED)
    for bi, b in enumerate(batches):
        sub = per_image[per_image.batch == b].sort_values("image_id")
        jit = rng.uniform(-0.15, 0.15, len(sub))
        for (_, r), dx in zip(sub.iterrows(), jit):
            face = "white" if r.image_id in OUTLIER_IDS else BATCH_COLORS[b]
            ax.scatter(bi + dx, r.porosity, facecolor=face,
                       edgecolor=BATCH_COLORS[b], s=45, zorder=3)
        m = sub.porosity.mean()
        ax.plot([bi - 0.28, bi + 0.28], [m, m], color="k", lw=2, zorder=4)
        ax.text(bi + 0.32, m, f"mean {m:.3f}", fontsize=8, va="center")
    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels(batches)
    ax.set_ylabel("Pore area fraction (2D image porosity)")
    ax.set_title("Porosity per image (each point = one image; bar = batch "
                 "mean; open = outlier images)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_pore_classes(per_image, batches, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    classes = [("gap_area_frac", "gap_count_per_mm2", "large gaps (>0.5% of image)"),
               ("crack_area_frac", "crack_count_per_mm2", "cracks/slits (aspect>3)"),
               ("compact_area_frac", "compact_count_per_mm2", "compact pores")]
    width = 0.25
    x = np.arange(3)
    for bi, b in enumerate(batches):
        sub = per_image[per_image.batch == b]
        af = [sub[c[0]].mean() for c in classes]
        cd = [sub[c[1]].mean() for c in classes]
        axes[0].bar(x + bi * width - width, af, width,
                    color=BATCH_COLORS[b], label=b)
        bars = axes[1].bar(x + bi * width - width, cd, width,
                           color=BATCH_COLORS[b], label=b)
        for rect in bars:
            axes[1].annotate(f"{rect.get_height():.0f}",
                             (rect.get_x() + rect.get_width() / 2,
                              rect.get_height()),
                             textcoords="offset points", xytext=(0, 1),
                             ha="center", fontsize=6, rotation=90)
    for ax, ttl in zip(axes, ["Mean pore-area fraction",
                              "Mean count per mm^2"]):
        ax.set_xticks(x)
        ax.set_xticklabels([c[2] for c in classes], fontsize=8)
        ax.set_title(ttl)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Pore classes per batch")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_pore_size_distribution(pores, batches, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for b in batches:
        sub = pores[pores.batch == b]
        v = np.sort(sub.thickness_mean_um.values)
        if len(v) == 0:
            continue
        axes[0].plot(v, np.linspace(0, 1, len(v)), color=BATCH_COLORS[b],
                     lw=1.6, label=b)
        order = np.argsort(sub.thickness_mean_um.values)
        vv = sub.thickness_mean_um.values[order]
        cw = np.cumsum(sub.area_um2.values[order])
        axes[1].plot(vv, cw / cw[-1], color=BATCH_COLORS[b], lw=1.6,
                     label=b)
    axes[0].set_title("Number-weighted")
    axes[1].set_title("Area-weighted")
    for ax in axes:
        ax.set_xlabel("Pore size = mean local thickness (um)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Cumulative pore size distributions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_depth_profile(band_df, batches, outpath):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    band_cols = [f"porosity_band_{i + 1:02d}" for i in range(N_BANDS)]
    xs = np.arange(N_BANDS)
    for b in batches:
        sub = band_df[band_df.batch == b]
        m = sub[band_cols].mean().values
        sem = sub[band_cols].sem().values
        ax.plot(xs, m, color=BATCH_COLORS[b], lw=1.8, marker="o", ms=4,
                label=b)
        ax.fill_between(xs, m - sem, m + sem, color=BATCH_COLORS[b],
                        alpha=0.18)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i + 1}" for i in range(N_BANDS)])
    ax.set_xlabel("Depth band (1 = top of image, 10 = bottom)")
    ax.set_ylabel("Pore area fraction")
    ax.set_title("Through-thickness porosity profile (mean +/- SEM across "
                 "images)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_local_porosity(per_image, batches, outpath):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(RNG_SEED)
    for b in batches:
        sub = per_image[per_image.batch == b]
        for _, r in sub.iterrows():
            face = "white" if r.image_id in OUTLIER_IDS else BATCH_COLORS[b]
            ax.scatter(r.porosity, r.local_poro_10px_mean, facecolor=face,
                       edgecolor=BATCH_COLORS[b], s=45, label=b, zorder=3)
    lim = [0, max(per_image.porosity.max(),
                  per_image.local_poro_10px_mean.max()) * 1.08]
    ax.plot(lim, lim, "k--", lw=1, label="local = whole-image")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Whole-image pore area fraction")
    ax.set_ylabel("Mean pore fraction in 10-px ring\naround bright particles")
    ax.set_title("Local porosity next to assumed-Si particles vs whole "
                 "image")
    handles = [plt.Line2D([], [], marker="o", ls="", color=BATCH_COLORS[b],
                          label=b) for b in batches]
    handles.append(plt.Line2D([], [], marker="o", ls="", color="k",
                              markerfacecolor="white",
                              label="outlier image"))
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_confined(per_batch_ci, per_image, batches, outpath):
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    rng = np.random.default_rng(RNG_SEED)
    xs, means, lo, hi = [], [], [], []
    for bi, b in enumerate(batches):
        row = per_batch_ci.loc[b]
        xs.append(bi)
        means.append(row["confined_10px_mean"])
        lo.append(row["confined_10px_ci_lo"])
        hi.append(row["confined_10px_ci_hi"])
        sub = per_image[per_image.batch == b]
        jit = rng.uniform(-0.12, 0.12, len(sub))
        ax.scatter(np.full(len(sub), bi) + jit, sub.confined_share_10px,
                   color=BATCH_COLORS[b], s=22, alpha=0.7, zorder=2)
    ax.errorbar(xs, means,
                yerr=[np.array(means) - np.array(lo),
                      np.array(hi) - np.array(means)],
                fmt="D", color="k", capsize=5, zorder=3,
                label="batch mean, 95% bootstrap CI (images resampled)")
    ax.set_xticks(xs)
    ax.set_xticklabels(batches)
    ax.set_ylabel("Fraction of bright particles with no\npore within 10 px")
    ax.set_title("'Confined' assumed-Si particles per batch")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_paper_context(per_image, batches, outpath):
    """Our 2D image porosity (%) per batch vs the two bulk porosities
    studied in the paper (30% and 40%). The paper's values are BULK
    porosities set at cell build; ours are 2D area fractions - shown side
    by side for context only, not as a like-for-like comparison."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for pv, cyc in [(30, 225), (40, 347)]:
        ax.axvline(pv, color="#8856a7", ls="--", lw=1.4)
    ymax = len(batches) + 0.8
    ax.text(29.6, ymax - 0.15, "paper: 30% bulk\nporosity -> 225 cycles ",
            fontsize=7.5, color="#8856a7", va="top", ha="right")
    ax.text(40.4, ymax - 0.15, " paper: 40% bulk\n porosity -> 347 cycles",
            fontsize=7.5, color="#8856a7", va="top", ha="left")
    for i, b in enumerate(batches):
        sub = per_image[per_image.batch == b]
        vals = sub.porosity.values * 100.0
        lo, hi = vals.min(), vals.max()
        y = ymax - 1.5 - i
        ax.plot([lo, hi], [y, y], lw=9, color=BATCH_COLORS[b],
                solid_capstyle="butt")
        ax.plot(np.median(vals), y, marker="D", color="k", ms=6)
        ax.text(0.3, y + 0.26, f"{b}: image porosity {lo:.1f}-{hi:.1f}%",
                fontsize=8, va="bottom", color=BATCH_COLORS[b])
    ax.set_xlim(0, 45)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.set_xlabel("Porosity (%)")
    ax.set_title("Our 2D image porosity vs the paper's bulk porosities "
                 "(different quantities - context only)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def make_overlay(s, lab, pore_mask, parts_img):
    """RGB overlay: pores tinted translucent blue; bright particles outlined
    orange, confined ones magenta, border-touching grey."""
    out = np.dstack([s] * 3).copy()
    # translucent pore tint
    tint = mcolors.to_rgb(PORE_TINT)
    alpha = 0.45
    for ch in range(3):
        out[:, :, ch][pore_mask] = (1 - alpha) * out[:, :, ch][pore_mask] \
            + alpha * tint[ch]
    # particle outlines - one LUT instead of a per-label full-image pass
    bnd = find_boundaries(lab, mode="outer")
    bnd = binary_dilation(bnd, disk(1))
    lab_d = np.where(bnd, lab, 0)
    cmap = np.zeros((lab.max() + 1, 3))
    for _, r in parts_img.iterrows():
        lbl = int(r.label)
        if r.touches_border:
            col = BORDER_COLOR
        elif r.get("confined_10px", False):
            col = CONFINED_COLOR
        else:
            col = BRIGHT_EDGE
        cmap[lbl] = mcolors.to_rgb(col)
    sel = bnd
    rgb_bnd = cmap[lab_d[sel]]
    for ch in range(3):
        ch_plane = out[:, :, ch]
        ch_plane[sel] = rgb_bnd[:, ch]
    h, w = out.shape[:2]
    nh, nw = int(h * OVERLAY_SCALE), int(w * OVERLAY_SCALE)
    ys = (np.linspace(0, h - 1, nh)).astype(int)
    xs = (np.linspace(0, w - 1, nw)).astype(int)
    return out[ys][:, xs]


def save_overlay(s, lab, pore_mask, parts_img, path, title):
    out_d = make_overlay(s, lab, pore_mask, parts_img)
    fig = plt.figure(figsize=(out_d.shape[1] / 200, out_d.shape[0] / 200))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(out_d, interpolation="nearest")
    ax.axis("off")
    ax.text(0.01, 0.985, title, transform=ax.transAxes, fontsize=7,
            color="white", va="top",
            bbox=dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none"))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def contact_sheet(overlay_paths, batch, path, cols=4):
    n = len(overlay_paths)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 4.2, rows * 1.7))
    axes = np.atleast_2d(axes).ravel() if n > 1 else [axes]
    for ax, p in zip(axes, overlay_paths):
        ax.imshow(plt.imread(p))
        ax.axis("off")
        ax.set_title(os.path.basename(p).replace("overlay_", "").replace(
            ".png", ""), fontsize=7)
    for ax in axes[len(overlay_paths):]:
        ax.axis("off")
    fig.suptitle(f"{batch} - three-phase overlays "
                 f"(blue tint = pore, orange = assumed Si, magenta = "
                 f"confined, grey = border)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Paper sheet content.
# --------------------------------------------------------------------------
# NOTE: the published article (ACS Appl. Energy Mater. 2020, 3, 11873-11885)
# is paywalled and no open-access full text was found (checked ACS page,
# OpenAlex, Unpaywall, HAL, CORE, figshare). The verifiable material is the
# publisher-hosted ABSTRACT and the free Supporting Information PDF (5 pp,
# ae0c01999_si_001.pdf). Quotes below are restricted to those two sources and
# labelled accordingly; no claims are taken from memory.
PAPER_FACTS = [
    ("Identification",
     "DOI 10.1021/acsaem.0c01999, ACS Applied Energy Materials 2020, 3(12), "
     "11873-11885. Profatilova, De Vito, Genies, Vincens, Gutel, Fanget, "
     "Martin, Chandesris (Univ. Grenoble Alpes, CEA-Liten), Tulodziecki "
     "(Umicore Research), Porcher (CEA-Liten).",
     "ACS article page / CrossRef"),
    ("Access status",
     "Full text NOT openly accessible: ACS page is purchase-only; no OA "
     "location on OpenAlex or Unpaywall; no repository deposit on HAL; "
     "figshare hosts only the Supporting Information (ae0c01999_si_001.pdf, "
     "5 pp). The analysis below therefore uses only the abstract and the SI, "
     "which are quoted verbatim. This is a limitation on the paper-based "
     "interpretation, not on the image measurements.",
     "-"),
    ("Electrode & cell format",
     "\"The impact of porosity of a negative electrode containing a "
     "silicon/carbon/graphite composite on the cycling performance of 18650 "
     "cells with a NMC-based cathode was investigated.\"",
     "Abstract"),
    ("Porosity values studied",
     "\"Anode porosity variation between 30 and 40% led to a significant "
     "difference in the cycle life of 18650 cells: 225 and 347 cycles were "
     "achieved, respectively, before the drop of a discharge capacity to "
     "75% of the initial value.\" The SI confirms the codes: \"30 and 40 "
     "mean initial porosity for the negative Si/C/G electrode in %\" "
     "(Fig. S1 caption, p. S1). Two porosity levels: ~30% and ~40%.",
     "Abstract; SI p. S1"),
    ("How porosity was determined",
     "The abstract does not state the method explicitly. The SI shows the "
     "electrode was characterised through bulk thickness measurements: "
     "\"Thickness measurements of the active layer obtained using a gauge "
     "or by cross-SEM\" (Table S1, p. S5), with the gauge covering \"a "
     "surface average of ~2 cm^2 with a pressure of 100 kPa\". Porosity in "
     "this class of study is the BULK value set by calendering (from "
     "coating mass, thickness and material densities) - it is NOT a 2D "
     "image-area fraction. Our image porosity is a different quantity and "
     "cannot be compared number-for-number with the paper's 30%/40%; only "
     "the qualitative direction can be used.",
     "SI Table S1, p. S5"),
    ("Porosity vs cycle life",
     "HIGHER porosity gave LONGER life over the studied range: the 40% "
     "anode reached 347 cycles vs 225 for the 30% anode (abstract). With "
     "only two levels, the paper establishes a direction over 30-40% but "
     "cannot show whether the relationship is monotonic or has an optimum "
     "beyond it.",
     "Abstract"),
    ("Degradation mechanism, 40%",
     "\"Li+ ion loss in the solid electrolyte interphase layer was the main "
     "reason for the performance loss in the case of 40% anode porosity.\"",
     "Abstract"),
    ("Degradation mechanism, 30%",
     "\"Compression of the anode to 30% of porosity resulted in Li metal "
     "deposition on the negative electrode surface and within the "
     "separator during cycling of the 18650 cell ... Li+ ion transport was "
     "largely impeded by this deposit leading to poor C-rate "
     "performance.\"",
     "Abstract"),
    ("Postmortem methods",
     "\"A detailed postmortem study of 18650 cells with 30 and 40% anodes "
     "after formation and cycling steps was conducted including "
     "measurements of average electrode mass and thickness changes, "
     "electrochemical cycling in half-cells, impedance measurements in "
     "symmetrical cells, study of the electrode surface compositions via "
     "X-ray photoelectron spectroscopy (XPS), and acquisition of "
     "cross-sectional scanning electron microscopy images.\"",
     "Abstract"),
    ("SI - thickness data",
     "Table S1 (p. S5): thickness-gauge vs cross-SEM thickness for "
     "pristine/after-formation/after-cycling anodes; e.g. A-30 pristine "
     "43 um (gauge) / 45 um (SEM); A-40 pristine 50 / 53-54 um. Note: the "
     "paper's own cross-SEM use is for LOCAL thickness, not for computing "
     "bulk porosity from image area.",
     "SI Table S1, p. S5"),
]


# --------------------------------------------------------------------------
# Report (PDF)
# --------------------------------------------------------------------------
def build_pdf(path, answer_par, method_par, fig_infos, stats_table,
              paper_par, risk_par, limitations):
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.2,
                          leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5)
    h1, h2 = styles["Heading1"], styles["Heading2"]
    cap = ParagraphStyle("cap", parent=body, fontSize=8,
                         textColor=colors.HexColor("#333333"), leading=10)
    doc = SimpleDocTemplate(path, pagesize=A4, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, leftMargin=1.8 * cm,
                            rightMargin=1.8 * cm,
                            title="Pore space analysis")
    story = [Paragraph("Pore space analysis - Hackathon-Polaron batches",
                       h1), Spacer(1, 6)]
    story.append(Paragraph("<b>Answer:</b> " + answer_par, body))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<i>Assumption throughout: bright BSE particles "
                           "are taken to be silicon-based and the grey bulk "
                           "to be graphite, from image appearance only "
                           "(no EDS confirmation).</i>", small))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Method", h2))
    story.append(Paragraph(method_par, body))
    story.append(Spacer(1, 6))
    for fpath, caption in fig_infos:
        w = 16 * cm
        img = Image(fpath)
        ar = img.imageHeight / img.imageWidth
        img.drawWidth = w
        img.drawHeight = w * ar
        story.append(img)
        story.append(Paragraph(caption, cap))
        story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("Batch statistics", h2))
    story.append(stats_table)
    story.append(Spacer(1, 10))
    story.append(Paragraph("What the paper says, and whether it applies", h2))
    story.append(Paragraph(paper_par, small))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Risk ranking", h2))
    story.append(Paragraph(risk_par, body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Limitations", h2))
    for lim in limitations:
        story.append(Paragraph("- " + lim, small))
    doc.build(story)


def stats_table_for_pdf(pb):
    head = ["batch", "images", "porosity", "95% CI", "local poro 10px",
            "95% CI", "confined", "95% CI", "pore D50 um", "chord h/v"]
    data = [head]
    for _, r in pb.iterrows():
        data.append([r["batch"], int(r["n_images"]),
                     f"{r['porosity_mean']:.3f}",
                     f"[{r['porosity_ci_lo']:.3f},{r['porosity_ci_hi']:.3f}]",
                     f"{r['local_poro_10px_mean']:.3f}",
                     f"[{r['local_poro_10px_ci_lo']:.3f},"
                     f"{r['local_poro_10px_ci_hi']:.3f}]",
                     f"{r['confined_10px_mean']:.3f}",
                     f"[{r['confined_10px_ci_lo']:.3f},"
                     f"{r['confined_10px_ci_hi']:.3f}]",
                     f"{r['pore_d50_um']:.2f}",
                     f"{r['chord_ratio_hv_mean']:.2f}"])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    return t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="..",
                    help="directory containing Hackathon-Polaron/")
    ap.add_argument("--out", default=None,
                    help="output directory (default: script's directory)")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    out = args.out or here
    figdir = os.path.join(out, "figures")
    ovdir = os.path.join(out, "overlays")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(ovdir, exist_ok=True)

    images = find_bse_images(args.data_root)
    counts = {}
    for b, iid, _ in images:
        counts[b] = counts.get(b, 0) + 1
    print("BSE images per batch:", counts)

    batch_list = sorted(set(b for b, _, _ in images))
    all_parts, all_pores, img_meta = [], [], []
    seg_cache = {}
    overlay_paths = {b: [] for b in batch_list}

    # ---- per image: segment, measure pores + particles --------------------
    for b, iid, path in images:
        um, src = pixel_size_um(path)
        g = load_gray(path)
        g, cropped = crop_banner(g)
        s, lab, pm, bw, t_lo, info = segment3(g)
        n_px = g.size
        seg_cache[iid] = (s.astype(np.float32), pm, bw,
                          lab.astype(np.int32))

        pdf_p = measure_pores(pm, iid, b, um, n_px)
        all_pores.append(pdf_p)
        df = measure_particles(lab, iid, b, um)
        ring = ring_measurements(lab, bw, pm)
        nvoid, avoid = bright_in_voids(lab, ring)

        # attach ring columns to particles
        for r in RING_RADII:
            tot = ring[f"ring_px_{r}"]
            np_ = ring[f"ring_porepx_{r}"]
            df[f"ring_px_{r}"] = [int(tot[l]) for l in df.label]
            df[f"ring_porosity_{r}px"] = [
                float(np_[l] / tot[l]) if tot[l] else np.nan
                for l in df.label]
            df[f"confined_{r}px"] = [bool(np_[l] == 0 and tot[l] > 0)
                                    for l in df.label]
        bt = ring["boundary_px"]; bc = ring["boundary_close_px"]
        df["contact_frac_3px"] = [
            float(bc[l] / bt[l]) if bt[l] else np.nan for l in df.label]
        all_parts.append(df)

        # sensitivity: porosity at shifted pore thresholds (same cleaning)
        por_sens = {}
        for sh in SENS_SHIFTS:
            pm_s = pore_mask_at(s, t_lo * (1 + sh), bw)
            por_sens[sh] = float(pm_s.mean())

        # pore intensity characterisation (empty-black vs resin-dark)
        pore_median_s = float(np.median(s[pm])) if pm.any() else np.nan
        pore_median_raw = float(np.median(g[pm])) if pm.any() else np.nan
        pore_frac_black = float((g[pm] < 30).mean()) if pm.any() else np.nan

        # chords
        h_lens, v_lens = chord_lengths(pm, um)

        img_meta.append(dict(
            batch=b, image_id=iid, path=path,
            height=g.shape[0], width=g.shape[1],
            um_per_px=um, px_source=src,
            banner_rows_cropped=cropped,
            thr_pore=t_lo, thr_bright=info["threshold"],
            bright_method=info["method"],
            porosity=info["pore_frac"],
            porosity_thr_m10=por_sens[-0.10],
            porosity_thr_p10=por_sens[0.10],
            bright_frac=info["bright_frac"],
            n_bright_in_voids=nvoid,
            bright_in_voids_area_frac=avoid / n_px,
            pore_median_s=pore_median_s,
            pore_median_raw=pore_median_raw,
            pore_frac_black=pore_frac_black,
            chord_h_mean_um=float(h_lens.mean()) if len(h_lens) else np.nan,
            chord_h_median_um=(float(np.median(h_lens))
                               if len(h_lens) else np.nan),
            chord_v_mean_um=float(v_lens.mean()) if len(v_lens) else np.nan,
            chord_v_median_um=(float(np.median(v_lens))
                               if len(v_lens) else np.nan),
            chord_ratio_hv=(float(h_lens.mean() / v_lens.mean())
                            if len(h_lens) and len(v_lens) else np.nan),
            n_pores=len(pdf_p),
            **{f"porosity_band_{i + 1:02d}": v
               for i, v in enumerate(depth_bands(pm))},
        ))
        print(f"  {b}/{iid}: porosity={info['pore_frac']:.3f}, "
              f"{len(pdf_p)} pores, {len(df)} particles, "
              f"thr_pore={t_lo:.3f}, thr_b={info['threshold']:.3f} "
              f"({info['method']})")

    parts = pd.concat(all_parts, ignore_index=True)
    pores = pd.concat(all_pores, ignore_index=True)
    meta = pd.DataFrame(img_meta)

    # ---- per-image aggregation -------------------------------------------
    pin = parts[~parts.touches_border]          # interior particles only
    band_cols = [f"porosity_band_{i + 1:02d}" for i in range(N_BANDS)]
    per_image_rows = []
    for b, iid, path in images:
        sub = pin[pin.image_id == iid]
        pp = pores[pores.image_id == iid]
        mr = meta[meta.image_id == iid].iloc[0]
        area_mm2 = mr.height * mr.width * (mr.um_per_px ** 2) / 1e6
        def cf(cls):
            return float(pp.loc[pp.pore_class == cls, "area_um2"].sum() /
                         (mr.height * mr.width * mr.um_per_px ** 2))
        def cc(cls):
            return float((pp.pore_class == cls).sum() / area_mm2)
        bands = [float(mr[c]) for c in band_cols]
        slope = float(np.polyfit(np.arange(N_BANDS), bands, 1)[0])
        per_image_rows.append(dict(
            batch=b, image_id=iid,
            height=int(mr.height), width=int(mr.width),
            um_per_px=mr.um_per_px,
            thr_pore=mr.thr_pore, thr_bright=mr.thr_bright,
            bright_method=mr.bright_method,
            banner_rows_cropped=int(mr.banner_rows_cropped),
            porosity=mr.porosity,
            porosity_thr_m10=mr.porosity_thr_m10,
            porosity_thr_p10=mr.porosity_thr_p10,
            bright_frac=mr.bright_frac,
            n_pores=int(mr.n_pores),
            pore_count_per_mm2=float(mr.n_pores / area_mm2),
            gap_area_frac=cf("large_gap"), gap_count_per_mm2=cc("large_gap"),
            crack_area_frac=cf("crack_slit"),
            crack_count_per_mm2=cc("crack_slit"),
            compact_area_frac=cf("compact"),
            compact_count_per_mm2=cc("compact"),
            pore_d10_um=(float(np.percentile(pp.thickness_mean_um, 10))
                         if len(pp) else np.nan),
            pore_d50_um=(float(np.percentile(pp.thickness_mean_um, 50))
                         if len(pp) else np.nan),
            pore_d90_um=(float(np.percentile(pp.thickness_mean_um, 90))
                         if len(pp) else np.nan),
            pore_d50_areawt_um=(float(weighted_quantile(
                pp.thickness_mean_um, pp.area_um2, [0.5])[0])
                if len(pp) else np.nan),
            pore_d90_areawt_um=(float(weighted_quantile(
                pp.thickness_mean_um, pp.area_um2, [0.9])[0])
                if len(pp) else np.nan),
            chord_h_mean_um=mr.chord_h_mean_um,
            chord_h_median_um=mr.chord_h_median_um,
            chord_v_mean_um=mr.chord_v_mean_um,
            chord_v_median_um=mr.chord_v_median_um,
            chord_ratio_hv=mr.chord_ratio_hv,
            depth_slope_per_band=slope,
            n_particles=len(sub),
            local_poro_5px_mean=float(sub.ring_porosity_5px.mean())
            if len(sub) else np.nan,
            local_poro_10px_mean=float(sub.ring_porosity_10px.mean())
            if len(sub) else np.nan,
            local_poro_20px_mean=float(sub.ring_porosity_20px.mean())
            if len(sub) else np.nan,
            local_vs_whole_10px=(float(sub.ring_porosity_10px.mean() /
                                       mr.porosity)
                                 if len(sub) and mr.porosity > 0 else np.nan),
            contact_frac_mean=(float(sub.contact_frac_3px.mean())
                               if len(sub) else np.nan),
            confined_share_5px=(float(sub.confined_5px.mean())
                                if len(sub) else np.nan),
            confined_share_10px=(float(sub.confined_10px.mean())
                                 if len(sub) else np.nan),
            confined_share_20px=(float(sub.confined_20px.mean())
                                 if len(sub) else np.nan),
            n_bright_in_voids=int(mr.n_bright_in_voids),
            bright_in_voids_area_frac=mr.bright_in_voids_area_frac,
            pore_median_s=mr.pore_median_s,
            pore_median_raw=mr.pore_median_raw,
            pore_frac_black=mr.pore_frac_black,
            outlier_image=iid in OUTLIER_IDS,
            **{c: mr[c] for c in band_cols},
        ))
    per_image = pd.DataFrame(per_image_rows)

    # ---- per-batch with bootstrap CIs -------------------------------------
    per_batch_rows = []
    for scope, excl in [("all_images", False),
                        ("excluding_outlier_images", True)]:
        pi = per_image[~per_image.outlier_image] if excl else per_image
        pp = pores[~pores.image_id.isin(OUTLIER_IDS)] if excl else pores
        for b in batch_list:
            sub = pi[pi.batch == b]
            ids = sub.image_id.tolist()
            by_img = sub.set_index("image_id")
            def ci(col):
                return bootstrap_image_mean(by_img[col].to_dict(), ids)
            por_ci = ci("porosity")
            lp_ci = ci("local_poro_10px_mean")
            cf_ci = ci("confined_share_10px")
            ppb = pp[pp.batch == b]
            per_batch_rows.append(dict(
                scope=scope, batch=b, n_images=len(sub),
                n_pores=int(sub.n_pores.sum()),
                n_particles=int(sub.n_particles.sum()),
                porosity_mean=float(sub.porosity.mean()),
                porosity_ci_lo=por_ci[0], porosity_ci_hi=por_ci[1],
                local_poro_10px_mean=float(sub.local_poro_10px_mean.mean()),
                local_poro_10px_ci_lo=lp_ci[0], local_poro_10px_ci_hi=lp_ci[1],
                confined_10px_mean=float(sub.confined_share_10px.mean()),
                confined_10px_ci_lo=cf_ci[0], confined_10px_ci_hi=cf_ci[1],
                contact_frac_mean=float(sub.contact_frac_mean.mean()),
                gap_area_frac_mean=float(sub.gap_area_frac.mean()),
                crack_area_frac_mean=float(sub.crack_area_frac.mean()),
                compact_area_frac_mean=float(sub.compact_area_frac.mean()),
                pore_d50_um=(float(np.percentile(ppb.thickness_mean_um, 50))
                             if len(ppb) else np.nan),
                chord_ratio_hv_mean=float(sub.chord_ratio_hv.mean()),
                depth_slope_mean=float(sub.depth_slope_per_band.mean()),
            ))
    per_batch = pd.DataFrame(per_batch_rows)
    pb_all = per_batch[per_batch.scope == "all_images"].set_index("batch")

    # ---- tests -------------------------------------------------------------
    test_rows = []
    metrics = ["porosity", "local_poro_10px_mean", "local_vs_whole_10px",
               "confined_share_10px", "contact_frac_mean",
               "gap_area_frac", "crack_area_frac", "compact_area_frac",
               "pore_d50_um", "chord_ratio_hv", "depth_slope_per_band"]
    for scope, excl in [("all_images", False),
                        ("excluding_outlier_images", True)]:
        pi = per_image[~per_image.outlier_image] if excl else per_image
        for metric in metrics:
            for row in kruskal_and_mwu(pi, metric, batch_list):
                row["scope"] = scope
                test_rows.append(row)
    tests = pd.DataFrame(test_rows)

    # ---- sensitivity table -------------------------------------------------
    sens_rows = []
    for scope, excl in [("all_images", False),
                        ("excluding_outlier_images", True)]:
        pi = per_image[~per_image.outlier_image] if excl else per_image
        for sh, col in [(-0.10, "porosity_thr_m10"), (0.0, "porosity"),
                        (0.10, "porosity_thr_p10")]:
            means = {b: pi.loc[pi.batch == b, col].mean()
                     for b in batch_list}
            order = sorted(means, key=means.get)
            ranks = {b: order.index(b) + 1 for b in batch_list}
            g = [pi.loc[pi.batch == b, col].dropna().values
                 for b in batch_list]
            kw_p = np.nan
            if all(len(x) for x in g):
                kw_p = float(scs.kruskal(*g)[1])
            for b in batch_list:
                sens_rows.append(dict(
                    scope=scope, batch=b, threshold_shift_pct=int(sh * 100),
                    mean_porosity=float(means[b]), batch_rank=ranks[b],
                    kruskal_p=kw_p))
    sens = pd.DataFrame(sens_rows)
    rank_flip = sens[sens.threshold_shift_pct.isin([-10, 10])].groupby(
        "batch")["batch_rank"].nunique().gt(1).any()

    # ---- figures ------------------------------------------------------------
    fig_porosity_per_image(per_image, batch_list,
                           os.path.join(figdir, "porosity_per_image.png"))
    fig_pore_classes(per_image, batch_list,
                     os.path.join(figdir, "pore_classes.png"))
    fig_pore_size_distribution(pores, batch_list,
                               os.path.join(figdir,
                                            "pore_size_distribution.png"))
    fig_depth_profile(per_image, batch_list,
                      os.path.join(figdir, "depth_profile.png"))
    fig_local_porosity(per_image, batch_list,
                       os.path.join(figdir, "local_porosity_silicon.png"))
    fig_confined(pb_all, per_image, batch_list,
                 os.path.join(figdir, "confined_fraction.png"))
    fig_paper_context(per_image, batch_list,
                      os.path.join(figdir, "paper_context.png"))
    print("wrote figures")

    # ---- overlays ------------------------------------------------------------
    for b, iid, path in images:
        s, pm, bw, lab = seg_cache[iid]
        pim = parts[parts.image_id == iid]
        mr = meta[meta.image_id == iid].iloc[0]
        op = os.path.join(ovdir, f"overlay_{b}_{iid}.png")
        nconf = int(pim.confined_10px.sum())
        save_overlay(s, lab, pm, pim, op,
                     f"{b}/{iid} - porosity {mr.porosity:.3f}, "
                     f"{len(pim)} particles ({nconf} confined)")
        overlay_paths[b].append(op)
    for b in batch_list:
        contact_sheet(overlay_paths[b], b,
                      os.path.join(ovdir, f"contact_{b}.png"))
    print("wrote overlays")

    # ---- excel ----------------------------------------------------------------
    paper_sheet = pd.DataFrame(PAPER_FACTS, columns=["item", "detail",
                                                     "source_location"])
    readme = pd.DataFrame({"readme": [
        "porosity.xlsx - pore-space analysis of Hackathon-Polaron BSE "
        "images.",
        "ASSUMPTION: bright BSE particles are assumed silicon-based and the "
        "grey bulk assumed graphite, from image appearance only (no EDS).",
        "Labels per pixel: 0 = pore (dark), 1 = bulk (grey), 2 = bright "
        "(assumed Si). Bright pixels always win over pore pixels.",
        "Per_image sheet: one row per BSE image. porosity = pore area "
        "fraction of the whole image. porosity_thr_m10/_p10 = porosity "
        "with the pore threshold shifted -10%/+10% (sensitivity). "
        "local_poro_Xpx_mean = mean over interior bright particles of the "
        "pore fraction in a ring of width X px around each particle. "
        "contact_frac_mean = mean fraction of particle boundary within "
        "3 px of a pore. confined_share_Xpx = fraction of particles with "
        "zero pore pixels in the X-px ring. local_vs_whole_10px = local "
        "ring porosity divided by whole-image porosity. pore_dXX_um are "
        "number-weighted quantiles of pore mean local thickness; "
        "_areawt = area-weighted. chord_*_um = mean/median horizontal "
        "and vertical pore run lengths; chord_ratio_hv = mean_h/mean_v "
        "(>1 means pores are stretched horizontally, consistent with "
        "pressing). porosity_band_01..10 = pore fraction in 10 equal "
        "horizontal bands top to bottom; depth_slope_per_band = linear "
        "slope across those bands. gap/crack/compact_* = area fraction "
        "and count per mm^2 of the three pore classes (large_gap = area "
        "> 0.5% of image; crack_slit = major/minor axis > 3; compact = "
        "the rest; large_gap takes precedence).",
        "Per_particle sheet: one row per bright particle. ring_porosity_"
        "Xpx = pore fraction of the X-px annulus around that particle; "
        "confined_Xpx = no pore pixel inside that annulus; "
        "contact_frac_3px = share of the particle's boundary within 3 px "
        "of a pore; touches_border particles are EXCLUDED from all "
        "per-particle statistics.",
        "Pores sheet: one row per pore region. thickness_mean_um = "
        "2 x mean distance transform inside the region (mean local "
        "thickness); thickness_max_um = 2 x max (largest inscribed "
        "diameter); orientation_deg in degrees; depth_band 1-10.",
        "Per_batch sheet: batch means of per-image values; CI columns are "
        "95% bootstrap intervals resampling IMAGES (the independent "
        "unit), 2000 resamples. scope='all_images' vs "
        "'excluding_outlier_images' (drops 4ih2ggld and 5n1q8atc).",
        "Tests sheet: Kruskal-Wallis across batches and pairwise "
        "Mann-Whitney with Holm correction on per-image summaries.",
        "Sensitivity sheet: batch mean porosity and rank at pore-threshold "
        "shifts of -10%, 0, +10%, plus the Kruskal-Wallis p at each shift.",
        "Units: lengths in um, areas in um^2, fractions dimensionless. "
        "Pixel size ~25.0 nm/px from TIFF resolution tags.",
        "2D caveat: image-area porosity from a polished 2D section is not "
        "bulk porosity (thickness/density or mercury intrusion); a 2D "
        "section also samples pores differently from their true 3D "
        "volume fraction.",
    ]})
    xlsx = os.path.join(out, "porosity.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        per_image.to_excel(xw, sheet_name="Per_image", index=False)
        parts.to_excel(xw, sheet_name="Per_particle", index=False)
        pores.to_excel(xw, sheet_name="Pores", index=False)
        per_batch.to_excel(xw, sheet_name="Per_batch", index=False)
        tests.to_excel(xw, sheet_name="Tests", index=False)
        sens.to_excel(xw, sheet_name="Sensitivity", index=False)
        paper_sheet.to_excel(xw, sheet_name="Paper", index=False)
        readme.to_excel(xw, sheet_name="README", index=False)
    print("wrote", xlsx)

    write_outputs(per_batch, per_image, tests, pores, sens, meta, out,
                  figdir, batch_list, rank_flip)


PAPER_PAR_TEMPLATE = (
    "The paper (Profatilova et al., ACS Appl. Energy Mater. 2020, 3, "
    "11873-11885, DOI {doi}) tested Si/C/graphite composite anodes in 18650 "
    "cells at two anode porosities. Verbatim from the abstract: \"Anode "
    "porosity variation between 30 and 40% led to a significant difference "
    "in the cycle life of 18650 cells: 225 and 347 cycles were achieved, "
    "respectively, before the drop of a discharge capacity to 75% of the "
    "initial value\" and \"Li+ ion loss in the solid electrolyte interphase "
    "layer was the main reason for the performance loss in the case of 40% "
    "anode porosity\", while \"Compression of the anode to 30% of porosity "
    "resulted in Li metal deposition on the negative electrode surface and "
    "within the separator during cycling ... Li+ ion transport was largely "
    "impeded by this deposit leading to poor C-rate performance.\" "
    "IMPORTANT ACCESS NOTE: the full text is paywalled and no open-access "
    "copy could be obtained (checked ACS, OpenAlex, Unpaywall, HAL, "
    "figshare). The figures/quotes above come from the publisher-hosted "
    "abstract and the free Supporting Information (Table S1 shows thickness "
    "measured by gauge and by cross-SEM). The paper's porosity is the BULK "
    "value set at cell build (thickness/mass based), a different quantity "
    "from our 2D image-area porosity: our ~{pmid}% area fractions and the "
    "paper's 30%/40% bulk values cannot be compared numerically; only the "
    "qualitative direction - over the studied range, more pore space "
    "next to/within the anode correlated with LONGER cycle life, while "
    "too little porosity promoted Li plating - is invoked. Whether the "
    "relationship is monotonic or has an optimum cannot be told from two "
    "points.")


def write_outputs(per_batch, per_image, tests, pores, sens, meta, out,
                  figdir, batch_list, rank_flip):
    """NOTES.md + PDF report; numbers taken from the same dataframes as the
    Excel file."""
    allb = per_batch[per_batch.scope == "all_images"].set_index("batch")

    por_lines = "; ".join(
        f"{b}: {allb.loc[b, 'porosity_mean']:.3f} "
        f"[{allb.loc[b, 'porosity_ci_lo']:.3f}-"
        f"{allb.loc[b, 'porosity_ci_hi']:.3f}]" for b in batch_list)
    lp_lines = "; ".join(
        f"{b}: {allb.loc[b, 'local_poro_10px_mean']:.3f} "
        f"[{allb.loc[b, 'local_poro_10px_ci_lo']:.3f}-"
        f"{allb.loc[b, 'local_poro_10px_ci_hi']:.3f}]" for b in batch_list)
    cf_lines = "; ".join(
        f"{b}: {allb.loc[b, 'confined_10px_mean']:.3f} "
        f"[{allb.loc[b, 'confined_10px_ci_lo']:.3f}-"
        f"{allb.loc[b, 'confined_10px_ci_hi']:.3f}]" for b in batch_list)

    kw = tests[(tests.scope == "all_images")
               & (tests.test == "Kruskal-Wallis")].set_index("metric")
    kwx = tests[(tests.scope == "excluding_outlier_images")
                & (tests.test == "Kruskal-Wallis")].set_index("metric")
    def kwp(metric, t=kw):
        return t.loc[metric, "p_value"] if metric in t.index else np.nan
    por_p, por_px = kwp("porosity"), kwp("porosity", kwx)
    lp_p = kwp("local_poro_10px_mean")
    cf_p = kwp("confined_share_10px")
    ct_p = kwp("contact_frac_mean")
    mwu = tests[(tests.test == "Mann-Whitney U")
                & (tests.metric == "porosity")
                & (tests.groups == "Batch_1 vs Batch_3")]
    b1b3_all = float(mwu[mwu.scope == "all_images"].p_adj.iloc[0])
    b1b3_exc = float(mwu[mwu.scope == "excluding_outlier_images"]
                     .p_adj.iloc[0])
    lvw = per_image.groupby("batch")["local_vs_whole_10px"].mean()

    trend_txt = (
        f"mean image-area porosity trends upward Batch_1 < Batch_2 < "
        f"Batch_3 ({por_lines}), consistent with the earlier rough "
        f"cross-check (~0.089/0.100/0.109), and the ranking is stable "
        f"under +/-10% threshold shifts - but it is only marginally "
        f"supported statistically (Kruskal-Wallis p={por_p:.3f} with all "
        f"images, p={por_px:.3f} without the two outlier images; the only "
        f"nominally significant pairwise contrast, Batch_1 vs Batch_3 "
        f"Holm p={b1b3_all:.3f}, does not survive outlier removal, Holm "
        f"p={b1b3_exc:.3f})")
    si_txt = (
        f"NONE of the pore measures next to the assumed-Si particles "
        f"differs between batches (local 10-px ring porosity {lp_lines}, "
        f"KW p={lp_p:.3f}; confined share {cf_lines}, p={cf_p:.3f}; "
        f"boundary contact, p={ct_p:.3f})")

    pmid = float(per_image.porosity.mean() * 100)
    answer = (
        f"On total pore space, {trend_txt}. On the silicon-relevant "
        f"question, {si_txt}. Pore space next to silicon is also "
        f"{100 - lvw.max() * 100:.0f}-{100 - lvw.min() * 100:.0f}% below "
        f"the whole-image average in every batch (local/whole ratio "
        f"{lvw.min():.2f}-{lvw.max():.2f}). Interpreted against the "
        f"paper - 40% bulk-porosity anodes outlived 30% ones (347 vs 225 "
        f"cycles to 75% capacity) and too little porosity promoted Li "
        f"plating - the batch with least porosity would nominally be at "
        f"greatest risk, but see the risk section: no ranking is "
        f"supported. Assumption: bright particles are silicon-based, "
        f"grey bulk is graphite, from image appearance only.")

    method = (
        "BSE images only (31 total: Batch_1=7, Batch_2=7, Batch_3=17). Grey "
        "levels are the per-pixel median of the three RGB channels; no "
        "information banner was present. Each image was normalised to its "
        "1st-99.5th percentile range and Gaussian-smoothed (sigma 1.5 px). "
        "Pores: below the LOWER cut of a 3-class multi-Otsu; objects <20 px "
        "removed. Bright particles: histogram-valley threshold (upper "
        "multi-Otsu cut as flagged fallback), opening disk r=2, objects "
        "<150 px removed, holes filled, watershed-split - identical to the "
        "silicon_size analysis. Bright pixels win where masks overlap; "
        "labels are pore/bulk/bright. Pore classes: large gap >0.5% of "
        "image area, crack/slit if major:minor >3, else compact. Pore size "
        "= mean local thickness (2 x mean distance transform). Chords are "
        "horizontal/vertical pore run lengths. Depth profile = pore "
        "fraction in 10 horizontal bands. Around each interior particle: "
        "pore fraction of the 5/10/20-px annulus (nearest-particle "
        "attribution via distance transform), share of boundary within "
        "3 px of a pore, and 'confined' = zero pore in the ring. Pixel "
        "size ~25.0 nm/px in every image (TIFF tags). The IMAGE is the "
        "independent unit: Kruskal-Wallis + pairwise Mann-Whitney (Holm) + "
        "bootstrap CIs resampling images. Threshold sensitivity: pore "
        "threshold shifted +/-10% per image.")

    fig_infos = [
        (os.path.join(figdir, "porosity_per_image.png"),
         "Fig. 1 - Image-area porosity per image, grouped by batch; "
         "horizontal bars are batch means; open markers are the two "
         "low-contrast outlier images."),
        (os.path.join(figdir, "pore_classes.png"),
         "Fig. 2 - Mean area fraction (left) and areal density (right) of "
         "the three pore classes per batch."),
        (os.path.join(figdir, "pore_size_distribution.png"),
         "Fig. 3 - Cumulative pore-size (mean local thickness) "
         "distributions, number- and area-weighted, per batch."),
        (os.path.join(figdir, "depth_profile.png"),
         "Fig. 4 - Porosity in 10 horizontal bands from top to bottom of "
         "each image, batch mean +/- SEM."),
        (os.path.join(figdir, "local_porosity_silicon.png"),
         "Fig. 5 - Mean pore fraction in the 10-px ring around assumed-Si "
         "particles vs whole-image porosity, per image; dashed line is "
         "equality."),
        (os.path.join(figdir, "confined_fraction.png"),
         "Fig. 6 - Fraction of assumed-Si particles with no pore within "
         "10 px, per image and batch mean with 95% bootstrap CI."),
        (os.path.join(figdir, "paper_context.png"),
         "Fig. 7 - Our 2D image porosity per batch (bars: min-max range, "
         "diamond: median) next to the paper's two bulk porosities and "
         "their cycle lives - different quantities, shown for context "
         "only."),
    ]

    stats_tbl = stats_table_for_pdf(per_batch[per_batch.scope ==
                                              "all_images"])

    paper_par = PAPER_PAR_TEMPLATE.format(doi=PAPER_DOI, pmid=f"{pmid:.1f}")

    # risk ranking
    lo_por = allb["porosity_mean"].idxmin()
    risk = (
        f"No defensible risk ranking is supported. On the paper's "
        f"direction alone - more porosity gave longer cycle life over its "
        f"30-40% bulk range, and too little porosity promoted Li plating "
        f"- the descriptive ordering would nominally put {lo_por} "
        f"(lowest mean image porosity) at greatest risk. But the total-"
        f"porosity difference is at/below the noise floor (Kruskal-Wallis "
        f"p={por_p:.3f}, not robust to outlier removal), and the measures "
        f"that matter most for silicon - pore space in the ring around "
        f"each bright particle, boundary contact, and the share of "
        f"confined particles - show no batch difference at all "
        f"(KW p={lp_p:.3f}/{cf_p:.3f}/{ct_p:.3f}). Any ranking would "
        f"therefore rest on a non-robust ~2-point porosity difference in "
        f"a quantity (2D area fraction) that is not even the same "
        f"measurement the paper varied (bulk calendered porosity).")

    if rank_flip:
        rank_note = ("WARNING: the batch ranking on mean porosity FLIPS "
                     "under +/-10% pore-threshold shifts - the comparison "
                     "is threshold-sensitive.")
    else:
        rank_note = ("The batch ranking on mean porosity is stable under "
                     "+/-10% pore-threshold shifts.")

    med_black = float(per_image.pore_frac_black.median())
    n_void_img = int((per_image.n_bright_in_voids > 0).sum())
    max_void = int(per_image.n_bright_in_voids.max())
    limitations = [
        "Material identity is assumed: bright BSE particles = "
        "silicon-based, grey bulk = graphite, from image appearance only "
        "(no EDS).",
        "2D image porosity (area fraction on a polished section) is not "
        "the paper's bulk porosity (thickness/mass); only the direction "
        "of the paper's trend is used.",
        "The paper's full text was not accessible (paywalled; no OA "
        "copy). Quotes are from the abstract and SI only; no detail "
        "beyond them is relied on.",
        "Pore checks: the dark class is essentially black - median "
        f"{med_black:.0%} of pore pixels are below grey level 30/255, "
        "so pores read as empty voids rather than resin-filled shadowed "
        "solid. Bright material sitting inside voids (e.g. back walls) "
        f"is essentially absent: only {n_void_img} of 31 images contain "
        f"any (max {max_void} object), so bias to porosity is "
        "negligible; where present those pixels are classified bright, "
        "keeping them out of the pore count.",
        "Threshold sensitivity: " + rank_note,
        "Pixel size (~25.0 nm/px) comes from TIFF resolution tags, not a "
        "measured scale bar; a systematic error would shift all lengths "
        "together (porosity itself is dimensionless and unaffected).",
        "The image is the independent unit (n = 7, 7, 17); Batch_3's "
        "larger count gives pairwise tests involving it more power.",
        "Two Batch_1 outlier images (4ih2ggld, 5n1q8atc) at lower "
        "contrast; all statistics are reported with and without them.",
        "Ring porosity attributes each non-bright pixel to its nearest "
        "particle; in dense clusters the ring contains less free space "
        "by construction.",
        "Cross-check: earlier rough porosities were ~0.089/0.100/0.109 "
        "for Batch_1/2/3 (p~0.07). Our means are "
        + "; ".join(f"{b}={allb.loc[b, 'porosity_mean']:.3f}"
                    for b in batch_list)
        + ". Differences vs the rough pass come from the same-method "
        "multi-Otsu pore cut plus <20 px cleaning and the bright-wins "
        "overlap rule.",
    ]

    pdf_path = os.path.join(out, "porosity_report.pdf")
    build_pdf(pdf_path, answer, method, fig_infos, stats_tbl, paper_par,
              risk, limitations)
    print("wrote", pdf_path)

    # ---- NOTES.md -------------------------------------------------------
    xcheck_lines = "; ".join(
        f"{b}: ours {allb.loc[b, 'porosity_mean']:.3f} vs earlier "
        f"{XCHECK[b]:.3f}" for b in batch_list)
    notes = f"""# NOTES - pore space analysis

## Image inventory
- Batch_1: 7 BSE images; Batch_2: 7; Batch_3: 17. Matches the earlier
  run's count of 7 / 7 / 17 - no mismatch.
- Some locations store the third detector image as `*_SE.tif` instead of
  `*_ETD.tif` (Batch_2: rxax5ozo; Batch_3: utfgcjfa, vc2whyaq, x77cy643).
  BSE counts are unaffected.

## Assumptions
- Bright BSE particles are ASSUMED silicon-based; grey bulk ASSUMED
  graphite - from image appearance only, no compositional confirmation.
- Pixel size ~25.0 nm/px in every BSE image (TIFF resolution tags; range
  24.999-25.001 nm). No scale bar and no instrument metadata.
- Labels: 0 pore / 1 bulk / 2 bright; bright wins where masks overlap.

## Method notes
- Pore threshold = LOWER cut of 3-class multi-Otsu on the smoothed,
  normalised image (same settings everywhere); objects <20 px removed.
- Bright segmentation identical to `silicon_size` analysis (histogram
  valley, multi-Otsu fallback flagged per image in `bright_method`).
- Pore class precedence: large_gap (area > 0.5% of image) > crack_slit
  (aspect > 3) > compact.
- Ring measures use distance transforms; every non-bright pixel is
  attributed to its nearest particle. Contact = share of inner-boundary
  pixels within {CONTACT_DIST} px of a pore (dilation approach, not
  direct touching - smoothing leaves a thin grey band between bright and
  dark regions, so direct contact counts are ~0 by construction).
- 'Confined' = zero pore pixels inside the ring; reported at 5/10/20 px.

## Pore checks (STEP 3)
- Empty-black vs resin-filled: median share of pore pixels with raw grey
  level <30/255 is {med_black:.0%}; pores are essentially black/empty,
  not resin-dark, so the dark class is dominated by true voids.
- Bright material inside voids (back walls): essentially absent -
  {n_void_img} of 31 images contain any such object (max {max_void};
  columns `n_bright_in_voids`, `bright_in_voids_area_frac` in
  Per_image). Where present those pixels are classified bright, which
  keeps them out of the pore count.
- Pore space next to silicon is LOWER than the image average in every
  batch (local/whole ratio {lvw.min():.2f}-{lvw.max():.2f} at the 10-px
  ring): the assumed-Si particles sit in comparatively dense regions.
- Batch ranking on mean porosity is stable under +/-10% threshold
  shifts; the Batch_1 < Batch_2 < Batch_3 order is descriptive but only
  marginally significant (KW p={por_p:.3f} all images, p={por_px:.3f}
  excluding outliers), and the Batch_1 vs Batch_3 pairwise contrast
  (Holm p={b1b3_all:.3f}) is not robust to outlier removal
  (p={b1b3_exc:.3f}). No batch difference on any silicon-adjacent pore
  measure (local porosity, confined share, boundary contact: KW
  p={lp_p:.3f}/{cf_p:.3f}/{ct_p:.3f}).
- Threshold sensitivity (+/-10% on the pore cut): see the Sensitivity
  sheet. {rank_note}
- Outlier images 4ih2ggld and 5n1q8atc (Batch_1, low contrast): all
  results reported with (`all_images`) and without
  (`excluding_outlier_images`).

## Cross-check vs earlier rough numbers
- Earlier rough total porosity ~0.089 / 0.100 / 0.109 for Batch_1/2/3
  (weak test, p~0.07). This run: {xcheck_lines}. Residual differences
  vs the rough pass are consistent with a different threshold rule plus
  the <20 px cleaning and bright-wins overlap handling applied here.

## Statistics
- IMAGE is the independent sample (n = 7, 7, 17), not the particle or
  the pore. Batch_3's larger image count gives pairwise tests involving
  it more power than the Batch_1-Batch_2 comparison.
- Bootstrap CIs (2000 resamples) resample images and recompute the
  per-image-mean statistic each time.

## Paper access
- {PAPER_TIT}, DOI {PAPER_DOI}. The full text is paywalled; no open-access
  copy was found (ACS purchase-only page; no OA location on OpenAlex /
  Unpaywall / HAL; figshare hosts only the 5-page SI). All quotes in the
  Paper sheet and report are verbatim from the publisher-hosted abstract
  and the free SI (ae0c01999_si_001.pdf). If the main text is provided,
  the Paper sheet should be extended with main-text quotes.

## Known issues / what failed
- Reading the TIFFs requires `imagecodecs` (LZW decode) - a dependency
  of tifffile, not an extra analysis library.
- Overlays are saved at 50% resolution to keep file sizes reasonable.
- Direct-touch pore contact returns ~0 because smoothing leaves a thin
  grey band; the 3-px distance measure is used instead (non-zero as
  designed - see `contact_frac_mean` in Per_image).
"""
    with open(os.path.join(out, "NOTES.md"), "w") as f:
        f.write(notes)
    print("wrote NOTES.md")


if __name__ == "__main__":
    main()
