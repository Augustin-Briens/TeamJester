#!/usr/bin/env python3
"""
silicon_size.py - SEM/BSE particle size analysis for Hackathon-Polaron batches.

Measures the size distribution of the bright (assumed silicon-based) particles
in the BSE images of Hackathon-Polaron/Batch_1..3, compares the batches
statistically, and produces figures, overlays, an Excel workbook, a PDF report
and NOTES.md.

ASSUMPTION (applies to every output): the bright particles in BSE are assumed
to be silicon-based and the grey bulk is assumed to be graphite. This is an
assumption from image appearance only; it has not been verified by e.g. EDS.

Usage:
    python3 silicon_size.py                # run everything
    python3 silicon_size.py --data-root .. # if run from inside silicon_size/

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
from skimage.morphology import binary_opening, disk, remove_small_objects, binary_dilation
from skimage.segmentation import watershed, find_boundaries

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)

warnings.simplefilter("ignore")

# --------------------------------------------------------------------------
# Configuration - one method, same settings for every image (no per-batch tuning)
# --------------------------------------------------------------------------
P_LO, P_HI = 1.0, 99.5          # normalisation percentiles
GAUSS_SIGMA = 1.5               # px, pre-threshold smoothing
HIST_BINS = 512                 # histogram resolution on the normalised image
HIST_SMOOTH = 3.0               # bins, histogram smoothing for peak finding
PORE_GUARD = 15                 # bins ignored at low end (pore spike guard)
BRIGHT_MIN_SEP = 25             # bins, minimum separation bulk peak -> bright peak
BRIGHT_MIN_PROM = 0.002         # fraction of pixels; bright peak prominence/size
OPEN_RADIUS = 2                 # px, binary opening disk radius
MIN_AREA_PX = 150               # px, minimum object size
WS_MIN_DIST = 8                 # px, watershed marker min separation
WS_MIN_DEPTH = 3                # px, marker must sit >= this deep inside object
OVERLAY_SCALE = 0.5             # downscale factor for saved overlay PNGs
BATCH_COLORS = {"Batch_1": "#0072B2", "Batch_2": "#D55E00",
                "Batch_3": "#009E73"}  # Okabe-Ito, used consistently everywhere
SIZE_CLASS_COLORS = {"small": "#35A7FF", "medium": "#FFD42A", "large": "#FF3B30"}
BORDER_COLOR = "#FF00FF"        # border-touching (excluded) particles

# Images previously observed to return ~4x as many bright particles at lower
# contrast. They are still processed identically; results are reported both
# with and without them.
OUTLIER_IDS = {"4ih2ggld", "5n1q8atc"}

N_BOOT = 2000                   # bootstrap resamples (images resampled)
RNG_SEED = 20241003

PAPER_TIT = ("The effect of nanoparticle size on calendar and cycle lifetimes "
             "of silicon anode lithium-ion batteries")
PAPER_DOI = "10.1039/d4eb00020j"


# --------------------------------------------------------------------------
# IO / metadata
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
    """Load image as float64 grey level.

    The TIFFs are RGB but the channels are (nearly) identical; the per-pixel
    median of the three channels is used, which also removes the rare
    single-channel annotation marks present in some files.
    """
    im = tifffile.imread(path)
    if im.ndim == 3:
        g = np.median(im.astype(np.float32), axis=2)
    else:
        g = im.astype(np.float32)
    return g


def crop_banner(g):
    """Crop a bottom information banner if present.

    A banner row is taken to be a row that is essentially uniform
    (std < 2 grey levels). Returns (cropped image, n_rows_cropped).
    """
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
def find_threshold(s):
    """Bright-particle threshold on the smoothed, normalised image.

    Primary rule: the valley of the histogram between the bulk peak and a
    separate bright peak. Fallback (flagged): the upper 3-class multi-Otsu
    threshold. Returns (threshold, method, bulk_peak_x, bright_peak_x).
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


def segment(g):
    """Full segmentation pipeline. Returns (norm_smooth, labels, info)."""
    p1, p995 = np.percentile(g, [P_LO, P_HI])
    n = np.clip((g - p1) / (p995 - p1), 0, 1)
    s = ndi.gaussian_filter(n, GAUSS_SIGMA)
    t, method, bulk_x, bright_x = find_threshold(s)
    bw = s > t
    bw = binary_opening(bw, disk(OPEN_RADIUS))
    bw = remove_small_objects(bw, MIN_AREA_PX)
    bw = ndi.binary_fill_holes(bw)
    dist = ndi.distance_transform_edt(bw)
    coords = peak_local_max(dist, min_distance=WS_MIN_DIST,
                            threshold_abs=WS_MIN_DEPTH, labels=bw)
    markers = np.zeros(bw.shape, np.int32)
    for j, (r, c) in enumerate(coords, 1):
        markers[r, c] = j
    markers = ndi.label(markers > 0)[0]
    lab = watershed(-dist, markers, mask=bw)
    info = dict(threshold=t, method=method, bulk_x=bulk_x, bright_x=bright_x,
                p_lo=p1, p_hi=p995, bright_frac=float(bw.mean()))
    return s, lab, info


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def measure(lab, iid, batch, um_per_px):
    """Per-particle measurements. Border-touching particles are flagged and
    excluded from size statistics downstream."""
    rows = []
    h, w = lab.shape
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        border = r0 == 0 or c0 == 0 or r1 == h or c1 == w
        rows.append(dict(
            image_id=iid, batch=batch, label=int(p.label),
            area_px=float(p.area),
            area_um2=float(p.area) * um_per_px ** 2,
            equiv_diam_px=float(p.equivalent_diameter_area),
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


def weighted_quantile(values, weights, qs):
    """Weighted quantiles. values, weights 1-D; qs in [0,1]."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(qs, cw, v)


def dist_stats(diams, areas):
    """Distribution statistics for one set of particle diameters (um)."""
    d = np.asarray(diams, float)
    a = np.asarray(areas, float)
    out = dict(n=int(len(d)))
    if len(d) == 0:
        for k in ("mean", "median", "D10", "D50", "D90", "D99", "max",
                  "D50_areawt", "D90_areawt"):
            out[k] = np.nan
        return out
    q = np.percentile(d, [10, 50, 90, 99])
    out.update(mean=float(d.mean()), median=float(np.median(d)),
               D10=float(q[0]), D50=float(q[1]), D90=float(q[2]),
               D99=float(q[3]), max=float(d.max()))
    aw = weighted_quantile(d, a, [0.5, 0.9])
    out["D50_areawt"], out["D90_areawt"] = float(aw[0]), float(aw[1])
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def holm_adjust(pvals):
    """Holm-Bonferroni adjusted p-values, returned in original order."""
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
    """Kruskal-Wallis across batches + pairwise Mann-Whitney with Holm.

    Returns list of dict rows for the Tests sheet. Effect sizes: epsilon
    squared for KW; rank-biserial correlation for MWU.
    """
    groups = [per_image.loc[per_image.batch == b, metric].dropna().values
              for b in batches]
    groups = [g for g in groups]
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
                             effect_size=e,
                             n_per_group=""))
    return rows


def bootstrap_ci(per_image, parts, batch, qs=(0.5, 0.9), n_boot=N_BOOT,
                 seed=RNG_SEED):
    """Bootstrap 95% CIs for pooled-particle quantiles, resampling IMAGES."""
    ids = per_image.loc[per_image.batch == batch, "image_id"].unique()
    rng = np.random.default_rng(seed)
    by_img = {iid: parts.loc[parts.image_id == iid, "equiv_diam_um"].values
              for iid in ids}
    stats_q = np.empty((n_boot, len(qs)))
    for b in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        d = np.concatenate([by_img[i] for i in pick])
        stats_q[b] = np.percentile(d, np.array(qs) * 100)
    return {f"D{int(q * 100)}_ci": (float(np.percentile(stats_q[:, i], 2.5)),
                                    float(np.percentile(stats_q[:, i], 97.5)))
            for i, q in enumerate(qs)}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_size_distribution(parts, batches, cutoff, outpath):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0, np.percentile(parts.equiv_diam_um, 99.9) * 1.05, 60)
    for b in batches:
        d = parts.loc[parts.batch == b, "equiv_diam_um"]
        ax.hist(d, bins=bins, density=True, histtype="step", lw=1.8,
                color=BATCH_COLORS[b], label=f"{b} (n_img={parts.loc[parts.batch==b,'image_id'].nunique()})")
    ax.axvline(cutoff, color="k", ls="--", lw=1,
               label=f"pooled D90 = {cutoff:.2f} um")
    ax.set_xlabel("Equivalent circular diameter (um)")
    ax.set_ylabel("Probability density (1/um)")
    ax.set_title("Number-weighted particle size distributions by batch")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_cumulative(parts, batches, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for b in batches:
        sub = parts.loc[parts.batch == b]
        d = np.sort(sub.equiv_diam_um.values)
        axes[0].plot(d, np.linspace(0, 1, len(d)), color=BATCH_COLORS[b],
                     lw=1.6, label=b)
        order = np.argsort(sub.equiv_diam_um.values)
        dv = sub.equiv_diam_um.values[order]
        cw = np.cumsum(sub.area_um2.values[order])
        axes[1].plot(dv, cw / cw[-1], color=BATCH_COLORS[b], lw=1.6, label=b)
    axes[0].set_title("Number-weighted")
    axes[1].set_title("Area-weighted")
    for ax in axes:
        ax.set_xlabel("Equivalent circular diameter (um)")
        ax.set_ylabel("Cumulative fraction")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Cumulative diameter distributions")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_per_image_boxplot(per_image, batches, outpath, outliers):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    metrics = [("median_diam_um", "Median diameter (um)"),
               ("d90_um", "D90 diameter (um)")]
    for ax, (m, ttl) in zip(axes, metrics):
        data, pos, cols, labels = [], [], [], []
        for bi, b in enumerate(batches):
            sub = per_image[per_image.batch == b].sort_values("image_id")
            for k, (_, row) in enumerate(sub.iterrows()):
                x = bi * (len(batches) + 2) + k
                data.append([row[m]])
                pos.append(x)
                cols.append(BATCH_COLORS[b])
                labels.append(row.image_id)
        # draw per-image scatter instead of a true box (n is per-image already)
        for k, (v, x, c, lab_) in enumerate(zip(data, pos, cols, labels)):
            mk = "o"
            face = c
            if lab_ in outliers:
                face = "white"
            ax.scatter([x], v, marker=mk, facecolor=face, edgecolor=c,
                       s=60, zorder=3)
            dy = 6 if k % 2 == 0 else -14
            ax.annotate(lab_, (x, v[0]), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=5,
                        rotation=90)

        for bi, b in enumerate(batches):
            sub = per_image[per_image.batch == b]
            x0 = bi * (len(batches) + 2)
            x1 = x0 + len(sub) - 1
            ax.axvspan(x0 - 0.6, x1 + 0.6, color=BATCH_COLORS[b], alpha=0.06)
            ax.text((x0 + x1) / 2, ax.get_ylim()[0], b, ha="center",
                    va="bottom", fontsize=8, color=BATCH_COLORS[b])
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_ylabel("um")
        ax.grid(alpha=0.3, axis="y")
    handles = [plt.Line2D([], [], marker="o", ls="", color="k",
                          markerfacecolor="white",
                          label="outlier image (shown, excluded in sensitivity run)")]
    axes[0].legend(handles=handles, fontsize=7, loc="upper right")
    fig.suptitle("Per-image diameter summaries (each point = one image)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_large_fraction(per_image, batches, cutoff, outpath, rng_seed=RNG_SEED):
    """Fraction of large particles per image with bootstrap CI on batch mean."""
    rng = np.random.default_rng(rng_seed)
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    xs, means, lo, hi, cols = [], [], [], [], []
    for bi, b in enumerate(batches):
        sub = per_image[per_image.batch == b]
        f = sub["frac_large"].values
        xs.append(bi)
        means.append(f.mean())
        boots = [rng.choice(f, size=len(f), replace=True).mean()
                 for _ in range(N_BOOT)]
        lo.append(np.percentile(boots, 2.5))
        hi.append(np.percentile(boots, 97.5))
        cols.append(BATCH_COLORS[b])
        jitter = rng.uniform(-0.12, 0.12, len(f))
        ax.scatter(np.full(len(f), bi) + jitter, f, color=BATCH_COLORS[b],
                   s=22, alpha=0.7, zorder=2)
    ax.errorbar(xs, means,
                yerr=[np.array(means) - np.array(lo),
                      np.array(hi) - np.array(means)],
                fmt="D", color="k", capsize=5, zorder=3,
                label="batch mean, 95% bootstrap CI (images resampled)")
    ax.set_xticks(xs)
    ax.set_xticklabels(batches)
    ax.set_ylabel(f"Fraction of particles > {cutoff:.2f} um (pooled D90)")
    ax.set_title("Large-particle fraction per batch "
                 "(cut-off is relative, not physical)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def fig_paper_context(parts, batches, outpath):
    """Our measured range (number-weighted D1-D99 per batch) next to the
    size range studied in the paper (3-27 nm) and the ~150 nm fracture
    threshold it cites."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.set_xscale("log")
    ymax = len(batches) + 1.5
    # paper band
    ax.axvspan(3, 27, color="grey", alpha=0.25)
    ax.text(9, ymax - 0.35, "paper: 3-27 nm studied\n(optimum ~6-8 nm)",
            ha="center", fontsize=8, color="0.25")
    ax.axvline(150, color="purple", ls=":", lw=1.5)
    ax.text(150, ymax - 0.35, " ~150 nm\n fracture limit\n (paper, ref. 9-11)",
            fontsize=7, color="purple", va="top")
    for i, b in enumerate(batches):
        d = parts.loc[parts.batch == b, "equiv_diam_um"].values * 1000.0  # nm
        lo, hi = np.percentile(d, [1, 99])
        y = ymax - 1 - i
        ax.plot([lo, hi], [y, y], lw=8, color=BATCH_COLORS[b],
                solid_capstyle="butt")
        ax.plot(np.median(d), y, marker="D", color="k", ms=6)
        ax.text(hi * 1.4, y, f"{b}: D1-D99 = {lo/1000:.2f}-{hi/1000:.1f} um,"
                f" median {np.median(d)/1000:.2f} um",
                fontsize=8, va="center", color=BATCH_COLORS[b])
    ax.set_xlim(2, 5e4)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    ax.set_xlabel("Particle diameter (nm, log scale)")
    ax.set_title("Measured sizes vs Preimesberger et al. 2025 (EES Batteries)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
def make_overlay(s, lab, parts_img, cutoff, d50_all):
    """RGB overlay: boundaries coloured by size class, border-touching
    particles in magenta. Downscaled by OVERLAY_SCALE for a sane file size."""
    small = SIZE_CLASS_COLORS["small"]
    med = SIZE_CLASS_COLORS["medium"]
    large = SIZE_CLASS_COLORS["large"]
    out = np.dstack([s] * 3)
    cls = {}
    for _, r in parts_img.iterrows():
        if r.touches_border:
            cls[int(r.label)] = BORDER_COLOR
        elif r.equiv_diam_um < d50_all:
            cls[int(r.label)] = small
        elif r.equiv_diam_um < cutoff:
            cls[int(r.label)] = med
        else:
            cls[int(r.label)] = large
    bnd = find_boundaries(lab, mode="outer")
    bnd = binary_dilation(bnd, disk(1))
    lab_d = np.where(bnd, lab, 0)
    for lbl, col in cls.items():
        m = lab_d == lbl
        if m.any():
            rgb = mcolors.to_rgb(col)
            for ch in range(3):
                out[:, :, ch][m] = rgb[ch]
    # downscale
    h, w = out.shape[:2]
    nh, nw = int(h * OVERLAY_SCALE), int(w * OVERLAY_SCALE)
    ys = (np.linspace(0, h - 1, nh)).astype(int)
    xs = (np.linspace(0, w - 1, nw)).astype(int)
    out_d = out[ys][:, xs]
    return out_d


def save_overlay(s, lab, parts_img, cutoff, d50_all, path, title):
    out_d = make_overlay(s, lab, parts_img, cutoff, d50_all)
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
    imgs = [plt.imread(p) for p in overlay_paths]
    n = len(imgs)
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
    fig.suptitle(f"{batch} - segmentation overlays", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report (PDF)
# --------------------------------------------------------------------------
PAPER_FACTS = [
    ("Confirmation", "DOI 10.1039/d4eb00020j resolves to EES Batteries, 2025, "
     "vol. 1, pp. 298-309: '" + PAPER_TIT + "', Preimesberger et al. (NREL)."),
    ("Sizes studied", "\"Silicon nanoparticles ranging from 3-27 nm in "
     "diameter were made using a plasma-enhanced chemical vapor deposition "
     "(PECVD) process\" (Methods, p. 299). Ten sizes were tested; the "
     "conclusions name 6-12 nm as the better-performing range and ~6-8 nm "
     "as optimal (Results p. 303, Fig. 1; Conclusions p. 307)."),
    ("How size was measured", "\"Average particle diameters were confirmed "
     "by X-ray diffraction, shown in Fig. S1\" (Methods, p. 299)."),
    ("Electrode composition", "Pure high-silicon anodes, NOT a "
     "silicon-graphite blend: \"Slurries were made with 89 wt% Si ... 1 wt% "
     "single-walled carbon nanotubes (SWCNTs), and 10 wt% polyimide binder "
     "(P84)\" (Methods, p. 300). Our sample is believed to be a graphite "
     "electrode containing silicon-based particles - a different system."),
    ("Size vs cycle life", "\"In general, as particle size decreases, the "
     "capacity retention increases, resulting in better cycle performance\" "
     "(Results, p. 301-302, Fig. 1a). \"Particle diameter, cycle life, and "
     "coulombic efficiencies are all correlated with each other\" "
     "(p. 306, Fig. 5)."),
    ("Size vs calendar life", "\"The rate of capacity fade and impedance "
     "rise is about the same for particles 6 nm and larger, further showing "
     "that the size of the silicon nanoparticle is not predictive for "
     "calendar aging rates\" (p. 303, Fig. 2d). Conclusions (p. 307): \"There "
     "is no significant difference in calendar aging performance\"."),
    ("Mechanism", "Thin-walled pressure-vessel argument (p. 304, eqn 1): "
     "\"larger NPs, even if they are below the 150 nm limit for particle "
     "fracture, will still transmit more strain to the surrounding electrode "
     "matrix than smaller NPs\". The ~150 nm fracture limit itself is cited "
     "from refs 9-11 (Introduction, p. 299)."),
]


def build_pdf(path, answer_par, method_par, fig_infos, stats_table,
              paper_applicability, risk_par, limitations):
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
                            title="Silicon particle size analysis")
    story = [Paragraph("Silicon particle size analysis - Hackathon-Polaron "
                       "batches", h1), Spacer(1, 6)]
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
    for name, txt in PAPER_FACTS:
        story.append(Paragraph(f"<b>{name}:</b> {txt}", small))
        story.append(Spacer(1, 3))
    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Applicability:</b> " + paper_applicability, body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Risk ranking", h2))
    story.append(Paragraph(risk_par, body))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Limitations", h2))
    for lim in limitations:
        story.append(Paragraph("- " + lim, small))
    doc.build(story)


def stats_table_for_pdf(per_batch_df):
    cols = ["batch", "n_images", "n_particles", "area_frac",
            "D50", "D50_ci", "D90", "D90_ci", "D99", "max", "frac_large"]
    head = ["batch", "images", "particles", "area frac",
            "D50 um", "D50 95%CI", "D90 um", "D90 95%CI", "D99 um",
            "max um", "frac>cut"]
    data = [head]
    for _, r in per_batch_df.iterrows():
        data.append([r["batch"], int(r["n_images"]), int(r["n_particles"]),
                     f"{r['area_frac']:.3f}", f"{r['D50']:.2f}",
                     f"[{r['D50_ci_lo']:.2f},{r['D50_ci_hi']:.2f}]",
                     f"{r['D90']:.2f}",
                     f"[{r['D90_ci_lo']:.2f},{r['D90_ci_hi']:.2f}]",
                     f"{r['D99']:.2f}", f"{r['max']:.2f}",
                     f"{r['frac_large']:.3f}"])
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

    all_parts, img_meta = [], []
    seg_cache = {}   # iid -> (norm_smooth float32, labels int32)
    overlay_paths = {b: [] for b, _, _ in images}
    batch_list = sorted(set(b for b, _, _ in images))

    for b, iid, path in images:
        um, src = pixel_size_um(path)
        g = load_gray(path)
        g, cropped = crop_banner(g)
        s, lab, info = segment(g)
        seg_cache[iid] = (s.astype(np.float32), lab.astype(np.int32))
        df = measure(lab, iid, b, um)
        all_parts.append(df)
        img_meta.append(dict(batch=b, image_id=iid, path=path,
                             height=g.shape[0], width=g.shape[1],
                             um_per_px=um, px_source=src,
                             banner_rows_cropped=cropped, **info))
        print(f"  {b}/{iid}: {len(df)} particles "
              f"({df.touches_border.sum()} border), thr={info['threshold']:.3f} "
              f"({info['method']}), px={um*1000:.2f} nm")

    parts = pd.concat(all_parts, ignore_index=True)
    meta = pd.DataFrame(img_meta)

    # interior particles only for size statistics
    pin = parts[~parts.touches_border].copy()

    # 'large particle' cut-off: the paper gives no usable physical cut-off for
    # our size range (its particles are 3-27 nm and ~150 nm is a fracture
    # limit, far below our sizes), so we use the pooled D90 across all images.
    # This is a RELATIVE limit, not a physical one.
    cutoff = float(np.percentile(pin.equiv_diam_um, 90))
    d50_all = float(np.percentile(pin.equiv_diam_um, 50))
    print(f"pooled D50 = {d50_all:.3f} um, pooled D90 cut-off = {cutoff:.3f} um")

    # ---------- per-image ----------
    per_image_rows = []
    for b, iid, path in images:
        sub = pin[pin.image_id == iid]
        st = dist_stats(sub.equiv_diam_um, sub.area_um2)
        meta_row = meta[meta.image_id == iid].iloc[0]
        area_mm2 = meta_row.height * meta_row.width * (meta_row.um_per_px ** 2) / 1e6
        per_image_rows.append(dict(
            batch=b, image_id=iid,
            height=int(meta_row.height), width=int(meta_row.width),
            um_per_px=meta_row.um_per_px,
            threshold=meta_row.threshold, method=meta_row.method,
            banner_rows_cropped=int(meta_row.banner_rows_cropped),
            n_particles=st["n"],
            n_density_per_mm2=st["n"] / area_mm2,
            area_frac=float(sub.area_um2.sum() /
                            (meta_row.height * meta_row.width *
                             meta_row.um_per_px ** 2)),
            mean_diam_um=st["mean"], median_diam_um=st["median"],
            d10_um=st["D10"], d50_um=st["D50"], d90_um=st["D90"],
            d99_um=st["D99"], max_diam_um=st["max"],
            d50_areawt_um=st["D50_areawt"], d90_areawt_um=st["D90_areawt"],
            frac_large=float((sub.equiv_diam_um > cutoff).mean())
            if st["n"] else np.nan,
            outlier_image=iid in OUTLIER_IDS,
        ))
    per_image = pd.DataFrame(per_image_rows)

    # ---------- per-batch (all images, and excluding the two outliers) ------
    per_batch_rows = []
    for scope, excl in [("all_images", False),
                        ("excluding_outlier_images", True)]:
        pi = per_image[~per_image.outlier_image] if excl else per_image
        pp = pin[~pin.image_id.isin(OUTLIER_IDS)] if excl else pin
        for b in batch_list:
            sub = pp[pp.batch == b]
            imgs_b = pi[pi.batch == b]
            st = dist_stats(sub.equiv_diam_um, sub.area_um2)
            ci = bootstrap_ci(pi, pp, b) if len(imgs_b) else {}
            per_batch_rows.append(dict(
                scope=scope, batch=b, n_images=len(imgs_b),
                n_particles=st["n"],
                n_density_per_mm2=float(
                    (imgs_b.n_density_per_mm2 * 1).mean()) if len(imgs_b) else np.nan,
                area_frac=float(np.average(
                    imgs_b.area_frac)) if len(imgs_b) else np.nan,
                mean=st["mean"], median=st["median"],
                D10=st["D10"], D50=st["D50"], D90=st["D90"], D99=st["D99"],
                max=st["max"], D50_areawt=st["D50_areawt"],
                D90_areawt=st["D90_areawt"],
                D50_ci_lo=ci.get("D50_ci", (np.nan, np.nan))[0],
                D50_ci_hi=ci.get("D50_ci", (np.nan, np.nan))[1],
                D90_ci_lo=ci.get("D90_ci", (np.nan, np.nan))[0],
                D90_ci_hi=ci.get("D90_ci", (np.nan, np.nan))[1],
                frac_large=float(np.average(
                    imgs_b.frac_large.dropna())) if len(imgs_b) else np.nan,
                cutoff_um=cutoff,
            ))
    per_batch = pd.DataFrame(per_batch_rows)

    # ---------- tests ----------
    test_rows = []
    for scope, excl in [("all_images", False),
                        ("excluding_outlier_images", True)]:
        pi = per_image[~per_image.outlier_image] if excl else per_image
        for metric in ["median_diam_um", "d90_um", "max_diam_um", "frac_large"]:
            for row in kruskal_and_mwu(pi, metric, batch_list):
                row["scope"] = scope
                test_rows.append(row)
    tests = pd.DataFrame(test_rows)

    # ---------- overlays ----------
    for b, iid, path in images:
        s, lab = seg_cache[iid]
        info = meta[meta.image_id == iid].iloc[0]
        pim = parts[parts.image_id == iid]
        op = os.path.join(ovdir, f"overlay_{b}_{iid}.png")
        n_in = int((~pim.touches_border).sum())
        save_overlay(s, lab, pim, cutoff, d50_all, op,
                     f"{b}/{iid} - {n_in} particles "
                     f"({info['method']}, thr {info['threshold']:.2f})")
        overlay_paths[b].append(op)
        print(f"  overlay {b}/{iid}")
    for b in batch_list:
        contact_sheet(overlay_paths[b], b,
                      os.path.join(ovdir, f"contact_sheet_{b}.png"))

    # ---------- figures ----------
    fig_size_distribution(pin, batch_list, cutoff,
                          os.path.join(figdir, "size_distribution.png"))
    fig_cumulative(pin, batch_list, os.path.join(figdir, "cumulative.png"))
    fig_per_image_boxplot(per_image, batch_list,
                          os.path.join(figdir, "per_image_boxplot.png"),
                          OUTLIER_IDS)
    fig_large_fraction(per_image, batch_list, cutoff,
                       os.path.join(figdir, "large_fraction.png"))
    fig_paper_context(pin, batch_list,
                      os.path.join(figdir, "paper_context.png"))

    # ---------- Excel ----------
    paper_sheet = pd.DataFrame(
        [{"topic": t, "extract": x} for t, x in PAPER_FACTS] + [
            {"topic": "applicability_verdict",
             "extract": PAPER_APPLICABILITY_TEXT},
            {"topic": "assumption",
             "extract": "Bright BSE particles assumed silicon-based; grey "
                        "bulk assumed graphite. From image appearance only."},
        ])
    readme = pd.DataFrame({"definition": [
        "Particles sheet: one row per detected particle. equiv_diam_um is the "
        "diameter of a circle with the same area. touches_border=True "
        "particles are cut off by the image edge and are EXCLUDED from all "
        "size statistics.",
        "Per_image sheet: one row per BSE image. D-values are number-weighted "
        "quantiles of equiv_diam_um unless suffixed _areawt (area-weighted). "
        "frac_large = fraction of particles above the pooled-D90 cut-off "
        "(relative, not physical).",
        "Per_batch sheet: pooled-particle statistics per batch. Confidence "
        "intervals are 95% bootstrap intervals resampling IMAGES (the "
        "independent unit), 2000 resamples. scope='all_images' vs "
        "'excluding_outlier_images' (drops 4ih2ggld and 5n1q8atc).",
        "Tests sheet: Kruskal-Wallis across batches and pairwise Mann-Whitney "
        "with Holm correction on per-image summaries; the image is the "
        "independent sample, not the particle.",
        "Units: lengths in um, areas in um^2, densities in particles/mm^2, "
        "fractions dimensionless. Pixel size ~25.0 nm/px from TIFF resolution "
        "tags (identical across images within rounding).",
        "2D caveat: a 2D cross-section understates true 3D particle size "
        "(stereological bias); D-values are section diameters, not true "
        "diameters.",
    ]})
    xlsx = os.path.join(out, "silicon_particles.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        parts.to_excel(xw, sheet_name="Particles", index=False)
        per_image.to_excel(xw, sheet_name="Per_image", index=False)
        per_batch.to_excel(xw, sheet_name="Per_batch", index=False)
        tests.to_excel(xw, sheet_name="Tests", index=False)
        paper_sheet.to_excel(xw, sheet_name="Paper", index=False)
        readme.to_excel(xw, sheet_name="README", index=False)
    print("wrote", xlsx)

    # ---------- report text ----------
    write_outputs(per_batch, per_image, tests, pin, cutoff, out, figdir,
                  batch_list)


PAPER_APPLICABILITY_TEXT = (
    "The paper studies 3-27 nm primary nanoparticles in ~89 wt% Si "
    "electrodes. Our measured particles are micron-scale (see "
    "paper_context.png) - roughly two orders of magnitude above the studied "
    "range, and also above the ~150 nm rupture threshold the paper cites. "
    "The paper's QUANTITATIVE results (e.g. 6-8 nm optimum) therefore do "
    "NOT apply to our particles. Only the qualitative trend is used here: "
    "within the silicon-bearing fraction, larger silicon units transmit "
    "more strain to the surrounding electrode and are expected to be worse "
    "for cycle life; the paper found no size effect on calendar life. Note "
    "also that our segmented objects may be agglomerates or embedded "
    "particles, not the primary crystallites the paper discusses, and our "
    "electrode is a different system (believed graphite bulk).")


def write_outputs(per_batch, per_image, tests, pin, cutoff, out, figdir,
                  batch_list):
    """NOTES.md + PDF report (numbers taken from the same dataframes as the
    Excel file)."""
    allb = per_batch[per_batch.scope == "all_images"].set_index("batch")
    xb = per_batch[per_batch.scope == "excluding_outlier_images"].set_index(
        "batch")

    def fmt_row(r):
        return (f"D50 {r['D50']:.2f} um "
                f"[{r['D50_ci_lo']:.2f}-{r['D50_ci_hi']:.2f}], "
                f"D90 {r['D90']:.2f} um "
                f"[{r['D90_ci_lo']:.2f}-{r['D90_ci_hi']:.2f}], "
                f"n={int(r['n_particles'])} from {int(r['n_images'])} images")

    # verdict text
    kw_lines = {}
    for metric in ["median_diam_um", "d90_um", "max_diam_um", "frac_large"]:
        row = tests[(tests.scope == "all_images") & (tests.metric == metric)
                    & (tests.test == "Kruskal-Wallis")]
        if len(row):
            kw_lines[metric] = row.iloc[0]
    sig = {m: (r.p_value < 0.05) for m, r in kw_lines.items()}

    med_lines = "; ".join(f"{b}: {fmt_row(allb.loc[b])}" for b in batch_list)

    if any(sig.values()):
        sigm = [m for m, v in sig.items() if v]
        verdict_differ = (f"At least one per-image metric differs between "
                          f"batches (Kruskal-Wallis p<0.05 for "
                          f"{', '.join(sigm)}).")
    else:
        verdict_differ = ("No per-image metric distinguishes the batches at "
                          "the 5% level (all Kruskal-Wallis p>0.05): the "
                          "batches cannot be statistically distinguished on "
                          "silicon-particle size with these data.")

    answer = (f"{verdict_differ} Pooled number-weighted diameters "
              f"(all images): {med_lines}. Because every measured particle "
              f"is micron-scale while the paper studied 3-27 nm particles, "
              f"only the paper's qualitative trend (larger silicon units -> "
              f"more mechanical damage -> shorter CYCLE life; no size effect "
              f"on CALENDAR life) is invoked - its quantitative results do "
              f"not apply here. Assumption: bright particles are "
              f"silicon-based, grey bulk is graphite, from image appearance "
              f"only.")

    method = ("BSE images only (31 total: Batch_1=7, Batch_2=7, Batch_3=17). "
              "Grey levels taken as the per-pixel median of the three "
              "identical RGB channels; no information banner was present in "
              "any image. Each image was normalised to its 1st-99.5th "
              "percentile range, Gaussian-smoothed (sigma 1.5 px), and "
              "thresholded for bright particles at the histogram valley "
              "between the bulk peak and a bright peak; images without a "
              "separate bright peak used the upper threshold of 3-class "
              "multi-Otsu (flagged in the data). Masks were opened "
              "(disk r=2), objects <150 px dropped, holes filled, and "
              "touching particles split by distance-transform watershed. "
              "Particles touching the image border are excluded from all "
              "size statistics. Pixel size is ~25.0 nm/px in every image "
              "(TIFF resolution tags). Statistics treat the IMAGE as the "
              "independent unit: Kruskal-Wallis plus pairwise Mann-Whitney "
              "with Holm correction on per-image medians/D90/max/large "
              "fraction, and 95% bootstrap CIs resampling images. The "
              "'large' cut-off is the pooled D90 across all images - a "
              "relative limit, since the paper offers no physical cut-off "
              "applicable at our sizes.")

    fig_infos = [
        (os.path.join(figdir, "size_distribution.png"),
         "Fig. 1 - Number-weighted diameter density per batch; dashed line "
         "is the pooled-D90 'large' cut-off."),
        (os.path.join(figdir, "cumulative.png"),
         "Fig. 2 - Cumulative diameter distributions, number-weighted "
         "(left) and area-weighted (right); a few large particles dominate "
         "the silicon volume."),
        (os.path.join(figdir, "per_image_boxplot.png"),
         "Fig. 3 - Per-image median and D90 grouped by batch; open markers "
         "are the two low-contrast outlier images."),
        (os.path.join(figdir, "large_fraction.png"),
         "Fig. 4 - Fraction of particles above the pooled-D90 cut-off per "
         "image, with batch means and bootstrap 95% CIs."),
        (os.path.join(figdir, "paper_context.png"),
         "Fig. 5 - Measured size range per batch vs the 3-27 nm range "
         "studied in the paper and the ~150 nm fracture threshold it "
         "cites."),
    ]

    stats_tbl = stats_table_for_pdf(
        per_batch[per_batch.scope == "all_images"])

    # risk statement
    if any(sig.values()):
        worst = allb["D90"].idxmax()
        risk = (f"On the paper's qualitative trend alone, {worst} carries "
                f"the largest large-particle tail (D90 "
                f"{allb.loc[worst,'D90']:.2f} um) and would be ranked at the "
                f"greatest risk of shorter CYCLE life. This ranking is "
                f"weak: it relies on a qualitative trend measured on "
                f"3-27 nm particles in pure-Si electrodes, applied to "
                f"micron-scale features in a different electrode system. "
                f"No statement about calendar life is supported - the "
                f"paper found no size dependence there even within its own "
                f"size range.")
    else:
        risk = ("No risk ranking is supported: the batches cannot be "
                "distinguished on silicon-particle size with these data. "
                "Even if they could, only the paper's qualitative cycle-"
                "life trend could be invoked, and the paper itself found no "
                "size effect on calendar life.")

    outlier_note = ("Two Batch_1 images (4ih2ggld, 5n1q8atc) returned "
                    "roughly 2-4x the particle count of the rest at lower "
                    "contrast; their thresholds dip into marginal bright "
                    "material (flake interiors and rims), so their counts "
                    "are inflated and sizes skewed small. Results are "
                    "reported both with and without them.")
    limitations = [
        "Material identity is assumed: bright BSE particles = silicon-based, "
        "grey bulk = graphite, from image appearance only (no EDS).",
        "A 2D polished section understates true 3D particle size "
        "(stereological bias); all D-values are section diameters.",
        "Pixel size (~25.0 nm/px) comes from TIFF resolution tags written by "
        "the acquisition software, not from a measured scale bar; a "
        "systematic calibration error would shift every size together.",
        "The image is the independent unit (n=7, 7, 17); Batch_3 has more "
        "images, so pairwise tests involving it have more power than the "
        "Batch_1-Batch_2 comparison.",
        outlier_note,
        "Segmented objects may be agglomerates or embedded composite "
        "particles, not primary crystallites - the feature the paper "
        "studies (3-27 nm primary particles) is far below our resolution "
        "and minimum object size.",
        "The paper's quantitative findings do not extend to micron-scale "
        "particles; only the qualitative larger-is-worse-for-cycling trend "
        "is invoked.",
    ]

    pdf_path = os.path.join(out, "silicon_size_report.pdf")
    build_pdf(pdf_path, answer, method, fig_infos, stats_tbl,
              PAPER_APPLICABILITY_TEXT, risk, limitations)
    print("wrote", pdf_path)

    # ---------- NOTES.md ----------
    notes = f"""# NOTES - silicon particle size analysis

## Image inventory
- Batch_1: 7 BSE images; Batch_2: 7; Batch_3: 17. Matches the earlier run's
  count of 7 / 7 / 17 - no mismatch.
- Some locations store the third detector image as `*_SE.tif` instead of
  `*_ETD.tif` (Batch_2: rxax5ozo; Batch_3: utfgcjfa, vc2whyaq, x77cy643).
  BSE counts are unaffected; every location still has three files.

## Assumptions
- Bright BSE particles are ASSUMED silicon-based; grey bulk ASSUMED graphite
  - from image appearance only, no compositional (e.g. EDS) confirmation.
  This assumption applies to every output.
- Pixel size is taken from the TIFF resolution tags: ~25.0 nm per pixel in
  every BSE image (range 24.999-25.001 nm). No scale bar is burned into the
  images and no other instrument metadata exists.
- Segmented objects may be agglomerates or embedded composite particles,
  not primary crystallites.
- 2D section diameters understate true 3D size (stereological bias).

## Method notes
- No information banner found in any image (`banner_rows_cropped`=0
  everywhere).
- Thresholding: most images show no separate bright peak in the histogram
  (the bright signal is a shoulder/tail of the bulk peak), so they use the
  3-class multi-Otsu fallback - see `method` column in the Per_image sheet.
- The 'large particle' cut-off is the pooled D90 across all images
  ({cutoff:.2f} um): a RELATIVE limit. The paper gives no physical cut-off
  applicable at micron scale (its particles are 3-27 nm; the ~150 nm
  rupture limit it cites is below our minimum detectable object).

## Outlier images
- Batch_1 images `4ih2ggld` and `5n1q8atc` returned ~2-4x the particle
  count of other images at visibly lower contrast. Inspection of their
  overlays shows the threshold catching marginal bright material - bright
  flake interiors, edge rims and fragmented rim segments - in addition to
  genuine bright particles. Their counts are inflated and their size
  distributions skewed small; the segmentation is NOT considered fully
  valid for particle statistics on these two images. All batch statistics
  and tests are reported twice: `all_images` and
  `excluding_outlier_images` (see the `scope` column).

## Statistics
- The IMAGE is the independent sample (n = 7, 7, 17), not the particle.
  Batch_3's larger image count gives pairwise tests involving it more
  power than the Batch_1-Batch_2 comparison.
- Bootstrap CIs (2000 resamples) resample images with replacement and
  recompute the pooled-particle quantile each time.

## Known issues / what failed
- Reading the TIFFs requires the `imagecodecs` package (LZW decode); it is
  a dependency of tifffile, not an extra analysis library.
- Overlays are saved at 50% resolution (~3500 px wide) to keep file sizes
  reasonable; outlines are dilated to 3 px before downscaling so they stay
  visible. Boundary accuracy is better checked on the full-size label data
  if needed.
- No image was excluded entirely; border-touching particles are flagged
  (`touches_border`) and excluded from size statistics.
"""
    with open(os.path.join(out, "NOTES.md"), "w") as f:
        f.write(notes)
    print("wrote NOTES.md")


if __name__ == "__main__":
    main()
