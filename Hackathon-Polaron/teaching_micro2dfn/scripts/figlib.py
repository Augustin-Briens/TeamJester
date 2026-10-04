"""Shared style + helpers for teaching_micro2dfn figures.

Phase colours match validated_comparison/panels/*.png exactly:
pore = near-black navy, bulk = grey, si candidate = orange,
uncertain bright = magenta.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
FIG = os.path.join(ROOT, "teaching_micro2dfn", "figures")
ANN = os.path.join(ROOT, "teaching_micro2dfn", "annotated_images")
os.makedirs(FIG, exist_ok=True)
os.makedirs(ANN, exist_ok=True)

# ---- palette -------------------------------------------------------------
C_PORE = "#1b1b30"
C_BULK = "#8d8d99"
C_SI = "#f08c1e"
C_UNCERT = "#e84393"
C_SI_EDGE = "#c56a00"

C_B1 = "#e4572e"
C_B2 = "#3a86ff"
C_B3 = "#2a9d8f"
C_NEW = "#8338ec"
BATCH_COLOR = {"Batch_1": C_B1, "Batch_2": C_B2, "Batch_3": C_B3,
               "New_Images_Batch": C_NEW}

C_INK = "#1d2433"
C_MUTE = "#5b6474"
C_ACCENT = "#0e6e6b"
C_WARN = "#b33939"
C_OK = "#2a9d8f"

SEG_CMAP = ListedColormap([C_PORE, C_BULK, C_SI])
SEG4_CMAP = ListedColormap([C_PORE, C_BULK, C_SI, C_UNCERT])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.edgecolor": "#aab2bf",
    "axes.linewidth": 0.8,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelcolor": C_INK,
    "text.color": C_INK,
    "xtick.color": C_MUTE,
    "ytick.color": C_MUTE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

PX_UM = 0.025


def recipe() -> dict:
    return json.load(open(os.path.join(
        ROOT, "validated_comparison", "recipe.json")))


def save(fig, name: str, subdir: str = "figures") -> str:
    out = os.path.join(FIG if subdir == "figures" else ANN, name)
    fig.savefig(out)
    plt.close(fig)
    print("  wrote", os.path.relpath(out, ROOT))
    return out


def scale_bar(ax, um: float, px_um: float, x_frac: float = 0.04,
              y_frac: float = 0.05, color: str = "white",
              height_frac: float = 0.012):
    """Verified scale bar: length in um converted to pixels (data coords).
    Verification: bar_px = um / px_um, drawn in data units."""
    x0 = x_frac * ax.get_xlim()[1]
    y0 = (1 - y_frac) * ax.get_ylim()[0] + y_frac * ax.get_ylim()[1] \
        if ax.get_ylim()[0] < ax.get_ylim()[1] else y_frac * ax.get_ylim()[0]
    # images use origin=upper -> y axis runs downward (ylim[0] > ylim[1])
    ytop, ybot = ax.get_ylim()
    y = ytop - y_frac * (ytop - ybot)
    bar_px = um / px_um
    h = height_frac * abs(ytop - ybot)
    ax.add_patch(plt.Rectangle((x0, y - h), bar_px, h,
                               facecolor=color, edgecolor="none", zorder=9))
    ax.text(x0 + bar_px / 2, y - h - 0.02 * abs(ytop - ybot), f"{um:g} µm",
            color=color, ha="center", va="bottom", fontsize=11,
            fontweight="bold", zorder=9)


def stage_strip(highlight: int, width: float = 12.4):
    """Recurring 'where we are' pipeline strip. highlight: 0-7 or -1."""
    stages = ["Acquire\nSEM images", "Correct\nflat-field",
              "Segment\nfrozen thresholds", "Classify\nobjects",
              "Measure\nmarkers", "Compare\nbatches",
              "Model\nDFN indicators", "Decide\naccept / act"]
    fig, ax = plt.subplots(figsize=(width, 0.62))
    ax.set_xlim(0, len(stages) * 1.12)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, s in enumerate(stages):
        x = i * 1.12
        on = i == highlight
        fc = C_ACCENT if on else "#eef1f5"
        tc = "white" if on else C_MUTE
        ax.add_patch(FancyBboxPatch((x + 0.02, 0.16), 0.92, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.08",
                     facecolor=fc, edgecolor=C_ACCENT if on else "#d5dbe4",
                     linewidth=1.2 if on else 0.7))
        ax.text(x + 0.48, 0.47, s, ha="center", va="center",
                fontsize=8.6, color=tc,
                fontweight="bold" if on else "normal", linespacing=1.0)
        if i < len(stages) - 1:
            ax.annotate("", xy=(x + 1.10, 0.47), xytext=(x + 0.95, 0.47),
                        arrowprops=dict(arrowstyle="-|>", color="#b7c0cc",
                                        lw=1.2))
    return fig


def tag_schematic(ax, text="SCHEMATIC — drawn for teaching, not measured"):
    ax.text(0.995, 0.015, text, transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8.5, color=C_WARN, style="italic")


def load_gray(image_id: str, batch: str) -> tuple[np.ndarray, np.ndarray]:
    """(raw uint8-as-float, flat-fielded) for one BSE image."""
    import sys
    sys.path.insert(0, ROOT)
    from vcompare.extract import Image
    path = os.path.join(ROOT, batch, f"{image_id}_BSE.tif")
    im = Image(path, batch)
    return im.raw, im.gray


def load_seg(image_id: str) -> np.ndarray:
    p = os.path.join(ROOT, "validated_comparison", "cache",
                     f"{image_id}_seg.npz")
    return np.load(p)["seg"]


def channel(image_id: str, batch: str, det: str) -> np.ndarray:
    import tifffile
    p = os.path.join(ROOT, batch, f"{image_id}_{det}.tif")
    img = tifffile.imread(p)
    if img.ndim == 3:
        img = img[..., 0]
    return np.ascontiguousarray(img[:, :-2])
