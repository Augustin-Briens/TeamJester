#!/usr/bin/env python3
"""Generate every figure for the micro2dfn masterclass.

All real-data figures come from the dataset TIFFs or saved pipeline
outputs.  Schematics are tagged SCHEMATIC.  Each printed line traces a
figure to its source data; crops and chosen objects are recorded in
annotated_images/annotations.csv.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Rectangle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figlib import (ANN, BATCH_COLOR, C_ACCENT, C_B1, C_B2, C_B3,
                    C_BULK, C_INK, C_MUTE, C_NEW, C_OK, C_PORE, C_SI,
                    C_SI_EDGE, C_UNCERT, C_WARN, FIG, PX_UM, ROOT,
                    SEG_CMAP, load_gray, load_seg, recipe, save,
                    scale_bar, stage_strip, tag_schematic, channel)

R = recipe()
T_PORE, T_SI, T_CORE = R["t_pore"], R["t_si"], R["t_core"]

REF_IMG, REF_BATCH = "img_hawkfj64", "Batch_3"   # nearest to B3 median vector
DISPUTED = [("img_4ih2ggld", (1292, 2316, 1343, 2367)),
            ("img_5n1q8atc", (671, 1695, 1619, 2643)),
            ("img_f1vzngrs", (0, 1024, 3444, 4468))]
REF_CROP = (441, 1465, 2987, 4011)               # centre 1024-px square

pif = pd.read_csv(os.path.join(ROOT, "validated_comparison",
                               "per_image_features.csv"))
pw = pd.read_csv(os.path.join(ROOT, "validated_comparison",
                              "pairwise_comparisons.csv"))
objs = pd.read_csv(os.path.join(ROOT, "validated_comparison",
                                "objects_classified.csv"))

ann_rows = []


def note(file, element, geometry, provenance, text):
    ann_rows.append(dict(file=file, element=element, geometry=geometry,
                         provenance=provenance, text=text))


# =====================================================================
# 1. recurring stage strips
# =====================================================================
for i in range(8):
    save(stage_strip(i), f"stage_{i}.png")

# =====================================================================
# 2. battery schematic (drawn)
# =====================================================================
def fig_cell():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 100, 6, fc="#b87333", ec="none"))
    ax.text(50, 3, "copper foil — electron collector", ha="center",
            va="center", color="white", fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((0, 6), 100, 30, fc="#eef0f3", ec="none"))
    rng = np.random.default_rng(4)
    for _ in range(46):  # graphite flakes
        x, y = rng.uniform(2, 98), rng.uniform(8, 34)
        ax.add_patch(Ellipse((x, y), rng.uniform(6, 13),
                     rng.uniform(1.2, 2.6), angle=rng.uniform(-15, 15),
                     fc=C_BULK, ec="#6f6f7a", lw=0.4))
    for _ in range(13):  # silicon particles
        x, y = rng.uniform(4, 96), rng.uniform(9, 33)
        ax.add_patch(Circle((x, y), rng.uniform(1.0, 1.8), fc=C_SI,
                            ec=C_SI_EDGE, lw=0.5))
    for _ in range(17):  # pores
        x, y = rng.uniform(4, 96), rng.uniform(8, 34)
        ax.add_patch(Ellipse((x, y), rng.uniform(2, 5),
                     rng.uniform(0.8, 1.8), angle=rng.uniform(-40, 40),
                     fc="#0f1420", ec="none"))
    ax.text(50, 21, "anode coating: graphite + silicon + binder + pores"
            " (pores carry electrolyte)", ha="center", va="center",
            fontsize=11, color=C_INK)
    ax.add_patch(Rectangle((0, 36), 100, 5, fc="#f6d365", ec="none",
                           alpha=0.9))
    ax.text(50, 38.5, "separator — passes Li⁺, blocks electrons",
            ha="center", va="center", fontsize=11, color=C_INK)
    ax.add_patch(Rectangle((0, 41), 100, 10, fc="#4c5e7a", ec="none"))
    ax.text(50, 46, "cathode (NMC) + aluminium foil", ha="center",
            va="center", color="white", fontsize=11, fontweight="bold")
    for x in (22, 50, 78):  # Li+ arrows through pores
        ax.annotate("", xy=(x, 11), xytext=(x, 40),
                    arrowprops=dict(arrowstyle="-|>", color="#117ab2",
                                    lw=2.2, linestyle="--"))
    ax.text(16, 26, "Li⁺ through electrolyte (pore network)",
            color="#117ab2", fontsize=10.5, rotation=90, va="center")
    for x in (35, 65):  # e- arrows in solid
        ax.annotate("", xy=(x, 5.5), xytext=(x, 34),
                    arrowprops=dict(arrowstyle="-|>", color=C_WARN,
                                    lw=2.2))
    ax.text(70, 27, "e⁻ through solid particles + binder matrix",
            color=C_WARN, fontsize=10.5, rotation=90, va="center")
    ax.annotate("", xy=(94, 6.5), xytext=(94, 33),
                arrowprops=dict(arrowstyle="-|>", color="#117ab2", lw=2))
    ax.text(96.5, 20, "Li⁺ inserts INTO particles", color="#117ab2",
            fontsize=10, rotation=90, va="center")
    tag_schematic(ax)
    return fig

save(fig_cell(), "fig_cell_schematic.png")

# =====================================================================
# 3. silicon swelling schematic + budget equation
# =====================================================================
def fig_swelling():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("a balloon inside the coating — needs room", fontsize=13)
    ax.add_patch(Circle((3.0, 5), 1.5, fc=C_SI, ec=C_SI_EDGE, lw=1.5))
    ax.add_patch(Circle((3.0, 5), 2.5, fill=False, ec=C_WARN, lw=1.6,
                        ls="--"))
    ax.annotate("", xy=(3.0, 7.5), xytext=(3.0, 6.5),
                arrowprops=dict(arrowstyle="<->", color=C_WARN, lw=1.4))
    ax.text(3.05, 5, "Si", ha="center", va="center", color="white",
            fontsize=15, fontweight="bold")
    ax.text(5.6, 7.15, "+~280% volume\nwhen full of lithium",
            fontsize=10.5, color=C_WARN)
    ax.add_patch(Circle((8.0, 5), 1.5, fc=C_SI, ec=C_SI_EDGE, lw=1.5))
    ax.add_patch(Ellipse((8.0, 5), 5.6, 4.4, fc="none", ec=C_PORE, lw=10,
                         alpha=0.25))
    ax.text(8.0, 5, "Si", ha="center", va="center", color="white",
            fontsize=15, fontweight="bold")
    ax.text(8.0, 8.6, "pore space = the room it can grow into",
            ha="center", fontsize=10.5, color=C_PORE)
    tag_schematic(ax, "SCHEMATIC — expansion value from literature, "
                      "not measured here")
    ax = axes[1]
    ax.axis("off")
    ax.set_title("our swelling budget = room minus demand", fontsize=13)
    ax.text(0.02, 0.86, "budget = pore_frac − 2.8·Si_frac − 0.1·bulk_frac",
            fontsize=13.5, family="Helvetica", color=C_INK)
    ax.text(0.02, 0.66, "2.8 ≈ silicon's +280% volume change (Li15Si4 "
                        "literature value)\n0.1 ≈ graphite's +10%",
            fontsize=11.5, color=C_MUTE)
    ax.text(0.02, 0.40, "budget < 0  →  expansion demand exceeds the "
                        "visible pore space\n"
                        "→ coating must thicken, crack, or push particles "
                        "apart", fontsize=11.5, color=C_WARN)
    ax.text(0.02, 0.14, "Caveat: it re-uses the same pore & silicon "
                        "measurements —\nit is a derived view, NOT "
                        "independent confirmation.",
            fontsize=11, color=C_MUTE, style="italic")
    return fig

save(fig_swelling(), "fig_si_swelling.png")

# =====================================================================
# 4. dataset tree
# =====================================================================
def fig_tree():
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    rows = [("batch", "a production shipment of anode-coated foil\n"
                     "(Batch_1, Batch_2, Batch_3 + New_Images)", 44),
            ("field of view", "one imaged location on one specimen —\n"
             "34 total (7 / 7 / 17 / 3); the INDEPENDENT unit", 34),
            ("channel", "same FOV re-imaged by a different detector\n"
             "(BSE + InLens + ETD or SE) — NOT new specimens", 24),
            ("crop", "a sub-window of one image used for teaching/\n"
             "inspection — pixels share the same photo", 14),
            ("object", "one segmented particle or pore\n"
             "(31,515 bright objects total) — never independent", 4)]
    w = 30
    for i, (name, desc, y) in enumerate(rows):
        x = 4 + i * 4
        ax.add_patch(FancyBboxPatch((x, y - 3.4), w, 6.6,
                     boxstyle="round,pad=0.15,rounding_size=0.6",
                     fc="#eef1f5", ec=C_ACCENT, lw=1.4))
        ax.text(x + w / 2, y + 1.6, name, ha="center", fontsize=13,
                fontweight="bold", color=C_ACCENT)
        ax.text(x + w / 2, y - 1.3, desc, ha="center", va="center",
                fontsize=9.2, color=C_INK)
        if i < 4:
            ax.annotate("", xy=(x + 4 + w + 4, y - 6.8),
                        xytext=(x + w / 2, y - 3.6),
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTE,
                                        lw=1.6))
    ax.text(50, 48.6, "What counts as a new piece of evidence?",
            ha="center", fontsize=15, fontweight="bold", color=C_INK)
    return fig

save(fig_tree(), "fig_dataset_tree.png")

# =====================================================================
# 5. inventory chart
# =====================================================================
def fig_inventory():
    man = pd.read_csv(os.path.join(ROOT, "teaching_micro2dfn",
                                   "dataset_manifest.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                           gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    batches = ["Batch_1", "Batch_2", "Batch_3", "New_Images_Batch"]
    counts = [len(man[man.batch == b]) for b in batches]
    bars = ax.bar(range(4), counts,
                  color=[BATCH_COLOR[b] for b in batches], width=0.62)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.3, str(c),
                ha="center", fontsize=14, fontweight="bold", color=C_INK)
    ax.set_xticks(range(4), [b.replace("_", " ").replace("New Images Batch",
                             "New images\n(batch unknown)")
                             for b in batches], fontsize=10)
    ax.set_ylabel("fields of view (BSE)")
    ax.set_ylim(0, 19)
    ax.set_title("34 fields of view — each ~175 × 40–58 µm", fontsize=13)
    ax = axes[1]
    sec = [int(man.has_etd.sum()), int(man.has_se.sum()),
           int(man.has_inlens.sum())]
    ax.barh([2, 1, 0], sec, color=[C_BULK, "#6f6f7a", C_ACCENT],
            height=0.55)
    ax.set_yticks([2, 1, 0], ["ETD (everhart–thornley)", "SE", "InLens"])
    for y, v in zip([2, 1, 0], sec):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=12, color=C_INK)
    ax.set_xlim(0, 37)
    ax.set_xlabel("fields of view with that second channel")
    ax.set_title("second channel = same location, different signal",
                 fontsize=13)
    return fig

save(fig_inventory(), "fig_inventory.png")

# =====================================================================
# 6. DFN schematic
# =====================================================================
def fig_dfn():
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    ax.add_patch(Rectangle((4, 6), 26, 34, fc="#f7efe4", ec=C_SI_EDGE,
                           lw=1.4))
    ax.text(17, 43, "ANODE (this is what we image)", ha="center",
            fontsize=11, fontweight="bold", color=C_INK)
    ax.add_patch(Rectangle((36, 6), 8, 34, fc="#f6d365", ec="none"))
    ax.text(40, 43, "separator", ha="center", fontsize=10, color=C_INK)
    ax.add_patch(Rectangle((50, 6), 26, 34, fc="#e8ecf4", ec="#4c5e7a",
                           lw=1.4))
    ax.text(63, 43, "CATHODE (NMC)", ha="center", fontsize=11,
            fontweight="bold", color=C_INK)
    ax.add_patch(Circle((14, 18), 5.5, fc=C_BULK, ec="#6f6f7a"))
    ax.add_patch(Circle((14, 18), 2.0, fc="#6f6f7a"))
    ax.text(14, 18, "r", ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.annotate("", xy=(19.5, 18), xytext=(8.5, 18),
                arrowprops=dict(arrowstyle="<->", color=C_INK, lw=1))
    ax.text(14, 9.5, "graphite particle\nLi⁺ diffuses in r",
            ha="center", fontsize=9, color=C_INK)
    ax.add_patch(Circle((24, 30), 3.6, fc=C_SI, ec=C_SI_EDGE))
    ax.text(24, 30, "r", ha="center", va="center", color="white",
            fontsize=12, fontweight="bold")
    ax.text(24, 24.5, "silicon particle\n(secondary phase)", ha="center",
            fontsize=9, color=C_INK)
    for y in (12, 22, 32):
        ax.annotate("", xy=(74, y), xytext=(6, y),
                    arrowprops=dict(arrowstyle="-|>", color="#117ab2",
                                    lw=1.6, linestyle="--"))
    ax.text(40, 3.0, "x — Li⁺ travels through electrolyte in the pores",
            ha="center", fontsize=10, color="#117ab2")
    ax.annotate("", xy=(80, 46), xytext=(2, 46),
                arrowprops=dict(arrowstyle="-|>", color=C_WARN, lw=1.6))
    ax.text(41, 48.2, "electrons travel in the solid, opposite way",
            ha="center", fontsize=10, color=C_WARN)
    ax.text(90, 34, "the model solves, at every x and t:", fontsize=11,
            fontweight="bold", color=C_INK)
    ax.text(90, 20, "• Li⁺ concentration\n   in electrolyte\n"
                    "• Li inside each\n   particle (r)\n"
                    "• potentials of\n   solid & electrolyte\n"
                    "• SEI growth\n• particle swelling\n   & stress",
            fontsize=10, va="center", color=C_MUTE)
    ax.set_title("Doyle–Fuller–Newman: the electrode as linked "
                 "continua — not individual particles",
                 fontsize=13.5, color=C_INK, loc="left")
    tag_schematic(ax)
    return fig

save(fig_dfn(), "fig_dfn_schematic.png")

# =====================================================================
# 7. reference image — raw with crop box
# =====================================================================
raw_ref, gray_ref = load_gray(REF_IMG, REF_BATCH)
seg_ref = load_seg(REF_IMG)
y0, y1, x0, x1 = REF_CROP

fig, ax = plt.subplots(figsize=(12.2, 4.6))
ax.imshow(raw_ref, cmap="gray", vmin=0, vmax=160, interpolation="none")
ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                       ec=C_WARN, lw=1.8))
ax.text(x0 + 14, y0 - 45, "teaching crop — every step-by-step view "
        "uses this window", color=C_WARN, fontsize=12, fontweight="bold",
        bbox=dict(fc="white", ec="none", alpha=0.82, pad=2.5))
scale_bar(ax, 25, PX_UM)
ax.set_title(f"{REF_IMG} — Batch_3 reference field of view "
             f"(raw BSE, untouched)", fontsize=13)
ax.set_xlabel("x (px, 25 nm each)"); ax.set_ylabel("y (px)")
note("fig_ref_raw", "crop rectangle", f"y[{y0}:{y1}] x[{x0}:{x1}]",
     "algorithm", "centre 1024x1024 crop; rule: deterministic centre "
     "crop of the median-typical reference image")
save(fig, "fig_ref_raw.png")

# ---- per-stage crop assets (annotated_images) ------------------------
def crop_save(arr, name, cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(arr[y0:y1, x0:x1], cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="none")
    ax.axis("off")
    out = os.path.join(ANN, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  wrote", os.path.relpath(out, ROOT))

crop_save(raw_ref, "ref_raw_crop.png", vmin=0, vmax=160)
crop_save(gray_ref, "ref_flat_crop.png", vmin=0.6, vmax=1.7)
crop_save(seg_ref, "ref_seg_crop.png", cmap=SEG_CMAP, vmin=0, vmax=2)

# =====================================================================
# 8. channels on img_f1vzngrs (normal contrast, all 3 channels)
# =====================================================================
def fig_channels():
    cid = "img_f1vzngrs"
    cy0, cy1, cx0, cx1 = 600, 1400, 3600, 4400
    bse = channel(cid, "Batch_1", "BSE")
    inl = channel(cid, "Batch_1", "Inlens")
    etd = channel(cid, "Batch_1", "ETD")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    for ax, im, name, note_ in zip(
            axes, (bse, inl, etd),
            ("BSE — backscattered electrons",
             "InLens — in-lens secondary electrons",
             "ETD — Everhart–Thornley SE"),
            ("brightness ~ mean atomic number\n"
             "(heavy atoms scatter more) —\nwhat we segment",
             "very surface-sensitive:\ntopography, fine cracks —\n"
             "acquisition-sensitive",
             "surface topography,\nslightly deeper view —\n"
             "texture comparison")):
        ax.imshow(im[cy0:cy1, cx0:cx1], cmap="gray", vmin=0, vmax=160,
                  interpolation="none")
        ax.set_title(name, fontsize=11.5)
        ax.text(0.5, -0.16, note_, transform=ax.transAxes, ha="center",
                fontsize=9.5, color=C_MUTE)
        ax.axis("off")
        scale_bar(ax, 10, PX_UM, color="white")
    fig.suptitle(f"{cid} (Batch_1) — the SAME field of view seen by "
                 f"three detectors; crop y[{cy0}:{cy1}] x[{cx0}:{cx1}]",
                 fontsize=13, fontweight="bold")
    note("fig_channels", "crop", f"y[{cy0}:{cy1}] x[{cx0}:{cx1}]",
         "algorithm", "identical crop on 3 registered channels")
    return fig

save(fig_channels(), "fig_channels.png")

# =====================================================================
# 9. flat-field correction
# =====================================================================
def fig_flatfield():
    row = raw_ref.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4),
                           gridspec_kw={"width_ratios": [1, 1, 0.75]})
    axes[0].imshow(raw_ref, cmap="gray", vmin=0, vmax=160,
                   interpolation="none")
    axes[0].set_title("raw", fontsize=12)
    axes[0].axhline(row, color=C_WARN, lw=0.8)
    axes[0].axis("off")
    axes[1].imshow(gray_ref, cmap="gray", vmin=0.6, vmax=1.7,
                   interpolation="none")
    axes[1].set_title("flat-field corrected", fontsize=12)
    axes[1].axhline(row, color=C_WARN, lw=0.8)
    axes[1].axis("off")
    ax = axes[2]
    ax.plot(raw_ref[row], np.arange(raw_ref.shape[1]),
            color=C_MUTE, lw=0.8, label="raw row")
    ax.plot(gray_ref[row] * 100, np.arange(gray_ref.shape[1]),
            color=C_ACCENT, lw=0.8, label="corrected (×100)")
    ax.set_title("brightness along the red row", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlabel("gray value"); ax.set_yticks([])
    fig.suptitle("Step — remove slow illumination drift so one ruler "
                 "applies across the frame", fontsize=13,
                 fontweight="bold")
    return fig

save(fig_flatfield(), "fig_flatfield.png")

# =====================================================================
# 10. histogram with thresholds (pooled Batch_3)
# =====================================================================
def fig_histogram():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    b3 = pif[pif.batch == "Batch_3"].image_id.tolist()
    samples = []
    for iid in b3:
        _, g = load_gray(iid, "Batch_3")
        sm = gaussian_filter(g, 1.5).ravel()
        samples.append(rng.choice(sm, 40_000, replace=False))
        del g, sm
    pooled = np.concatenate(samples)
    fig, ax = plt.subplots(figsize=(11.6, 4.6))
    n, bins, patches = ax.hist(pooled, bins=300, color="#c8cdd6",
                             edgecolor="none")
    for p, b in zip(patches, bins[:-1]):
        if b < T_PORE:
            p.set_facecolor(C_PORE)
        elif b >= T_SI:
            p.set_facecolor(C_SI)
        else:
            p.set_facecolor(C_BULK)
    for t, lab, c in ((T_PORE, f"t_pore = {T_PORE:.3f}\nbelow → pore",
                       C_PORE),
                      (T_SI, f"t_si = {T_SI:.3f}\nabove → bright", C_SI_EDGE),
                      (T_CORE, f"t_core = {T_CORE:.3f}\ninterior test for "
                       "'real particle'", C_UNCERT)):
        ax.axvline(t, color=c, lw=2)
        ymax = ax.get_ylim()[1]
        ax.text(t, ymax * (0.82 if t != T_CORE else 0.58), lab,
                rotation=90, va="top", ha="right", fontsize=10, color=c,
                fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("flat-fielded brightness (bulk ≈ 1.0)")
    ax.set_ylabel("pixels (log)")
    ax.set_title("multi-Otsu picks the two cuts that best split the "
                 "pooled Batch_3 histogram into 3 classes\n"
                 "fitted ONCE on the reference batch and frozen — "
                 "the same ruler for every batch", fontsize=12.5)
    return fig

save(fig_histogram(), "fig_histogram.png")

# =====================================================================
# 11. segmentation map + speckle removal
# =====================================================================
def fig_seg_steps():
    from scipy.ndimage import gaussian_filter
    from skimage.measure import label
    sm = gaussian_filter(gray_ref, 1.5)
    rough = np.ones(sm.shape, np.uint8)
    rough[sm < T_PORE] = 0
    rough[sm >= T_SI] = 2
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    axes[0].imshow(gray_ref[y0:y1, x0:x1], cmap="gray", vmin=0.6, vmax=1.7,
                   interpolation="none")
    axes[0].set_title("flat-fielded crop", fontsize=12)
    axes[1].imshow(rough[y0:y1, x0:x1], cmap=SEG_CMAP, vmin=0, vmax=2,
                   interpolation="none")
    n_sp = int((label(rough == 2).max() -
                label(seg_ref == 2).max()))
    axes[1].set_title("thresholded\n(speckle still in)", fontsize=12)
    axes[2].imshow(seg_ref[y0:y1, x0:x1], cmap=SEG_CMAP, vmin=0, vmax=2,
                   interpolation="none")
    axes[2].set_title("after size floor\n"
                      "(bright<150px, pore<20px → bulk)",
                      fontsize=11.5)
    for ax in axes:
        ax.axis("off")
        scale_bar(ax, 10, PX_UM)
    fig.suptitle("one ruler, two cuts, then a size floor — same rule "
                 "for all 34 images", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

save(fig_seg_steps(), "fig_seg_steps.png")

# =====================================================================
# 12. watershed split
# =====================================================================
def fig_watershed():
    from scipy.ndimage import distance_transform_edt
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.segmentation import watershed
    # window with most si objects (documented rule)
    best, best_n = None, -1
    sub = objs[(objs.image_id == REF_IMG)]
    for cy in range(0, seg_ref.shape[0] - 512, 256):
        for cx in range(0, seg_ref.shape[1] - 512, 256):
            m = ((sub.centroid_y_um / PX_UM >= cy)
                 & (sub.centroid_y_um / PX_UM < cy + 512)
                 & (sub.centroid_x_um / PX_UM >= cx)
                 & (sub.centroid_x_um / PX_UM < cx + 512))
            if m.sum() > best_n:
                best_n, best = m.sum(), (cy, cx)
    cy, cx = best
    wy0, wy1, wx0, wx1 = cy, cy + 512, cx, cx + 512
    si = (seg_ref[wy0:wy1, wx0:wx1] == 2)
    lab_plain = label(si)
    dist = distance_transform_edt(si)
    coords = peak_local_max(dist, min_distance=8, labels=si)
    mk = np.zeros(si.shape, np.int32)
    mk[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    ws = watershed(-dist, mk, mask=si)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6))
    axes[0].imshow(raw_ref[wy0:wy1, wx0:wx1], cmap="gray", vmin=0,
                   vmax=160, interpolation="none")
    axes[0].set_title("raw window\n(rule: most Si objects)", fontsize=11.5)
    axes[1].imshow(lab_plain, cmap="tab20", interpolation="none")
    axes[1].set_title(f"before watershed:\n{lab_plain.max()} blobs — "
                      "touching particles merge", fontsize=11.5)
    axes[2].imshow(np.where(si, ws, 0), cmap="tab20", interpolation="none")
    axes[2].set_title(f"after watershed:\n{ws.max()} objects — split at "
                      "the waist", fontsize=11.5)
    for ax in axes:
        ax.axis("off")
        scale_bar(ax, 5, PX_UM)
    fig.suptitle(f"{REF_IMG} window y[{wy0}:{wy1}] x[{wx0}:{wx1}] — "
                 "watershed can over-split; check borders before "
                 "trusting sizes", fontsize=12.5, fontweight="bold")
    note("fig_watershed", "window", f"y[{wy0}:{wy1}] x[{wx0}:{wx1}]",
         "algorithm", "512px window maximising si_particle count")
    return fig

save(fig_watershed(), "fig_watershed.png")

# =====================================================================
# 13. object classifier
# =====================================================================
def fig_classifier():
    sub = objs[objs.image_id == REF_IMG]
    in_crop = sub[(sub.centroid_y_um / PX_UM >= y0)
                  & (sub.centroid_y_um / PX_UM < y1)
                  & (sub.centroid_x_um / PX_UM >= x0)
                  & (sub.centroid_x_um / PX_UM < x1)]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0),
                           gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.imshow(raw_ref[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=160,
              interpolation="none")
    lab_img = None
    from vcompare.extract import split_bright, px_params
    lab_img = split_bright(seg_ref == 2, px_params(PX_UM))
    for _, r in in_crop.iterrows():
        m = lab_img[y0:y1, x0:x1] == r.label
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        col = C_SI if r.kind == "si_particle" else C_UNCERT
        ax.plot(xs, ys, ".", color=col, ms=0.4, alpha=0.85)
    # borderline object: closest interior_median to t_core in crop
    bl = in_crop.iloc[(in_crop.interior_median - T_CORE).abs().argmin()]
    by, bx = bl.centroid_y_um / PX_UM - y0, bl.centroid_x_um / PX_UM - x0
    ax.add_patch(Circle((bx, by), 30, fill=False, ec="white", lw=1.6))
    ax.annotate("borderline object\n"
                f"interior={bl.interior_median:.2f} vs t_core={T_CORE:.2f}\n"
                f"solidity={bl.solidity:.2f} → {bl.kind}",
                xy=(bx, by), xytext=(bx - 330, by - 120),
                color="white", fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.2))
    ax.set_title("crop — every bright object becomes\n"
                 "si_particle (orange) or uncertain bright (magenta)",
                 fontsize=11.5)
    ax.axis("off")
    scale_bar(ax, 10, PX_UM)
    ax = axes[1]
    for kind, c in (("si_particle", C_SI), ("bright_fine", C_UNCERT)):
        d = sub[sub.kind == kind]
        ax.scatter(d.solidity, d.interior_median, s=4, alpha=0.35,
                   color=c, label=f"{kind} (n={len(d)})")
    ax.axhline(T_CORE, color=C_UNCERT, lw=1.6, ls="--")
    ax.axvline(0.75, color=C_MUTE, lw=1.2, ls=":")
    ax.scatter(bl.solidity, bl.interior_median, s=130, facecolor="none",
               edgecolor=C_INK, lw=1.8, zorder=5)
    ax.annotate("the circled object", xy=(bl.solidity, bl.interior_median),
                xytext=(0.77, bl.interior_median + 0.25), fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=C_INK))
    ax.text(0.78, T_CORE + 0.05,
            f"t_core = {T_CORE:.3f} = 5th percentile of clear-particle "
            "interiors", fontsize=9.5, color=C_UNCERT)
    ax.set_xlabel("solidity (1 = smooth, compact)")
    ax.set_ylabel("interior brightness (flat-fielded)")
    ax.set_title("the rule — bright AND compact AND bright enough\n"
                 "inside; it is a geometric+brightness label, "
                 "NOT chemistry", fontsize=11.5)
    ax.legend(fontsize=9, markerscale=3, loc="upper left")
    note("fig_classifier", "borderline object",
         f"label={bl.label} centroid_px=({bl.centroid_y_um/PX_UM:.0f},"
         f"{bl.centroid_x_um/PX_UM:.0f})",
         "agent-interpretation",
         f"interior {bl.interior_median:.3f} vs t_core {T_CORE:.3f}; "
         f"solidity {bl.solidity:.2f}; classified {bl.kind} — PROVISIONAL")
    return fig

save(fig_classifier(), "fig_classifier.png")

# =====================================================================
# 14. annotated measurement on the reference crop
# =====================================================================
def fig_measure():
    from scipy.ndimage import distance_transform_edt
    from skimage.measure import label
    sub = objs[(objs.image_id == REF_IMG) & (objs.kind == "si_particle")]
    in_crop = sub[(sub.centroid_y_um / PX_UM >= y0 + 80)
                  & (sub.centroid_y_um / PX_UM < y1 - 80)
                  & (sub.centroid_x_um / PX_UM >= x0 + 80)
                  & (sub.centroid_x_um / PX_UM < x1 - 80)
                  & (~sub.touches_border)]
    # teaching choice: a clearly visible ~3 um particle
    p = in_crop.iloc[(in_crop.eq_diam_um - 3.0).abs().argmin()]
    py_px, px_px = p.centroid_y_um / PX_UM, p.centroid_x_um / PX_UM
    r_px = p.eq_diam_um / PX_UM / 2
    pore = seg_ref == 0
    dist, inds = distance_transform_edt(~pore, return_indices=True)
    si_mask = seg_ref == 2
    from vcompare.extract import split_bright, px_params
    lab_img = split_bright(si_mask, px_params(PX_UM))
    pm = lab_img == p.label
    bnd = pm & ~np.roll(pm, 1, 0)
    bys, bxs = np.where(bnd & (dist > 0))
    pore_px = (0, 0)
    dist_um = 0.0
    if len(bys):
        i = np.argmin(dist[bys, bxs])
        pore_px = (int(inds[0, bys[i], bxs[i]]),
                   int(inds[1, bys[i], bxs[i]]))
        dist_um = float(dist[bys[i], bxs[i]]) * PX_UM
    pore_lab = label(pore)
    cnts = np.bincount(pore_lab.ravel())
    big = np.argmax(cnts[1:]) + 1
    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    ax.imshow(raw_ref[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=160,
              interpolation="none")
    big_m = pore_lab[y0:y1, x0:x1] == big
    ys, xs = np.where(big_m)
    ax.plot(xs, ys, ".", color="#29e6e6", ms=0.3, alpha=0.5)
    cy, cx = py_px - y0, px_px - x0
    ax.add_patch(Circle((cx, cy), r_px, fill=False, ec="#29e6e6", lw=1.8))
    ax.annotate("", xy=(cx + r_px, cy + r_px * 0.72),
                xytext=(cx - r_px, cy + r_px * 0.72),
                arrowprops=dict(arrowstyle="<->", color="#29e6e6", lw=1.4))
    ax.text(cx, cy + r_px * 0.72 + 26,
            f"eq. diameter = {p.eq_diam_um:.2f} µm\n"
            f"(√(4A/π), A={p.area_um2:.1f} µm²)",
            color="#29e6e6", fontsize=10, ha="center", fontweight="bold")
    if len(bys):
        ax.annotate("", xy=(pore_px[1] - x0, pore_px[0] - y0),
                    xytext=(bxs[i] - x0, bys[i] - y0),
                    arrowprops=dict(arrowstyle="->", color="#ffe066",
                                    lw=1.8))
        ax.text((pore_px[1] + bxs[i]) / 2 - x0 - 60,
                (pore_px[0] + bys[i]) / 2 - y0 - 40,
                f"nearest pore\n≈{dist_um:.2f} µm", color="#ffe066",
                fontsize=10, fontweight="bold")
    ax.text(30, 60, "cyan speckle = largest connected pore region\n"
            "(its share of pore area = pore_largest_region_frac)",
            color="#29e6e6", fontsize=10)
    scale_bar(ax, 10, PX_UM, color="white")
    ax.set_title(f"{REF_IMG} crop — worked measurements "
                 "(agent annotations over raw pixels)", fontsize=12)
    ax.axis("off")
    note("fig_measure", "eq-diam circle + dimension line",
         f"centroid_px=({py_px:.0f},{px_px:.0f}) r_px={r_px:.0f}",
         "agent-interpretation",
         f"eq_diam {p.eq_diam_um:.2f} um — PROVISIONAL")
    note("fig_measure", "nearest-pore arrow",
         f"to px {pore_px}", "algorithm",
         f"distance-transform nearest pore ≈{dist_um:.2f} um")
    note("fig_measure", "largest-pore overlay", f"label={big}",
         "algorithm", "largest connected pore component (cyan speckle)")
    return fig

save(fig_measure(), "fig_measure.png")

# =====================================================================
# 15. the three disputed Batch_1 images
# =====================================================================
def fig_disputed3():
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    stats = pif.set_index("image_id")
    for ax, (iid, (cy0, cy1, cx0, cx1)) in zip(axes, DISPUTED):
        raw, _ = load_gray(iid, "Batch_1")
        ax.imshow(raw[cy0:cy1, cx0:cx1], cmap="gray", vmin=0, vmax=160,
                  interpolation="none")
        s = stats.loc[iid]
        ax.set_title(f"{iid}\nSi:bulk contrast={s.si_bulk_contrast:.2f}  "
                     f"bright={s.bright_frac_raw*100:.1f}%", fontsize=10.5)
        ax.axis("off")
        scale_bar(ax, 10, PX_UM)
        note("fig_disputed3", "crop", f"{iid} y[{cy0}:{cy1}] "
              f"x[{cx0}:{cx1}]", "algorithm",
              "same 1024px crops as expert_review_sheet.csv")
    fig.suptitle("the three flagged Batch_1 regions — extra bright "
                 "material; img_f1vzngrs has NORMAL contrast",
                 fontsize=13, fontweight="bold")
    return fig

save(fig_disputed3(), "fig_disputed3.png")

# =====================================================================
# 16. frozen vs floating ruler
# =====================================================================
def fig_frozen_floating():
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_multiotsu
    rng = np.random.default_rng(0)
    rows = []
    for _, r in pif.iterrows():
        _, g = load_gray(r.image_id, r.batch)
        sm = gaussian_filter(g, 1.5).ravel()
        s = rng.choice(sm, 60_000, replace=False)
        t1, t2 = threshold_multiotsu(s, classes=3, nbins=256)
        rows.append(dict(image_id=r.image_id, batch=r.batch,
                         t_si_float=t2,
                         bright_float=float((sm >= t2).mean()),
                         bright_frozen=float((sm >= T_SI).mean())))
        del g, sm
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax = axes[0]
    for b, c in BATCH_COLOR.items():
        if b == "New_Images_Batch":
            continue
        d = df[df.batch == b]
        ax.scatter(d.bright_frozen, d.bright_float,
                   color=c, label=b, s=42, edgecolor="white", lw=0.5)
    lim = (0, max(df.bright_float.max(), df.bright_frozen.max()) * 1.12)
    ax.plot(lim, lim, ls="--", color=C_MUTE, lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("bright fraction — frozen shared ruler")
    ax.set_ylabel("bright fraction — own floating ruler")
    ax.set_title("if each photo cut its own thresholds…", fontsize=12)
    ax.legend(fontsize=10)
    ax = axes[1]
    d = df.sort_values("bright_float")
    ax.scatter(range(len(d)), d.bright_frozen, color=C_BULK, s=34,
               label="frozen ruler")
    ax.scatter(range(len(d)), d.bright_float, color=C_UNCERT, s=34,
               label="floating ruler")
    for i, (_, rr) in enumerate(d.iterrows()):
        if rr.image_id in ("img_4ih2ggld", "img_5n1q8atc"):
            ax.annotate(rr.image_id.replace("img_", ""),
                        xy=(i, rr.bright_float), xytext=(i - 4.6,
                        rr.bright_float - 0.012), fontsize=9,
                        color=C_WARN)
    ax.set_xticks([]); ax.set_xlabel("31 images, sorted by floating value")
    ax.set_ylabel("bright fraction")
    ax.set_title("…the floating ruler re-counts the disputed photos\n"
                 "(20.7% vs 11.9% bright) — absorbed or amplified, "
                 "never neutral", fontsize=11.5)
    ax.legend(fontsize=10)
    fig.suptitle("the frozen ruler (fitted on Batch_3) is the measuring "
                 "instrument — the floating ruler is a diagnostic only",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig

save(fig_frozen_floating(), "fig_frozen_floating.png")

# =====================================================================
# 17. 9-feature dot grid
# =====================================================================
def fig_dotgrid():
    feats = [("si_candidate_frac", "Si candidate fraction"),
             ("uncertain_bright_frac", "uncertain-bright fraction"),
             ("si_d50_um", "Si D50 (µm)"),
             ("si_d90_um", "Si D90 (µm)"),
             ("si_clustering_R", "Si clustering R"),
             ("pore_frac", "resolved pore fraction"),
             ("pores_per_mpx", "pores per Mpx"),
             ("pores_per_mpx_noise_adj", "pores per Mpx (noise-adj)"),
             ("pore_anisotropy", "pore anisotropy (h/v)")]
    flagged = {"img_4ih2ggld", "img_5n1q8atc", "img_f1vzngrs"}
    fig, axes = plt.subplots(3, 3, figsize=(12.6, 9.4))
    for ax, (f, name) in zip(axes.ravel(), feats):
        for j, b in enumerate(("Batch_1", "Batch_2", "Batch_3")):
            d = pif[pif.batch == b]
            x = np.full(len(d), j) + np.linspace(-0.22, 0.22, len(d))
            ax.scatter(x, d[f], s=34, color=BATCH_COLOR[b], alpha=0.85,
                       edgecolor="white", lw=0.4, zorder=3)
            for xx, (_, rr) in zip(x, d.iterrows()):
                if rr.image_id in flagged:
                    ax.scatter(xx, rr[f], s=90, facecolor="none",
                               edgecolor=C_WARN, lw=1.6, zorder=4)
            ax.hlines(d[f].median(), j - 0.3, j + 0.3, color=C_INK,
                      lw=2.2, zorder=5)
        row = pw[(pw.feature == f) & (pw.pair == "Batch_1-Batch_3")]
        if len(row) and row.iloc[0].significant_bh:
            ax.set_title(name + "   [BH-significant B1–B3]",
                         fontsize=10.5, color=C_WARN)
        else:
            ax.set_title(name, fontsize=10.5)
        ax.set_xticks(range(3), ["B1", "B2", "B3"], fontsize=10)
        ax.tick_params(labelsize=9)
    fig.suptitle("one dot = one image (the independent unit); "
                 "bar = batch median;  red rings = flagged Batch_1 "
                 "regions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

save(fig_dotgrid(), "fig_dotgrid.png")

# =====================================================================
# 18. exact permutation test illustration
# =====================================================================
def fig_permtest():
    f = "pores_per_mpx"
    a = pif[pif.batch == "Batch_1"][f].to_numpy()
    b = pif[pif.batch == "Batch_3"][f].to_numpy()
    obs = np.median(a) - np.median(b)
    allv = np.concatenate([a, b])
    idx = np.array(list(itertools.combinations(range(len(allv)),
                                               len(a))))
    sel = allv[idx]
    mask = np.ones((len(idx), len(allv)), bool)
    mask[np.arange(len(idx))[:, None], idx] = False
    rest = np.broadcast_to(allv, (len(idx), len(allv)))[mask]
    rest = rest.reshape(len(idx), len(b))
    diffs = np.median(sel, axis=1) - np.median(rest, axis=1)
    p = float((np.abs(diffs) >= abs(obs) - 1e-12).mean())
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    ax.hist(diffs, bins=80, color="#c8cdd6", edgecolor="none")
    ax.axvline(obs, color=C_B1, lw=2.2)
    ax.text(obs - 1.0, ax.get_ylim()[1] * 0.55,
            f"observed\n+{obs:.1f} pores/Mpx\np = {p:.4f}",
            color=C_B1, fontsize=10.5, fontweight="bold", ha="right")
    ax.set_xlabel("median difference under random batch labels")
    ax.set_ylabel("count of splits")
    ax.set_title(f"exact permutation test — all {len(diffs):,} ways to\n"
                 "split 24 images into 7 vs 17 (no seed, no model)",
                 fontsize=11.5)
    ax = axes[1]
    row = pw[(pw.feature == f) & (pw.pair == "Batch_1-Batch_3")].iloc[0]
    ax.errorbar([1], [obs],
                yerr=[[obs - row.ci_lo], [row.ci_hi - obs]],
                fmt="o", color=C_B1, ms=9, capsize=8, lw=2)
    ax.axhline(0, color=C_MUTE, lw=1)
    ax.axhline(row.mdd_critical, color=C_ACCENT, ls="--", lw=1.4)
    ax.text(1.04, row.mdd_critical, f" smallest detectable\n at α=0.05: "
            f"{row.mdd_critical:.0f}/Mpx", fontsize=9.5, color=C_ACCENT,
            va="center")
    ax.axhline(row.mdd_80pct, color=C_WARN, ls=":", lw=1.4)
    ax.text(1.04, row.mdd_80pct, f" 80%-power shift:\n"
            f"{row.mdd_80pct:.0f}/Mpx", fontsize=9.5, color=C_WARN,
            va="center")
    ax.set_xlim(0.5, 2.1)
    ax.set_xticks([1], ["Batch_1 − Batch_3\nmedian diff + bootstrap CI"])
    ax.set_ylabel("pores per Mpx")
    ax.set_title("the same difference with its interval —\n"
                 "and the smallest differences this test could even see",
                 fontsize=11.5)
    print(f"  [check] permutation p recomputed = {p:.6f} "
          f"(saved {row.p_exact:.6f})")
    return fig

save(fig_permtest(), "fig_permtest.png")

# =====================================================================
# 19. LOO influence
# =====================================================================
def fig_loo():
    loo = pd.read_csv(os.path.join(ROOT, "validated_comparison",
                                   "loo_sweep.csv"))
    d = loo[(loo.pair == "Batch_1-Batch_3")
            & (loo.feature == "pores_per_mpx")]
    d = d[d.dropped.isin(pif[pif.batch == "Batch_1"].image_id)]
    flagged = {"img_4ih2ggld", "img_5n1q8atc", "img_f1vzngrs"}
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    xs = np.arange(len(d))
    cols = [C_WARN if i in flagged else C_B3
            for i in d.dropped]
    ax.bar(xs, -np.log10(d.p_exact), color=cols, width=0.6)
    ax.axhline(-np.log10(0.05), color=C_MUTE, ls="--", lw=1.2)
    ax.text(len(d) - 0.4, -np.log10(0.05) + 0.15, "p = 0.05",
            color=C_MUTE, fontsize=9.5)
    ax.set_xticks(xs, [i.replace("img_", "") for i in d.dropped],
                  fontsize=9.5)
    ax.set_ylabel("−log10 exact p")
    ax.set_xlabel("Batch_1 image dropped (leave-one-out)")
    ax.set_title("the pore-count finding survives dropping ANY single\n"
                 "Batch_1 image — it is batch-level, not one weird photo",
                 fontsize=12.5)
    ax.text(0.01, 0.93, "red = the three flagged regions",
            transform=ax.transAxes, fontsize=9.5, color=C_WARN)
    return fig

save(fig_loo(), "fig_loo.png")

# =====================================================================
# 20. noise adjustment
# =====================================================================
def fig_noise():
    na = R["noise_adj"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    for b in ("Batch_1", "Batch_2", "Batch_3"):
        d = pif[pif.batch == b]
        ax.scatter(d.noise_mad, d.pores_per_mpx, color=BATCH_COLOR[b],
                   s=42, label=b, edgecolor="white", lw=0.5)
    xs = np.linspace(6, 12, 10)
    ax.plot(xs, na["slope"] * xs + na["intercept"], color=C_WARN,
            ls="--", lw=1.6)
    ax.text(11.7, na["slope"] * 11.7 + na["intercept"] - 18,
            f"reference fit: slope {na['slope']:.1f}\n(R²={na['ref_r2']:.2f})",
            fontsize=9.5, color=C_WARN, ha="right")
    ax.set_xlabel("image noise (MAD on bulk)")
    ax.set_ylabel("resolved pores per Mpx")
    ax.set_title("noisier photos resolve more pores —\npart of the count "
                 "is an imaging effect", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax = axes[1]
    for j, b in enumerate(("Batch_1", "Batch_2", "Batch_3")):
        d = pif[pif.batch == b]
        x = np.full(len(d), j * 2)
        ax.scatter(x - 0.28, d.pores_per_mpx, color=C_MUTE, s=30,
                   alpha=0.7)
        ax.scatter(x + 0.28, d.pores_per_mpx_noise_adj,
                   color=BATCH_COLOR[b], s=34)
        ax.hlines([d.pores_per_mpx.median(),
                   d.pores_per_mpx_noise_adj.median()],
                  [j * 2 - 0.5, j * 2 + 0.08],
                  [j * 2 - 0.08, j * 2 + 0.5],
                  color=C_INK, lw=2)
    ax.set_xticks(np.arange(3) * 2, ["B1", "B2", "B3"])
    ax.set_ylabel("pores per Mpx")
    ax.set_title("grey = raw counts;  colour = after removing the\n"
                 "noise-predicted part (fit on Batch_3 only)",
                 fontsize=11.5)
    return fig

save(fig_noise(), "fig_noise_adj.png")

# =====================================================================
# 21. real PyBaMM time trace (2 cycles, Batch_3 centre point)
# =====================================================================
def fig_trace():
    sys.path.insert(0, ROOT)
    import warnings
    warnings.filterwarnings("ignore")
    from micro2dfn import simulate, dfn_inputs
    import pybamm
    mk = pd.read_csv(os.path.join(ROOT, "dfn_output", "markers_all.csv"))
    bands = dfn_inputs.batch_bands(mk[mk.batch == "Batch_3"])
    pts = dfn_inputs.sweep_points(bands, n_points=0)
    param = simulate.base_parameters()
    param.update(dfn_inputs.to_pybamm_params(pts.iloc[0].to_dict()),
                 check_already_exists=False)
    model = simulate.build_model(degradation=True)
    sim = pybamm.Simulation(model, parameter_values=param,
                            experiment=simulate._experiment(2))
    sim.solve()
    sol = sim.solution
    t = sol["Time [h]"].entries
    V = sol["Voltage [V]"].entries
    I = sol["Current [A]"].entries
    stress = sol["X-averaged negative secondary particle surface "
                 "tangential stress [Pa]"].entries / 1e6
    thick = sol["Cell thickness change [m]"].entries * 1e6
    cap = sol["Discharge capacity [A.h]"].entries
    bounds = []
    tt = 0.0
    names = ["discharge", "rest", "charge", "CV hold", "rest"]
    colmap = {"discharge": "#dceeff", "rest": "#f2f2f2",
              "charge": "#ffe8cc", "CV hold": "#fde2e2"}
    for step in sol.cycles[0].steps:
        dt = step["Time [h]"].entries
        bounds.append((tt, tt + dt[-1] - dt[0]))
        tt += dt[-1] - dt[0]
    # per-step boundaries across both cycles
    bounds = []
    for cyc in sol.cycles:
        for i, step in enumerate(cyc.steps):
            e = step["Time [h]"].entries
            bounds.append((e[0], e[-1], names[i]))
    fig, axes = plt.subplots(4, 1, figsize=(11.8, 8.6), sharex=True,
                           gridspec_kw={"hspace": 0.14})
    for ax in axes:
        for x0b, x1b, nm in bounds:
            ax.axvspan(x0b, x1b, color=colmap[nm], alpha=0.55, lw=0)
    axes[0].plot(t, V, color=C_INK, lw=1.4)
    axes[0].set_ylabel("cell V")
    axes[0].set_title("one sweep point (Batch_3 band centre) — real "
                      "PyBaMM output, 2 of the 5 protocol cycles",
                      fontsize=12.5)
    axes[1].plot(t, I, color=C_INK, lw=1.4)
    axes[1].set_ylabel("current (A)")
    axes[1].axhline(0, color=C_MUTE, lw=0.6)
    axes[2].plot(t, stress, color=C_B1, lw=1.4)
    axes[2].set_ylabel("Si surface\nstress (MPa)")
    pk = np.nanargmax(stress[:len(t) // 2])
    axes[2].annotate(f"in-cycle peak {stress[pk]:.1f} MPa\n(during "
                     "discharge = max lithiation)", xy=(t[pk], stress[pk]),
                     xytext=(t[pk] + 0.55, stress[pk] + 2.0),
                     fontsize=9.5, color=C_B1,
                     arrowprops=dict(arrowstyle="->", color=C_B1))
    axes[3].plot(t, thick, color=C_ACCENT, lw=1.4)
    axes[3].set_ylabel("cell thickness\nchange (µm)")
    axes[3].set_xlabel("time (h)")
    for x0b, x1b, nm in bounds[:5]:
        axes[0].text((x0b + x1b) / 2, axes[0].get_ylim()[1] * 0.97,
                     nm, ha="center", va="top", fontsize=8.5, color=C_MUTE)
    axes[0].annotate("capacity = ∫I dt over the DISCHARGE step only\n"
                     "(last−first of the counter)", xy=(bounds[0][1] - 0.05,
                     2.95), xytext=(1.5, 3.35), fontsize=9.5,
                     color=C_INK, arrowprops=dict(arrowstyle="->",
                     color=C_INK))
    fig.align_ylabels(axes)
    return fig

save(fig_trace(), "fig_trace.png")

# =====================================================================
# 22. image -> DFN input trace
# =====================================================================
def fig_inputmap():
    bands_df = pd.read_csv(os.path.join(ROOT, "dfn_output",
                                        "dfn_input_bands.csv"))
    b3 = bands_df[bands_df.batch == "Batch_3"].set_index("param")
    row = pif[(pif.image_id == REF_IMG)].iloc[0]
    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 56)
    ax.add_patch(FancyBboxPatch((1, 20), 21, 26, boxstyle="round,pad=0.3",
                                fc="#eef1f5", ec=C_ACCENT))
    ax.text(11.5, 43, "one image", ha="center", fontsize=12,
            fontweight="bold", color=C_ACCENT)
    ax.text(11.5, 33, f"{REF_IMG}\n(Batch_3)\n\npore_frac = "
            f"{row.pore_frac:.3f}\nSi cand = {row.si_candidate_frac:.3f}\n"
            f"Si D50 = {row.si_d50_um:.2f} µm\npores/Mpx = "
            f"{row.pores_per_mpx:.0f}", ha="center", va="center",
            fontsize=10)
    ax.annotate("", xy=(34, 33), xytext=(23, 33),
                arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.6))
    ax.text(28.5, 35.4, "median\nover 17", ha="center", fontsize=9,
            color=C_MUTE)
    ax.add_patch(FancyBboxPatch((35, 20), 24, 26, boxstyle="round,pad=0.3",
                                fc="#eef1f5", ec=C_ACCENT))
    ax.text(47, 43, "batch band (median [range])", ha="center",
            fontsize=12, fontweight="bold", color=C_ACCENT)
    ax.text(47, 32.5, f"porosity {b3.at['porosity','med']:.3f}\n"
            f"Si act. share {b3.at['si_act_med','med']:.3f}"
            f"–{b3.at['si_act_hi','med']:.3f}\n"
            f"Si radius {b3.at['si_radius_um','med']:.2f} µm\n"
            f"Bruggeman {b3.at['bruggeman','med']:.2f}",
            ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(66, 33), xytext=(60, 33),
                arrowprops=dict(arrowstyle="-|>", color=C_MUTE, lw=1.6))
    ax.text(63, 35.4, "1 centre\n+16 LHS", ha="center", fontsize=9,
            color=C_MUTE)
    ax.add_patch(FancyBboxPatch((67, 20), 31, 26, boxstyle="round,pad=0.3",
                                fc="#fdf2e3", ec=C_SI_EDGE))
    ax.text(82.5, 43, "17 PyBaMM runs per batch", ha="center",
            fontsize=12, fontweight="bold", color=C_SI_EDGE)
    ax.text(82.5, 32.5,
            "NEG porosity ← resolved + assumed sub-res\n"
            "NEG Si vol frac ← active share (a bracket)\n"
            "Si radius ← D50/2   (Gr radius FIXED 5.86 µm)\n"
            "Bruggeman ← 2-D proxy\n"
            "thickness, E, ν ← ASSUMED ranges\n"
            "cathode ← rescaled to hold reference N/P",
            ha="center", va="center", fontsize=9.6)
    ax.text(50, 14.5, "an image never becomes a cell: per-image markers "
            "→ batch medians → ONE point inside an assumption band\n"
            "→ 17 model runs that sweep the unmeasured inputs",
            ha="center", fontsize=11.5, color=C_INK, style="italic")
    ax.text(50, 6.5, "pore count is NOT among the inputs — the pore "
            "finding lives outside this model entirely",
            ha="center", fontsize=11, color=C_WARN)
    ax.set_title("tracing one field of view into the model",
                 fontsize=13.5, loc="left")
    return fig

save(fig_inputmap(), "fig_inputmap.png")

# =====================================================================
# 23. sweep bands + declared thresholds
# =====================================================================
def fig_sweepbands():
    res = pd.read_csv(os.path.join(ROOT, "dfn_output",
                                   "dfn_results.csv"))
    thr = {"discharge_cap_last_Ah": 0.25, "thickness_change_um": 0.5,
           "si_stress_MPa": 0.5, "min_neg_surface_v": 0.01}
    names = {"discharge_cap_last_Ah": "cycle-5 discharge capacity (Ah)",
             "thickness_change_um": "peak thickness change (µm)",
             "si_stress_MPa": "peak Si stress (MPa)",
             "min_neg_surface_v": "min anode surface V"}
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.4))
    for ax, (ind, t) in zip(axes, thr.items()):
        for j, b in enumerate(("Batch_1", "Batch_2", "Batch_3")):
            d = res[(res.batch == b) & (res.variant == "accessible")][ind]
            x = np.full(len(d), j) + np.linspace(-0.25, 0.25, len(d))
            ax.scatter(x, d, s=26, color=BATCH_COLOR[b], alpha=0.8,
                       edgecolor="white", lw=0.3)
            ax.hlines(d.median(), j - 0.33, j + 0.33, color=C_INK, lw=2)
        ax.set_title(names[ind], fontsize=10)
        ax.set_xticks(range(3), ["B1", "B2", "B3"], fontsize=9)
        ax.tick_params(labelsize=8.5)
    fig.suptitle("17 model runs per batch — the SPREAD is our own "
                 "assumptions (thickness, sub-res pores, Si share, "
                 "E, ν),\nnot measurement error or cell-to-cell "
                 "variation", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

save(fig_sweepbands(), "fig_sweepbands.png")

# =====================================================================
# 24. paired differences forest
# =====================================================================
def fig_paired():
    d = pd.read_csv(os.path.join(ROOT, "dfn_output",
                                 "dfn_paired_differences.csv"))
    names = {"discharge_cap_last_Ah": "cycle-5 cap (Ah)",
             "thickness_change_um": "peak thickness (µm)",
             "si_stress_MPa": "peak Si stress (MPa)",
             "min_neg_surface_v": "min anode V"}
    cols = {"discharge_cap_last_Ah": 4, "thickness_change_um": 3,
            "si_stress_MPa": 2, "min_neg_surface_v": 1}
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 4.6))
    for ax, (ind, cn) in zip(axes, cols.items()):
        sub = d[d.indicator == ind]
        t = sub.declared_threshold.iloc[0]
        for j, (_, r) in enumerate(sub.iterrows()):
            c = C_WARN if r.verdict == "meaningful_difference" else (
                C_MUTE if r.verdict == "below_threshold" else "#b8860b")
            ax.errorbar([j], [r.median_paired_diff],
                        yerr=[[r.median_paired_diff - r.diff_lo],
                              [r.diff_hi - r.median_paired_diff]],
                        fmt="o", ms=9, capsize=7, color=c, lw=1.8)
        ax.axhline(0, color="#b7c0cc", lw=0.9)
        ax.axhspan(-t, t, color="#eef1f5", zorder=0)
        ax.text(1.0, t * 0.75, f"declared\nmeaningful\n>±{t:g}",
                fontsize=8, color=C_MUTE, ha="center")
        ax.set_title(names[ind], fontsize=10.5)
        ax.set_xticks(range(3), ["B1−B3", "B2−B3", "B1−B2"], fontsize=9)
        ax.set_xlim(-0.6, 2.6)
    fig.suptitle("paired same-assumption differences — red = meaningful "
                 "per the up-front rule; grey = below threshold;\n"
                 "capacity differences are real but SMALLER than what "
                 "we declared meaningful", fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig

save(fig_paired(), "fig_paired.png")

# =====================================================================
# 25. ablation before/after
# =====================================================================
def fig_ablation():
    stages = ["reference\ncell", "our geometry\n(no degradation)",
              "SHIPPED\n(buggy)", "repaired\n(current)"]
    cap1 = [5.401, 4.951, 4.585, 5.086]
    cap5 = [5.415, 4.961, 3.243, 5.101]
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    ax.bar(x - 0.17, cap1, width=0.32, color="#c8cdd6",
           label="cycle 1")
    ax.bar(x + 0.17, cap5, width=0.32, color=C_ACCENT,
           label="cycle 5")
    ax.axhline(5.0, color=C_MUTE, ls="--", lw=1)
    ax.text(3.45, 5.02, "nominal 5 Ah", fontsize=8.5, color=C_MUTE,
            va="bottom", ha="right")
    for i, (a, b) in enumerate(zip(cap1, cap5)):
        ax.text(i - 0.17, a + 0.06, f"{a:.2f}", ha="center", fontsize=8.5)
        ax.text(i + 0.17, b + 0.06, f"{b:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(x, stages, fontsize=9.5)
    ax.set_ylabel("discharge capacity (Ah)")
    ax.set_ylim(0, 6.4)
    ax.legend(fontsize=9)
    ax.set_title("the shipped model lost ~40% capacity in 5 cycles —\n"
                 "a parameter bug (LAM 3600× too fast + graphite params "
                 "on Si)", fontsize=11)
    ax = axes[1]
    ax.axis("off")
    ax.set_title("what the repair changed (8/8 checks pass)", fontsize=11.5)
    ax.text(0.02, 0.92,
            "BEFORE:  graphite OCP/kinetics/mechanics copied onto silicon\n"
            "             LAM constant 1e-3 s⁻¹ (should be 2.78e-7 s⁻¹)\n"
            "             capacity read at cycle end of a running counter\n"
            "             plating warnings without a reference cell\n\n"
            "AFTER:   silicon keeps its own chemistry (Bonkile2024)\n"
            "             published OKane2022 degradation constants\n"
            "             capacity = discharge-step counter diff (=∫I dt)\n"
            "             cathode rescaled to hold reference N/P\n\n"
            "checks are self-consistency, NOT proof of accuracy.",
            fontsize=10.5, va="top", family="Helvetica",
            color=C_INK)
    return fig

save(fig_ablation(), "fig_ablation.png")

# =====================================================================
# 26. crosswalk agreement
# =====================================================================
def fig_crosswalk():
    cw = pd.read_csv(os.path.join(ROOT, "reconcile_output",
                                  "metric_crosswalk.csv"))
    cw = cw.sort_values("corr")
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    cols = [C_WARN if v < 0.5 else (C_ACCENT if v >= 0.8 else "#b8860b")
            for v in cw["corr"]]
    ax.barh(range(len(cw)), cw["corr"], color=cols, height=0.62)
    ax.set_yticks(range(len(cw)),
                  [f"{c}" for c in cw.concept], fontsize=10)
    for i, v in enumerate(cw["corr"]):
        ax.text(v + 0.015 if v >= 0 else v - 0.015, i, f"{v:.2f}",
                va="center", fontsize=9.5,
                ha="left" if v >= 0 else "right")
    ax.axvline(0.8, color=C_MUTE, ls="--", lw=1)
    ax.set_xlim(-0.4, 1.12)
    ax.set_xlabel("per-image correlation, polaron_qc vs micro2dfn "
                  "(n=31 images)")
    ax.set_title("two independent codebases agree on shared measurements\n"
                 "— the ONE disagreement is exactly the Si classifier "
                 "decision", fontsize=12.5)
    return fig

save(fig_crosswalk(), "fig_crosswalk.png")

# =====================================================================
# 27. new image assignment
# =====================================================================
def fig_newbatch():
    a = pd.read_csv(os.path.join(ROOT, "new_image_assignment",
                                 "assignments.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4),
                           gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    envcols = ["inside_env_Batch_1", "inside_env_Batch_2",
               "inside_env_Batch_3"]
    x = np.arange(len(a))
    for j, c in enumerate(envcols):
        b = c.replace("inside_env_", "")
        ax.bar(x + (j - 1) * 0.26, a[c], width=0.24,
               color=BATCH_COLOR.get(b, C_NEW),
               label=b.replace("_", " "))
    ax.axhline(9, color=C_MUTE, ls="--", lw=1)
    ax.set_xticks(x, [i.replace("img_", "") for i in a.image_id],
                  fontsize=9.5)
    ax.set_ylabel("features inside batch envelope (of 9)")
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    ax.set_title("envelope coverage — how many of 9 features\n"
                 "fit inside each batch's observed range", fontsize=11.5)
    ax = axes[1]
    ax.axis("off")
    rows = []
    for _, r in a.iterrows():
        rows.append(f"{r.image_id}:  -> {r.assigned_batch}\n"
                    f"    {r.confidence}")
    ax.text(0.02, 0.95, "\n".join(rows), fontsize=10.5, va="top",
            family="Helvetica", color=C_INK)
    ax.set_title("calls — similarity evidence, NOT provenance",
                 fontsize=11.5)
    return fig

save(fig_newbatch(), "fig_newbatch.png")

# =====================================================================
# 28. prospective roadmap
# =====================================================================
def fig_roadmap():
    fig, ax = plt.subplots(figsize=(12.4, 6.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    steps = [
        ("1. Before imaging", "confirm pixel size, detector\nsettings, FOV count >= plan",
         "stop if acquisition differs"),
        ("2. Intake", "hash files; record batch &\nspecimen identity",
         "reject unlabelled files"),
        ("3. Quality gate", "contrast / noise / sharpness\nguards vs reference envelope",
         "outside -> REVIEW, no verdict"),
        ("4. Frozen analysis", "same thresholds, classifier,\nfeatures, tests as this run",
         "any recipe change = new study"),
        ("5. Blinded compare", "save results BEFORE\nunblinding labels",
         "results exist before opinions"),
        ("6. Independent check", "expert sheet / EDS /\nrepeat imaging",
         "unresolved -> report unresolved"),
        ("7. Decision", "predeclared rules only:\nsupported / possible / unclear",
         "no post-hoc score invention"),
    ]
    layout = [(2 + i * 25, 58) for i in range(4)] + \
             [(8.5 + i * 30, 18) for i in range(3)]
    w, h, gh = 22, 15, 11
    for i, (t, body, gate) in enumerate(steps):
        x, y = layout[i]
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.2,rounding_size=0.5",
                     fc="#eef1f5", ec=C_ACCENT, lw=1.3))
        ax.text(x + w / 2, y + h - 3.2, t, ha="center", fontsize=11,
                fontweight="bold", color=C_ACCENT)
        ax.text(x + w / 2, y + h / 2 - 2.2, body, ha="center",
                va="center", fontsize=9.2)
        ax.add_patch(FancyBboxPatch((x, y - gh - 3), w, gh,
                     boxstyle="round,pad=0.2,rounding_size=0.5",
                     fc="#fdf2e3", ec=C_SI_EDGE, lw=1.1))
        ax.text(x + w / 2, y - gh / 2 - 3, gate, ha="center",
                va="center", fontsize=8.8, color=C_SI_EDGE)
        nxt = layout[i + 1] if i + 1 < len(steps) else None
        if nxt and nxt[1] == y:          # same-row arrow
            ax.annotate("", xy=(nxt[0] - 1.0, y + h / 2),
                        xytext=(x + w + 0.4, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTE,
                                        lw=1.3))
        elif nxt:                        # wrap arrow row1 -> row2
            ax.annotate("", xy=(nxt[0] - 1.0, nxt[1] + h / 2),
                        xytext=(x + w + 0.4, y + h / 2),
                        arrowprops=dict(arrowstyle="-|>", color=C_MUTE,
                                        lw=1.3,
                                        connectionstyle="arc3,rad=0.12"))
        ax.annotate("", xy=(x + w / 2, y - 2.2),
                    xytext=(x + w / 2, y - 0.6),
                    arrowprops=dict(arrowstyle="-|>", color=C_SI_EDGE,
                                    lw=1.0, linestyle="--"))
    ax.text(50, 96, "prospective protocol for the NEXT batch — "
            "freeze the ruler before seeing the data",
            ha="center", fontsize=14, fontweight="bold")
    ax.text(50, 90.5, "teal = work steps      orange = pass / "
            "stop-for-review gates", ha="center", fontsize=10.5,
            color=C_MUTE)
    fig.tight_layout()
    return fig

save(fig_roadmap(), "fig_roadmap.png")

# =====================================================================
# 29. orientation exploratory (appendix)
# =====================================================================
def fig_orientation():
    o = pd.read_csv(os.path.join(
        ROOT, "archive", "orientation_features.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    for ax, key, name in zip(
            axes,
            ("pore_orient_vh", "fft_log10_vh"),
            ("pore-object verticality (v/h, 1=aligned)",
             "FFT vertical/horizontal texture energy (log10)")):
        for j, b in enumerate(("Batch_1", "Batch_2", "Batch_3")):
            d = o[o.batch == b]
            ax.scatter(np.full(len(d), j)
                       + np.linspace(-0.2, 0.2, len(d)),
                       d[key], color=BATCH_COLOR[b], s=36,
                       edgecolor="white", lw=0.4)
            ax.hlines(d[key].median(), j - 0.28, j + 0.28, color=C_INK,
                      lw=2)
        ax.set_xticks(range(3), ["B1", "B2", "B3"])
        ax.set_ylabel(key)
        ax.set_title(name, fontsize=11)
    fig.suptitle("WITHDRAWN claim (archived) — FFT direction measured "
                 "frame aspect ratio, not texture;\n"
                 "object-shape verticality is unexplained, FFT agreement "
                 "was artefact — see archive/ARCHIVE.md", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig

try:
    save(fig_orientation(), "fig_orientation.png")
except Exception as e:
    print("  orientation fig skipped:", e)

# =====================================================================
# 30. adjudication six tests
# =====================================================================
def fig_adjudication():
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 56)
    rows = [
        ("T1", "contrast-correct the interiors",
         "94% of ambiguous objects pass the particle cut vs 45% "
         "on baseline", "fine Si"),
        ("T2", "object shape & position",
         "solidity 0.82, ~1.05 µm discs, only 8% at image border",
         "fine Si"),
        ("T3", "neighbourhood test",
         "adjacent objects are compact discs (0.84), not crescent "
         "rims; they self-cluster", "fine Si clusters"),
        ("T4", "re-fit thresholds per image",
         "floating ruler counts MORE bright (20.7% vs 11.9%) — "
         "not threshold noise", "fine Si"),
        ("T5", "texture on InLens & ETD",
         "ambiguous regions have ≥ real-particle texture "
         "(1.20–1.23×)", "fine Si"),
        ("T6", "acquisition signature",
         "whole-image median shift z=0.0; only the bright tail "
         "moved (z=1.8)", "material change"),
    ]
    ax.text(50, 54, "six tests on the disputed bright material "
            "(reconcile_output/adjudication.csv)", ha="center",
            fontsize=13.5, fontweight="bold")
    for i, (t, q, r, d) in enumerate(rows):
        y = 47 - i * 8.2
        ax.add_patch(FancyBboxPatch((2, y - 2.6), 7, 5.6,
                     boxstyle="round,pad=0.15", fc=C_ACCENT, ec="none"))
        ax.text(5.5, y, t, ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(11, y + 1.4, q, fontsize=10.5, fontweight="bold")
        ax.text(11, y - 1.7, r, fontsize=9.6, color=C_MUTE)
        ax.text(97, y, "→ " + d, fontsize=10, color=C_OK, ha="right",
                fontweight="bold")
    ax.text(50, 0.0, "verdict: LIKELY FINE SILICON — but identity still "
            "needs EDS (brightness ≠ chemistry); the dim contrast on "
            "2/3 regions is a secondary anomaly",
            ha="center", fontsize=10.5, color=C_WARN)
    return fig

save(fig_adjudication(), "fig_adjudication.png")

# ---- write annotation provenance ------------------------------------
pd.DataFrame(ann_rows).to_csv(os.path.join(ANN, "annotations.csv"),
                              index=False)
print("annotations.csv rows:", len(ann_rows))
print("DONE")
