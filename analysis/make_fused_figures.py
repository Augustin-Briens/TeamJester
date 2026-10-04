#!/usr/bin/env python3
"""Annotated figures for the fused-model report.

Arrows point at objects computed from the stored segmentation masks —
largest pore region, elongated cracks, silicon clusters — so every callout
is anchored on a real measured object, not hand-placed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import measure, morphology
from scipy import ndimage as ndi
from common import OUT, FIG_DIR as FIG, TABLE_DIR, MASK_DIR, OVL_DIR, BATCH_COLORS as BCOL

plt.rcParams.update({"font.size": 10, "figure.dpi": 200})
CLASSES = ["Batch_1", "Batch_2", "Batch_3"]

# ---------- 1. annotated overlay panels ------------------------------------
REP = {"Batch_1": "fzrt2k6r", "Batch_2": "b3esycq1", "Batch_3": "pl8uabbv"}
SCALE = 0.25  # overlay PNGs are the masks downscaled 4x

def objects(mask, iid):
    d = np.load(os.path.join(MASK_DIR, f"{mask}_{iid}_masks.npz"))
    pore, si, lab = d["pore"], d["silicon"], d["labels"]
    H, W = pore.shape
    out = {}
    # largest pore region
    cc = measure.label(pore, connectivity=2)
    props = measure.regionprops(cc)
    big = max(props, key=lambda p: p.area)
    out["largest pore region\n(fragmented network)"] = big.centroid
    # elongated crack: pore object, thin + vertical-ish
    cands = [p for p in props if p.area > 400 and p.axis_major_length > 3 * max(p.axis_minor_length, 1)]
    if cands:
        p = max(cands, key=lambda p: p.axis_major_length)
        out["elongated crack-like pore"] = p.centroid
    # silicon cluster: densest local Si neighbourhood (downsampled density conv)
    if si.any():
        small = si[::4, ::4].astype(float)
        dens = ndi.uniform_filter(small, size=125)  # ~500 px window
        iy, ix = np.unravel_index(np.argmax(dens), dens.shape)
        out["densest silicon cluster"] = (iy * 4, ix * 4)
    # biggest isolated pore far from Si
    if si.any():
        dist = ndi.distance_transform_edt(~si)
        pm = pore.copy(); pm[:40] = False; pm[-40:] = False
        pm[:, :40] = False; pm[:, -40:] = False
        iy, ix = np.unravel_index(np.argmax(dist * pm), pore.shape)
        out["deepest pore pocket"] = (iy, ix)
    return out, (pore, si)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
LEG = {}
for ax, (b, iid) in zip(axes, REP.items()):
    img = plt.imread(os.path.join(OVL_DIR, f"overlay_{b}_{iid}.png"))
    ax.imshow(img)
    calls, _ = objects(b, iid)
    colors = ["#FF3B30", "#FF9500", "#0A84FF", "#30D158"]
    for n, ((txt, (ry, rx)), c) in enumerate(zip(calls.items(), colors), start=1):
        oy, ox = ry * SCALE, rx * SCALE
        ax.plot(ox, oy, "o", ms=26, mfc=c, mec="white", mew=2, alpha=0.92, zorder=5)
        ax.text(ox, oy, str(n), ha="center", va="center", fontsize=11,
                color="white", fontweight="bold", zorder=6)
        LEG.setdefault(b, []).append((n, c, txt))
    ax.set_title(f"{b} · img_{iid}", color=BCOL[b], fontsize=12, fontweight="bold")
    ax.axis("off")
    # legend: colored number + label, two columns under the image
    for k, (n, c, txt) in enumerate(LEG[b]):
        ax.text(0.02 + 0.5 * (k % 2), -0.06 - 0.09 * (k // 2), f"{n} {txt}",
                transform=ax.transAxes, fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.22", fc=c, ec="none", alpha=0.9))
fig.suptitle("Segmentation overlays — numbered markers sit on the objects the metrics count\n"
             "(blue = pore, amber = silicon, grey = graphite bulk)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fused_annotated_overlays.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 2. marker panels with callouts ---------------------------------
full = pd.read_csv(os.path.join(TABLE_DIR, "features.csv"))
full = full[full.subset == "full"].reset_index(drop=True)
exp = pd.read_csv(os.path.join(TABLE_DIR, "exp_newfeat.csv"))
exp = exp[exp.subset == "full"]
full = full.merge(exp[["image_id", "elong_pore_n_mm2", "inl_lbp_flat"]], on="image_id")

panels = [("pore_frac", "resolved porosity", "B1 below the baseline band —\ndenser packing"),
          ("pore_largest_cc_frac", "largest connected pore region", "B1 most fragmented —\nagrees with teammate pipeline"),
          ("elong_pore_n_mm2", "elongated crack-like pores / mm²", "B3 most, B2 fewest —\n'fewer surface tears' hypothesis"),
          ("inl_lbp_flat", "Inlens flat-texture share", "B2 smoothest surface")]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (f, ttl, note) in zip(axes, panels):
    b3 = full[full.batch == "Batch_3"][f]
    lo, hi = np.percentile(b3, [5, 95])
    ax.axhspan(lo, hi, color=BCOL["Batch_3"], alpha=0.12)
    ax.axhline(b3.median(), color=BCOL["Batch_3"], ls="--", lw=1)
    for b in CLASSES:
        v = full[full.batch == b][f]
        xi = CLASSES.index(b)
        ax.scatter(xi + np.random.default_rng(1).normal(0, .05, len(v)), v,
                   color=BCOL[b], s=42, edgecolor="white", zorder=3)
        ax.plot([xi], [v.median()], marker="_", ms=26, mew=3, color=BCOL[b])
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, fontsize=8)
    ax.annotate(note, xy=(0.02, 0.03), xycoords="axes fraction", fontsize=8,
                color="#333", style="italic")
    ax.set_title(ttl, fontsize=10); ax.set_xlim(-0.6, 2.6)
fig.suptitle("The surviving signals — per-image values against the Batch_3 reference band")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fused_marker_calls.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 3. ensemble vote heatmap ---------------------------------------
# four signals per image: distance model, RF top-8, DINOv2-ETD embedding, focused B2/B3
import categorise as CAT
kept = pd.read_csv(os.path.join(TABLE_DIR, "baseline_stats.csv"))
gates = pd.read_csv(os.path.join(TABLE_DIR, "exp_newfeat_gates.csv"))
rep = pd.read_csv(os.path.join(TABLE_DIR, "repeatability.csv"))
newf = [f for f, ok in zip(gates.feature, gates.keep) if ok and f in exp.columns]
feats_all = [f for f in kept.feature if f in full.columns] + newf
miss = [f for f in newf if f not in full.columns]
if miss:
    full = full.merge(exp[["image_id"] + miss], on="image_id")
y = full["batch"].values; ids = full["image_id"].values

# distance vote (leave-self-out)
votes = pd.DataFrame({"image_id": ids, "batch": y})
d_vote = []
for i in range(len(full)):
    trdf = full.drop(index=i)
    feats = CAT.select_features(trdf, feats_all, rep)
    model = CAT.build_model(trdf, feats)
    _, best, _, _, _ = CAT.dist_and_predict(full.iloc[i], model)
    d_vote.append(best)
votes["distance"] = d_vote

# RF vote from saved LOO preds
rfv = pd.read_csv(os.path.join(TABLE_DIR, "rf_loo_tuned.csv"))
votes = votes.merge(rfv[["image_id", "pred"]].rename(columns={"pred": "rf_top8"}), on="image_id")

# embedding vote (leave-self-out over stored PCA'd ETD embeddings)
em = np.load(os.path.join(OUT, "emb_model.npz"), allow_pickle=True)
T, eids, eb = em["T"], em["ids"], em["batches"]
e_vote = {}
for i, iid in enumerate(eids):
    tr = np.arange(len(T)) != i
    cents = {b: T[tr][eb[tr] == b].mean(0) for b in CLASSES}
    dd = {b: np.linalg.norm(T[i] - c) for b, c in cents.items()}
    e_vote[iid] = min(dd, key=dd.get)
votes["embedding"] = votes.image_id.map(e_vote)

# focused B2/B3 vote
from sklearn.linear_model import LogisticRegression
b23fe = ["bse_bulk_texture", "inl_lbp_flat", "elong_pore_n_mm2", "etd_roughness", "gr_st_coherence"]
sub = full[full.batch.isin(["Batch_2", "Batch_3"])]
Xs = sub[b23fe].values.astype(float); ys = sub["batch"].values
pv = {}
for i in range(len(sub)):
    tr = np.arange(len(sub)) != i
    pv[sub.image_id.iloc[i]] = LogisticRegression(max_iter=2000).fit(Xs[tr], ys[tr]).predict(Xs[i:i+1])[0]
votes["b23_focused"] = votes.image_id.map(pv)

votes.to_csv(os.path.join(TABLE_DIR, "ensemble_votes.csv"), index=False)
sig = ["distance", "rf_top8", "embedding", "b23_focused"]
M = votes[sig].apply(lambda s: s.map({c: i for i, c in enumerate(CLASSES)})).values
truth = votes["batch"].map({c: i for i, c in enumerate(CLASSES)}).values

fig, ax = plt.subplots(figsize=(9.5, 7))
disp = np.where(M == truth[:, None], M, -1)
cmap = matplotlib.colors.ListedColormap(["#444444"] + [BCOL[c] for c in CLASSES])
ax.imshow(disp, aspect="auto", cmap=cmap, vmin=-1, vmax=2)
ax.set_xticks(range(4)); ax.set_xticklabels(["distance\nmodel", "RF\ntop-8", "DINOv2\nETD", "B2-vs-B3\nfocused"])
ylabs = [f"{r.image_id}  ({r.batch})" for r in votes.itertuples()]
ax.set_yticks(range(len(votes))); ax.set_yticklabels(ylabs, fontsize=7)
ax.set_title("Four independent signals per image — colour = call, grey cell = wrong call\n"
             "(leave-self-out throughout; B2-vs-B3 column only covers B2/B3 rows)")
for sp in ax.spines.values():
    sp.set_visible(False)
ax.grid(which="major", color="white", lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fused_ensemble_votes.png"), bbox_inches="tight")
plt.close(fig)

# ---------- 4. fusion architecture diagram ---------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.axis("off")
def box(x, y, w, h, txt, fc="#f4f4f4", ec="#555", fs=9):
    ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, fc=fc, ec=ec,
                 lw=1.4, zorder=2, boxstyle="round,pad=0.012", mutation_scale=1))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, zorder=3)
def arr(x1, y1, x2, y2, c="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))

box(0.01, 0.62, 0.16, 0.26, "SEM images\nBSE + Inlens + ETD", "#eef")
box(0.24, 0.78, 0.22, 0.16, "RULER — teammate pipeline\nfrozen B3 thresholds +\nbright-fine artefact filter", "#e8f6e8")
box(0.24, 0.5, 0.22, 0.2, "AUDIT — TeamJester\nquality guards,\nnoise-artefact gating", "#fdeeee")
box(0.53, 0.78, 0.2, 0.16, "markers (his)\n+ features (ours)", "#eef")
box(0.53, 0.44, 0.2, 0.26, "label engine\nsolid / possible / can't-tell\nHolm + artefact + session rules", "#fef6e0")
box(0.79, 0.62, 0.19, 0.24, "VOTING CONSENSUS\ndistance · RF top-8\nDINOv2-ETD · B2/B3-focused", "#e8e8fa")
box(0.53, 0.08, 0.2, 0.18, "his DFN stack\nswelling · stress · SEI\n(indicators, not predictions)", "#eee")
box(0.79, 0.08, 0.19, 0.18, "PRODUCT\nwebapp verdict + % surety\n+ evidence card + PDF", "#dff3e3")

arr(0.17, 0.75, 0.24, 0.84); arr(0.17, 0.68, 0.24, 0.6)
arr(0.46, 0.86, 0.53, 0.86); arr(0.46, 0.6, 0.53, 0.58)
arr(0.63, 0.78, 0.63, 0.7); arr(0.73, 0.57, 0.79, 0.7)
arr(0.63, 0.44, 0.63, 0.26); arr(0.73, 0.17, 0.79, 0.17)
arr(0.88, 0.62, 0.88, 0.26)
ax.text(0.5, 0.97, "Fused pipeline — one agreed label vocabulary, two pipelines reporting into it",
        ha="center", fontsize=12, fontweight="bold")
ax.text(0.5, 0.01, "Rules: a claim needs (1) Holm-surviving interval, (2) decorrelation from imaging guards, "
        "(3) within-session survival for count/spatial markers — else it is 'possible' or 'can't-tell'.",
        ha="center", fontsize=8.5, color="#666")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fused_architecture.png"), bbox_inches="tight")
plt.close(fig)
print("figures done:", [f for f in os.listdir(FIG) if f.startswith("fused_")])
