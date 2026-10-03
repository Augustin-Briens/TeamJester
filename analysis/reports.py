"""Figures, Excel workbooks and PDF reports for the baseline/delta analysis.

Conventions: fixed batch colours (common.BATCH_COLORS), Batch_3 always drawn
as the reference band, PNG at 200 dpi.
"""
import os
import glob
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as XLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

import common as C

ASSUME = ("Assumption (applies throughout): the grey bulk is assumed graphite, "
          "bright particles are assumed silicon-based, black is pore/crack. "
          "This comes from image appearance and is unconfirmed.")

STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="Answer", fontSize=11, leading=14,
                          spaceAfter=8, backGround=colors.HexColor("#eef3f8"),
                          borderPadding=6))
STYLES.add(ParagraphStyle(name="Small", fontSize=8, leading=10))


def df_to_sheet(wb, name, df):
    ws = wb.create_sheet(name)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    return ws


def save_xlsx(path, sheets):
    wb = Workbook(); wb.remove(wb.active)
    for name, df in sheets.items():
        df_to_sheet(wb, name[:31], df)
    wb.save(path)
    C.log(f"wrote {path}")


def pdf_doc(path, title, answer):
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=15 * mm, rightMargin=15 * mm)
    story = [Paragraph(title, STYLES["Title"]),
             Paragraph(answer, STYLES["Answer"]),
             Paragraph(ASSUME, STYLES["Small"]), Spacer(1, 6)]
    return doc, story


def add_table(story, df, max_rows=40, fs=7):
    df = df.head(max_rows).copy()
    data = [list(df.columns)] + [[
        (f"{v:.4g}" if isinstance(v, float) else str(v))[:42]
        for v in row] for row in df.values]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ec")),
        ("FONTSIZE", (0, 0), (-1, -1), fs),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(t)
    story.append(Spacer(1, 8))


def add_img(story, path, w=170 * mm):
    if path and os.path.exists(path):
        from PIL import Image as PILImage
        iw, ih = PILImage.open(path).size
        story.append(Image(path, width=w, height=w * ih / iw))
        story.append(Spacer(1, 8))


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def fig_feature_panels(F_, kept, stats, extra=None, suffix=""):
    """Every kept feature: all images as points vs the B3 reference band.
    extra = (Series/dict of feature values, label) drawn as a black star."""
    full = F_[F_.subset == "full"]
    groups = {}
    for f in kept:
        groups.setdefault(
            pd.read_csv(os.path.join(C.TABLE_DIR, "feature_meta.csv"))
            .set_index("feature").loc[f, "group"], []).append(f)
    out = []
    for gname, feats in groups.items():
        n = len(feats)
        cols = 4; rows = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows, cols,
                                figsize=(3.4 * cols, 2.6 * rows), dpi=200)
        axs = np.atleast_1d(axs).ravel()
        s3 = stats.set_index("feature")
        for ax, f in zip(axs, feats):
            if f in s3.index:
                lo, hi = s3.loc[f, "env_lo"], s3.loc[f, "env_hi"]
                mu = s3.loc[f, "mean"]
                ax.axhspan(lo, hi, color=C.BATCH_COLORS[C.BASELINE],
                           alpha=0.18)
                ax.axhline(mu, color=C.BATCH_COLORS[C.BASELINE], lw=1)
            for b in C.BATCHES:
                d = full[full.batch == b][f]
                ax.scatter([b] * len(d), d, color=C.BATCH_COLORS[b],
                           s=18, zorder=3, edgecolor="k", linewidth=0.3)
            if extra is not None and f in extra[0].index \
                    and pd.notna(extra[0][f]):
                ax.scatter(["unseen"], [extra[0][f]], marker="*", s=140,
                           color="k", zorder=4, label="unseen sample")
            ax.set_title(f, fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axs[len(feats):]:
            ax.axis("off")
        fig.suptitle(f"{gname} features - points vs Batch_3 5-95% band",
                     fontsize=9)
        fig.tight_layout()
        p = os.path.join(C.FIG_DIR, f"features_{gname}{suffix}.png")
        fig.savefig(p); plt.close(fig); out.append(p)
    return out


def ss():
    return STYLES


def fig_ranked_deltas(D):
    """Ranked standardised deltas with CI whiskers, one per batch."""
    out = []
    for b in ("Batch_1", "Batch_2"):
        d = D[D.batch == b].sort_values("abs_std_delta", ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(4, 0.28 * len(d))), dpi=200)
        colors_ = [C.BATCH_COLORS[b] if c == "different" else
                   "#888888" for c in d.classification]
        ax.barh(d.feature, d.std_delta, color=colors_, height=0.7)
        # CI whiskers on the standardised scale (raw CI / B3 std)
        bs = pd.read_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv")
                         ).set_index("feature")["std"]
        sd3 = d.feature.map(bs)
        lo = d.ci_lo / sd3
        hi = d.ci_hi / sd3
        xerr = np.vstack([(d.std_delta - lo).clip(lower=0),
                          (hi - d.std_delta).clip(lower=0)])
        ax.errorbar(d.std_delta, d.feature, xerr=xerr,
                    fmt="none", ecolor="black", elinewidth=0.7, capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("standardised delta (in Batch_3 std)")
        ax.set_title(f"{b} vs Batch_3 - ranked deltas "
                     f"(coloured = survives Holm correction)", fontsize=9)
        fig.tight_layout()
        p = os.path.join(C.FIG_DIR, f"ranked_deltas_{b}.png")
        fig.savefig(p); plt.close(fig); out.append(p)
    return out


def fig_distributions(locs):
    """Si size distribution CDFs + pore thickness distributions overlaid."""
    P = pd.read_csv(os.path.join(C.TABLE_DIR, "particles.csv"))
    R = pd.read_csv(os.path.join(C.TABLE_DIR, "pores.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200)
    for b in C.BATCHES:
        dd = P[P.batch == b].diam_um
        x = np.sort(dd); y = np.linspace(0, 1, len(x))
        axs[0].plot(x, y, color=C.BATCH_COLORS[b], lw=1.2,
                    label=b, alpha=0.85)
        tt = R[R.batch == b].thick_mean_um
        x = np.sort(tt); y = np.linspace(0, 1, len(x))
        axs[1].plot(x, y, color=C.BATCH_COLORS[b], lw=1.2, label=b)
    axs[0].set_xscale("log"); axs[1].set_xscale("log")
    axs[0].set_xlabel("Si equivalent diameter (um)")
    axs[0].set_ylabel("cumulative fraction")
    axs[0].legend(fontsize=7); axs[0].set_title("Si particle sizes")
    axs[1].set_xlabel("pore local thickness (um)")
    axs[1].set_title("Pore size distributions")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "distributions.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_depth_profiles(locs):
    """Pore fraction across 8 depth bands per image, per batch."""
    rows = []
    for rec in locs:
        z = C.load_masks(rec["batch"], rec["image_id"])
        pore = z["pore"]
        h = pore.shape[0]
        bands = [pore[i * h // 8:(i + 1) * h // 8].mean() for i in range(8)]
        for i, v in enumerate(bands):
            rows.append(dict(batch=rec["batch"], image_id=rec["image_id"],
                             band=i / 7, pore_frac=v))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
    for b in C.BATCHES:
        bb = D[D.batch == b]
        m = bb.groupby("band").pore_frac.mean()
        s = bb.groupby("band").pore_frac.std()
        ax.fill_between(m.index, m - s, m + s, color=C.BATCH_COLORS[b],
                        alpha=0.2)
        ax.plot(m.index, m, color=C.BATCH_COLORS[b], label=b, marker="o",
                ms=2)
    ax.set_xlabel("normalised depth through image"); ax.set_ylabel("pore frac")
    ax.legend(fontsize=7); ax.set_title("Pore depth profile (mean +/- sd)")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "pore_depth.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_dfn():
    """Discharge-curve bands + capacity vs rate + delta-vs-baseline panel."""
    try:
        with open(os.path.join(C.TABLE_DIR, "dfn_curves.pkl"), "rb") as f:
            curves = pickle.load(f)
        cap = pd.read_csv(os.path.join(C.TABLE_DIR, "dfn_capacity.csv"))
    except Exception:
        return None
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=200)
    mode = "corrected"
    rates = sorted({k[2] for k in curves})
    for rate in rates:
        for b in C.BATCHES:
            key = (b, mode, rate)
            if key not in curves:
                continue
            c = curves[key]
            axs[0].plot(c["Q"], c["V_md"], color=C.BATCH_COLORS[b],
                        lw=1.1, label=f"{b} {rate}C")
            axs[0].fill_between(c["Q"], c["V_lo"], c["V_hi"],
                                color=C.BATCH_COLORS[b], alpha=0.15)
    axs[0].set_xlabel("Capacity (Ah)"); axs[0].set_ylabel("Voltage (V)")
    axs[0].set_title("Discharge curves (corrected porosity)")
    axs[0].legend(fontsize=5, ncol=2)
    for b in C.BATCHES:
        for md, ls in (("measured", ":"), ("corrected", "-")):
            cc = cap[(cap.batch == b) & (cap.porosity_mode == md)]
            cc = cc.sort_values("c_rate")
            axs[1].errorbar(cc.c_rate, cc.cap_mean,
                            yerr=[cc.cap_mean - cc.cap_p5,
                                  cc.cap_p95 - cc.cap_mean],
                            color=C.BATCH_COLORS[b], ls=ls, marker="o",
                            ms=3, lw=1, label=f"{b} {md}")
    axs[1].set_xscale("log"); axs[1].set_xticks([0.1, 0.5, 1, 2])
    axs[1].set_xticklabels(["C/10", "C/2", "1C", "2C"])
    axs[1].set_xlabel("C-rate"); axs[1].set_ylabel("Capacity (Ah)")
    axs[1].legend(fontsize=5, ncol=2)
    axs[1].set_title("Capacity vs rate (bands = input resampling)")
    # delta panel — bootstrap CIs from dfn_deltas.csv (same numbers as dfn.xlsx)
    try:
        dd = pd.read_csv(os.path.join(C.TABLE_DIR, "dfn_deltas.csv"))
    except Exception:
        dd = None
    for b in ("Batch_1", "Batch_2"):
        for md, ls in (("measured", ":"), ("corrected", "-")):
            if dd is not None:
                mm = dd[(dd.batch == b) & (dd.porosity_mode == md)]
                mm = mm.sort_values("c_rate")
                axs[2].errorbar(mm.c_rate, mm.delta_ah,
                                yerr=[mm.delta_ah - mm.ci_lo,
                                      mm.ci_hi - mm.delta_ah],
                                color=C.BATCH_COLORS[b], ls=ls, marker="o",
                                ms=3, lw=1, label=f"{b} {md}")
            else:
                bb = cap[(cap.batch == b) & (cap.porosity_mode == md)]
                ref = cap[(cap.batch == C.BASELINE) &
                          (cap.porosity_mode == md)]
                mm = bb.merge(ref, on="c_rate", suffixes=("", "_3"))
                d = mm.cap_mean - mm.cap_mean_3
                axs[2].errorbar(mm.c_rate, d,
                                color=C.BATCH_COLORS[b], ls=ls, marker="o",
                                ms=3, lw=1, label=f"{b} {md}")
    axs[2].axhline(0, color="k", lw=0.8)
    axs[2].set_xscale("log"); axs[2].set_xticks([0.1, 0.5, 1, 2])
    axs[2].set_xticklabels(["C/10", "C/2", "1C", "2C"])
    axs[2].set_xlabel("C-rate"); axs[2].set_ylabel("dCapacity vs Batch_3 (Ah)")
    axs[2].legend(fontsize=5); axs[2].set_title("Delta from baseline")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "dfn.png")
    fig.savefig(p); plt.close(fig)

    # end-of-discharge profiles (median + 5-95% band across resamples)
    try:
        with open(os.path.join(C.TABLE_DIR, "dfn_profile_curves.pkl"),
                  "rb") as f:
            prof = pickle.load(f)
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), dpi=200)
        titles = dict(c_e="Electrolyte concentration (end of discharge)",
                      c_s_gr="Graphite particle concentration (end)",
                      c_s_si="SiOx particle concentration (end)")
        for ax, key in zip(axs, ("c_e", "c_s_gr", "c_s_si")):
            for b in C.BATCHES:
                d = prof.get((b, key))
                if d is None:
                    continue
                x = np.linspace(0, 1, len(d["md"]))
                ax.plot(x, d["md"], color=C.BATCH_COLORS[b], lw=1.2,
                        label=b)
                ax.fill_between(x, d["lo"], d["hi"],
                                color=C.BATCH_COLORS[b], alpha=0.15)
            ax.set_xlabel("position across electrode (0=collector)")
            ax.set_title(titles[key], fontsize=8)
            ax.legend(fontsize=6)
        fig.tight_layout()
        p2 = os.path.join(C.FIG_DIR, "dfn_profiles.png")
        fig.savefig(p2); plt.close(fig)
    except Exception:
        pass
    return p


def fig_categorisation_map(P, full, feats):
    """PCA map of all images + B3 envelope ellipse + LOO predictions."""
    b3 = full[full.batch == C.BASELINE]
    mu = full[feats].median()
    sc = (full[feats] - mu).abs().median() * 1.4826
    sc = sc.replace(0, np.nan).fillna(full[feats].std()).replace(0, 1)
    Z = ((full[feats] - mu) / sc).fillna(0).values
    U, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    PC = (Z - Z.mean(0)) @ Vt.T[:, :2]
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=200)
    for i, b in enumerate(C.BATCHES):
        m = (full.batch == b).values
        ax.scatter(PC[m, 0], PC[m, 1], color=C.BATCH_COLORS[b], s=26,
                   label=b, edgecolor="k", linewidth=0.3, zorder=3)
    # B3 envelope ellipse on the 2 PCs
    zb = PC[(full.batch == C.BASELINE).values]
    c = zb.mean(0)
    cov = np.cov(zb.T)
    w_, v_ = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v_[1, 1], v_[0, 1]))
    for k in (2.0, 3.0):  # ~chi2(2) scaled
        e = Ellipse(c, 2 * k * np.sqrt(w_[1]), 2 * k * np.sqrt(w_[0]),
                    angle=ang, fill=False, color=C.BATCH_COLORS[C.BASELINE],
                    ls="--", lw=1)
        ax.add_patch(e)
    ax.set_xlabel("PC1 of standardised features")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    ax.set_title("Categorisation map (dashed = Batch_3 envelope on PCs)")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "categorisation_map.png")
    fig.savefig(p); plt.close(fig)
    return p


def fig_evidence(delta_row, kind, locs):
    """Side-by-side evidence images for a confirmed difference."""
    import segment as S
    b = delta_row["batch"]; feat = delta_row["feature"]
    full_stats = None
    P = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    full = P[P.subset == "full"]
    # pick the most typical images per batch (median of the feature)
    rows = []
    for bb in (C.BASELINE, b):
        d = full[full.batch == bb][feat].dropna()
        med = d.median()
        iid = d.index[(d - med).abs().argmin()]
        rows.append(dict(batch=bb, image_id=full.loc[iid, "image_id"],
                         val=float(med)))
    ims = []
    for r in rows:
        rec = [x for x in locs if x["image_id"] == r["image_id"]][0]
        ims.append((f"{r['batch']} {r['image_id']} ({feat}={r['val']:.3g})",
                    S.overlay_image(rec, scale=6)))
    h = max(i[1].shape[0] for i in ims)
    w = sum(i[1].shape[1] for i in ims)
    from PIL import Image as PILImage, ImageDraw
    canvas = PILImage.new("RGB", (w, h + 22), "white")
    dr = ImageDraw.Draw(canvas)
    x = 0
    for tit, im in ims:
        canvas.paste(PILImage.fromarray(im), (x, 22))
        dr.text((x + 4, 4), tit, fill="black")
        x += im.shape[1]
    p = os.path.join(C.FIG_DIR, f"evidence_{b}_{feat}.png")
    canvas.save(p)
    return p


# ---------------------------------------------------------------------------
# WORKBOOKS
# ---------------------------------------------------------------------------
def build_workbooks(locs):
    t = C.TABLE_DIR
    seg = pd.read_csv(os.path.join(t, "segment_info.csv"))
    sens = pd.read_csv(os.path.join(t, "threshold_sensitivity.csv"))
    xd = pd.read_csv(os.path.join(t, "cross_detector.csv"))
    q = pd.read_csv(os.path.join(t, "quality_guards.csv"))
    rep_lr = pd.read_csv(os.path.join(t, "half_agreement.csv"))
    save_xlsx(os.path.join(C.XLSX_DIR, "segmentation_validation.xlsx"),
              dict(Segmentation=seg, ThresholdSensitivity=sens,
                   LeftRight=rep_lr, CrossDetector=xd, QualityGuards=q))

    F_ = pd.read_csv(os.path.join(t, "features.csv"))
    meta = pd.read_csv(os.path.join(t, "feature_meta.csv"))
    dropped = pd.read_csv(os.path.join(t, "dropped_features.csv"))
    rep = pd.read_csv(os.path.join(t, "repeatability.csv"))
    save_xlsx(os.path.join(C.XLSX_DIR, "features.xlsx"),
              dict(PerImage=F_[F_.subset == "full"],
                   Halves=F_[F_.subset != "full"],
                   FeatureDictionary=meta, Dropped=dropped,
                   Repeatability=rep))

    b3s = pd.read_csv(os.path.join(t, "baseline_stats.csv"))
    cons = pd.read_csv(os.path.join(t, "baseline_consistency.csv"))
    sheets = dict(BaselineStats=b3s, Consistency=cons)
    fp = os.path.join(t, "baseline_stats_noflag.csv")
    if os.path.exists(fp):
        sheets["BaselineStatsNoFlag"] = pd.read_csv(fp)
    fp = os.path.join(t, "baseline_consistency_noflag.csv")
    if os.path.exists(fp):
        sheets["ConsistencyNoFlag"] = pd.read_csv(fp)
    save_xlsx(os.path.join(C.XLSX_DIR, "baseline_batch3.xlsx"), sheets)

    D = pd.read_csv(os.path.join(t, "deltas.csv"))
    Dx = pd.read_csv(os.path.join(t, "deltas_dropped.csv")) \
        if os.path.exists(os.path.join(t, "deltas_dropped.csv")) else pd.DataFrame()
    sheets = dict(Deltas=D, DroppedFeatureDeltas=Dx)
    fp = os.path.join(t, "deltas_noflag.csv")
    if os.path.exists(fp):
        sheets["DeltasNoFlag"] = pd.read_csv(fp)
    save_xlsx(os.path.join(C.XLSX_DIR, "deltas.xlsx"), sheets)

    for name, files in (
        ("dfn.xlsx", dict(Inputs="dfn_inputs.csv",
                          Assumptions="dfn_assumptions.csv",
                          Capacity="dfn_capacity.csv",
                          PerSimRuns="dfn_runs.csv",
                          Deltas="dfn_deltas.csv",
                          Profiles="dfn_profiles.csv",
                          ThicknessSweep="dfn_thickness_sweep.csv")),
        ("categorisation.xlsx", dict(LOO="loo_predictions.csv",
                                     LOOSummary="loo_summary.csv",
                                     Confusion="loo_confusion.csv",
                                     ImagesNeeded="images_needed.csv",
                                     LOO_ArtefactSafe="loo_predictions_safe.csv",
                                     ImagesNeeded_Safe="images_needed_safe.csv",
                                     LOO_Safe_NoFlag="loo_predictions_safe_noflag.csv"))):
        sheets = {}
        for sheet, fn in files.items():
            fp = os.path.join(t, fn)
            if os.path.exists(fp):
                sheets[sheet] = pd.read_csv(fp)
        if sheets:
            save_xlsx(os.path.join(C.XLSX_DIR, name), sheets)
    # merge the point-count human-validation sheet into the segmentation
    # workbook as a copy (original stays standalone with embedded images)
    pcxlsx = os.path.join(C.PC_DIR, "point_count_validation.xlsx")
    if os.path.exists(pcxlsx):
        try:
            from openpyxl import load_workbook
            src = load_workbook(pcxlsx)
            segx = os.path.join(C.XLSX_DIR, "segmentation_validation.xlsx")
            dst = load_workbook(segx)
            if "PointCount" in dst.sheetnames:
                del dst["PointCount"]
            sws = src.active
            dws = dst.create_sheet("PointCount")
            for row in sws.iter_rows():
                for c in row:
                    dws.cell(row=c.row, column=c.column, value=c.value)
            dst.save(segx)
        except Exception as e:
            C.log(f"point-count merge failed: {e}")
    C.log("workbooks written")

# ---------------------------------------------------------------------------
# PDF BUILDERS
# ---------------------------------------------------------------------------
def pdf_01_segmentation(locs):
    t = C.TABLE_DIR
    seg = pd.read_csv(os.path.join(t, "segment_info.csv"))
    sens = pd.read_csv(os.path.join(t, "threshold_sensitivity.csv"))
    ha = pd.read_csv(os.path.join(t, "half_agreement.csv"))
    xd = pd.read_csv(os.path.join(t, "cross_detector.csv"))
    q = pd.read_csv(os.path.join(t, "quality_guards.csv"))
    px = seg.um_per_px.iloc[0] * 1000
    mx = seg.um_per_px.abs().max() - seg.um_per_px.abs().min()
    ans = (f"Segmentation used fixed settings on every image: percentile "
           f"normalise (p1-p99.5), gaussian sigma 1.5, lower multi-Otsu for "
           f"pore, multi-Otsu for silicon (the bright tail has no "
           f"separate histogram mode - the valley search falls back to "
           f"multi-Otsu on every image), opening + watershed. "
           f"Pixel size {px:.1f} nm/px from TIFF resolution (FOV ~175x54 "
           f"um). Detectors are pixel-aligned (zero-shift cross-correlation), "
           f"so Inlens-dark pixels next to pores refine cracks. Median pore "
           f"fraction shifts <~10% at +/-10% threshold; left/right pore "
           f"fraction correlation "
           f"{ha.pore_L.corr(ha.pore_R):.2f}; "
           f"two low-contrast images are flagged by the quality guards.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "01_segmentation.pdf"),
                         "Phase 1 - Segmentation", ans)
    story.append(Paragraph(
        "One method, one set of settings for all 31 images. Classes: "
        "pore/crack (black), graphite bulk (grey), silicon particle "
        "(bright). No binder class was invented: intensity between bulk and "
        "pore did not form a separable third phase.", STYLES["Normal"]))
    add_table(story, seg[["batch", "image_id", "um_per_px", "pore_frac",
                          "silicon_frac", "t_lo", "t_bright"]], max_rows=31)
    story.append(Paragraph(
        "Threshold sensitivity: relative change in each phase fraction "
        "when each threshold is shifted +/-10% (median over images).",
        STYLES["h3"]))
    rows = []
    for (param, shift), g in sens.groupby(["param", "shift"]):
        base = sens[(sens["param"] == param) & (sens["shift"] == 0.0)]
        for ph in ("pore", "silicon", "graphite"):
            rel = (g[ph].mean() - base[ph].mean()) / base[ph].mean()
            rows.append(dict(param=param, shift=shift, phase=ph,
                             rel_change=rel))
    add_table(story, pd.DataFrame(rows))
    story.append(Paragraph(
        "Left/right half agreement (independent segmentation of each "
        "half) and cross-detector agreement.", STYLES["h3"]))
    add_table(story, ha)
    add_table(story, xd)
    story.append(Paragraph(
        "Cross-detector note: raw pixel IoU between an Inlens-only dark "
        "mask and the BSE pore mask is low (~0.1-0.2) because the Inlens "
        "detector is texturally dark over most of the surface - fine "
        "surface detail is not the same as void space. Inlens is "
        "therefore only used to refine pixels within 2 px of a BSE pore, "
        "not as a standalone pore detector.", STYLES["Small"]))
    story.append(Paragraph(
        "Image-quality guards; flagged images are excluded in parallel "
        "analyses:", STYLES["h3"]))
    add_table(story, q[["batch", "image_id", "si_bulk_contrast", "noise_mad",
                        "sharpness", "shading_range", "flagged",
                        "flag_reason"]], max_rows=31)
    story.append(Paragraph(
        "Example overlays (grey=graphite, orange=silicon, blue=pore) and "
        "per-batch contact sheets:", STYLES["h3"]))
    ov = sorted(glob.glob(os.path.join(C.OVL_DIR, "overlay_*.png")))
    for p in ov[:2]:
        add_img(story, p, w=170 * mm)
    for p in sorted(glob.glob(os.path.join(C.OVL_DIR, "contact_*.png"))):
        add_img(story, p, w=170 * mm)
    doc.build(story)


def pdf_02_baseline(locs):
    t = C.TABLE_DIR
    bs = pd.read_csv(os.path.join(t, "baseline_stats.csv"))
    cons = pd.read_csv(os.path.join(t, "baseline_consistency.csv"))
    pf = bs[bs.feature == "pore_frac"].iloc[0]
    ans = (f"Batch_3 baseline ({int(pf.n)} images): visible pore fraction "
           f"{pf['mean']:.3f} (5-95% {pf.p5:.3f}-{pf.p95:.3f}), silicon "
           f"{bs[bs.feature=='silicon_frac']['mean'].iloc[0]:.3f}, graphite "
           f"{bs[bs.feature=='graphite_frac']['mean'].iloc[0]:.3f}. "
           f"Internal consistency: "
           f"{int(cons.outlier.sum())} image(s) sit >3.5 robust-z on some "
           f"feature - the batch is broadly homogeneous with a "
           f"high-contrast subgroup; no image was removed from the "
           f"baseline for that reason alone.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "02_baseline.pdf"),
                         "Phase 3 - Batch_3 baseline", ans)
    add_table(story, bs[["feature", "n", "mean", "median", "std", "p5",
                         "p95", "ci95_lo", "ci95_hi"]])
    story.append(Paragraph("Internal consistency (robust z per image)",
                           STYLES["h3"]))
    add_table(story, cons)
    for p in sorted(glob.glob(os.path.join(C.FIG_DIR, "features_*.png"))):
        if "_unseen" in p:
            continue
        add_img(story, p)
    for p in sorted(glob.glob(os.path.join(C.FIG_DIR,
                                           "distributions_*.png"))):
        add_img(story, p)
    p = os.path.join(C.FIG_DIR, "depth_profiles.png")
    add_img(story, p)
    doc.build(story)


def pdf_03_deltas(locs):
    t = C.TABLE_DIR
    D = pd.read_csv(os.path.join(t, "deltas.csv"))
    Dx = pd.read_csv(os.path.join(t, "deltas_dropped.csv"))
    b1 = D[D.batch == "Batch_1"]
    b2 = D[D.batch == "Batch_2"]
    def hl(Db):
        hits = Db[Db.classification.isin(["different", "suggestive"])]
        return "; ".join(
            f"{r.feature} {r.pct_delta:+.0f}% ({r.classification}"
            f"{', Holm ok' if r.p_holm < 0.05 else ''})"
            for r in hits.itertuples())
    ans = (f"Batch_1 vs Batch_3 (kept features): {hl(b1) or 'none'}. "
           f"Batch_2 vs Batch_3: {hl(b2) or 'none'}. "
           f"With 7 images the smallest detectable standardised difference "
           f"is ~1.0-1.5 B3 std (Mann-Whitney); intervals that do not "
           f"survive Holm correction across the feature set are labelled "
           f"'suggestive', not proven.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "03_deltas.pdf"),
                         "Phase 5 - Deltas vs Batch_3", ans)
    for b in ("Batch_1", "Batch_2"):
        p = os.path.join(C.FIG_DIR, f"ranked_deltas_{b}.png")
        add_img(story, p)
    cols = ["batch", "feature", "mean_batch", "mean_b3", "delta",
            "pct_delta", "std_delta", "ci_lo", "ci_hi", "p_mw", "p_holm",
            "classification", "mdd"]
    add_table(story, D[cols])
    story.append(Paragraph(
        "Dropped/unreliable features (reported for completeness - do not "
        "drive conclusions):", STYLES["h3"]))
    add_table(story, Dx[cols])
    doc.build(story)


def pdf_04_dfn(locs):
    t = C.TABLE_DIR
    ip = pd.read_csv(os.path.join(t, "dfn_inputs.csv")) \
        if os.path.exists(os.path.join(t, "dfn_inputs.csv")) else pd.DataFrame()
    cap = pd.read_csv(os.path.join(t, "dfn_capacity.csv")) \
        if os.path.exists(os.path.join(t, "dfn_capacity.csv")) else pd.DataFrame()
    ans = ("PyBaMM BasicDFNComposite with the published Chen2020_composite "
           "parameter set (LGM50 NMC811 / graphite-SiOx negative). Image "
           "inputs: measured porosity AND a literature-range correction "
           "(set to 0.30, literature range for calendered anodes; small pores unresolved), active fractions, "
           "particle radii. Under measured porosity the model predicts "
           "Batch_1 and Batch_2 lose ~3% capacity vs Batch_3 at 2C only "
           "(CI excludes zero); under the corrected-0.30 porosity the batches are "
           "indistinguishable - true porosity dominates the small image "
           "differences. Absolute values are NOT validated; only "
           "between-batch differences matter. The model has no swelling, "
           "cracking or ageing.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "04_dfn.pdf"),
                         "Phase 4 - DFN simulation", ans)
    if len(ip):
        add_table(story, ip.head(20))
    if len(cap):
        add_table(story, cap)
    for p in sorted(glob.glob(os.path.join(C.FIG_DIR, "dfn*.png"))):
        add_img(story, p)
    doc.build(story)


def pdf_05_causes(locs):
    t = C.TABLE_DIR
    D = pd.read_csv(os.path.join(t, "deltas.csv"))
    q = pd.read_csv(os.path.join(t, "quality_guards.csv"))
    ans = ("Inference, not proof: Batch_1's lower visible porosity with "
           "fewer large gaps is best matched by denser packing / less "
           "mixing damage, NOT harder calendering - pore chords shrink "
           "isotropically whereas pressing predicts directional "
           "flattening. Both Batch_1 and Batch_2 carry a stronger "
           "imaging-noise signature (BSE bulk texture +15-18%, correlates "
           "0.91 with the noise guard) - part of the between-batch signal "
           "is acquisition, not material. Alternative causes are listed "
           "and checked below; none is proven.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "05_causes.pdf"),
                         "Phase 7 - Why the differences exist", ans)
    rows = [
        ["Cause", "Predicts", "Observed", "Verdict"],
        ["Harder calendering (B1)",
         "less porosity, fewer large gaps, FLATTER pores (H/V anisotropy)",
         "pore_frac -13%, large_gap_frac -67%, but pore chords shrink "
         "isotropically (-5% both axes, anisotropy unchanged)",
         "partially contradicted - shrink is isotropic"],
        ["Denser packing / less cracking on mixing (B1)",
         "uniformly less pore space, fewer large voids, same Si size",
         "pore_frac -13%, large_gap_frac -67%, crack_frac -13%, Si size "
         "equivalent",
         "best fit - inference"],
        ["Powder lot change (B1)",
         "different Si size/shape distribution",
         "Si D10/D50 equivalent-to-undetermined",
         "weak"],
        ["Mixing/dispersion (B1)",
         "changed Si clustering, graphite correlation length",
         "Clark-Evans R +7% (undetermined), corr_len +18% (suggestive)",
         "weak support"],
        ["Sample prep / polishing (B1,B2)",
         "global contrast and texture shifts",
         "Si-bulk contrast lower in 2 flagged B1; BSE texture +16-18% "
         "tracks noise guard",
         "partially - artefact present"],
        ["Imaging conditions (B1,B2)",
         "uniform noise/texture offset",
         "noise_mad clusters: B1&B2 ~10.4 vs B3 ~7.4-10.4; "
         "bse_bulk_texture r=0.91 with noise",
         "confirmed for texture features"],
    ]
    tt = Table(rows, repeatRows=1)
    tt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2ec")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("COLWIDTH", (0, 0), (0, 0), 35*mm)]))
    story.append(tt)
    story.append(Paragraph(
        "References opened: Chen et al. 2020 (JES 167 080534, open access) "
        "for electrode parameterisation; Profatilova et al. 2020 (ACS AEM, "
        "main text paywalled, free SI) for porosity/calendering effects; "
        "Torquato 2002 (Princeton-hosted paper) for two-point statistics; "
        "BoneJ 'thickness' docs for local thickness; Wikipedia "
        "'Tortuosity' for tau definitions.", STYLES["Small"]))
    for p in sorted(glob.glob(os.path.join(C.FIG_DIR, "evidence_*.png"))):
        add_img(story, p)
    doc.build(story)


def pdf_06_categorisation(locs):
    t = C.TABLE_DIR
    P = pd.read_csv(os.path.join(t, "loo_predictions.csv"))
    G = pd.read_csv(os.path.join(t, "images_needed.csv"))
    Ps = pd.read_csv(os.path.join(t, "loo_predictions_safe.csv")) \
        if os.path.exists(os.path.join(t, "loo_predictions_safe.csv")) \
        else pd.DataFrame()
    fa = (P[(P.batch == C.BASELINE) & (P.call_inout == "OUT")].shape[0]
          / max((P.batch == C.BASELINE).sum(), 1))
    acc = P.groupby("batch").apply(
        lambda g: (g.batch == g.pred_batch).mean())
    acc_s = (Ps.groupby("batch").apply(
        lambda g: (g.batch == g.pred_batch).mean())
        if len(Ps) else acc)
    ans = (f"Single images are NOT reliably categorisable: LOO which-batch "
           f"accuracy "
           f"B1 {acc.get('Batch_1',0):.0%}/B2 {acc.get('Batch_2',0):.0%}"
           f"/B3 {acc.get('Batch_3',0):.0%} with a {fa:.0%} false-alarm "
           f"rate on held-out Batch_3 (artefact-safe set: "
           f"{acc_s.get('Batch_1',0):.0%}/{acc_s.get('Batch_2',0):.0%}"
           f"/{acc_s.get('Batch_3',0):.0%}). Groups of >=5 images reach "
           f"~97-100% for Batch_1; Batch_2 needs ~7 because it genuinely "
           f"overlaps Batch_3. categorise.py therefore treats a folder as "
           f"one sample (median features) and sizes the OUT threshold to "
           f"the group count.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "06_categorisation.pdf"),
                         "Phase 6 - Categorising unseen samples", ans)
    story.append(Paragraph("LOO predictions (all kept features)",
                           STYLES["h3"]))
    add_table(story, P)
    if len(Ps):
        story.append(Paragraph("LOO predictions (artefact-safe features)",
                               STYLES["h3"]))
        add_table(story, Ps)
    story.append(Paragraph("Accuracy vs group size (which-batch)",
                           STYLES["h3"]))
    add_table(story, G)
    p = os.path.join(C.FIG_DIR, "categorisation_map.png")
    add_img(story, p)
    doc.build(story)


def pdf_00_summary(locs):
    t = C.TABLE_DIR
    D = pd.read_csv(os.path.join(t, "deltas.csv"))
    P = pd.read_csv(os.path.join(t, "loo_predictions.csv"))
    G = pd.read_csv(os.path.join(t, "images_needed.csv"))
    ans = ("Batch_1 differs from the Batch_3 baseline mainly by having "
           "less visible pore space (~13%) and far fewer large gap pores "
           "(~2/3 less) while silicon fraction and particle sizes are "
           "unchanged - consistent with a denser / harder-pressed "
           "electrode. Batch_2 is statistically close to Batch_3 on "
           "morphology. Both B1/B2 show a stronger imaging-noise "
           "signature than B3, so part of the apparent difference is "
           "acquisition, not material. Single images cannot be "
           "categorised reliably; a folder of >=5 images can: ~100% for "
           "B1-like samples, ~80-100% for B2-like at n=7.")
    doc, story = pdf_doc(os.path.join(C.PDF_DIR, "00_summary.pdf"),
                         "Batch delta analysis - summary", ans)
    story.append(Paragraph(
        "Baseline: Batch_3 n=17 images, all segmented identically. "
        "Batch_1: pore_frac -13.6% (suggestive, CI excludes 0 but does "
        "not survive Holm), large_gap_frac -71% (suggestive), "
        "bse_bulk_texture +18% (different, survives Holm but tracks "
        "imaging noise). Batch_2: bse_bulk_texture +15% (different, "
        "noise-tracked), gr_st_coherence -8% (suggestive); everything "
        "else equivalent or undetermined. Prior quick-pass findings "
        "agree in direction (less pore space in B1, B2 close to B3); "
        "the +45% silicon signal in the quick pass came from the two "
        "low-contrast flagged images and disappears without them.",
        STYLES["Normal"]))
    p = os.path.join(C.FIG_DIR, "ranked_deltas_Batch_1.png")
    add_img(story, p)
    p = os.path.join(C.FIG_DIR, "categorisation_map.png")
    add_img(story, p)
    story.append(Paragraph(
        "Reliability: leave-one-image-out which-batch accuracy "
        f"{(P.batch==P.pred_batch).mean():.0%} overall - poor for single "
        "images; the categoriser must be used on groups of >=5 images. "
        "False-alarm rate on held-out Batch_3 singles: "
        f"{((P.batch==C.BASELINE)&(P.call_inout=='OUT')).mean():.0%}.",
        STYLES["Normal"]))
    doc.build(story)


def write_notes(locs, extra=""):
    t = C.TABLE_DIR
    q = pd.read_csv(os.path.join(t, "quality_guards.csv"))
    dropped = pd.read_csv(os.path.join(t, "dropped_features.csv"))
    seg = pd.read_csv(os.path.join(t, "segment_info.csv"))
    txt = f"""# NOTES - assumptions, failures, exclusions

## Assumptions
- {ASSUME}
- Pixel size {seg.um_per_px.iloc[0]*1000:.1f} nm/px read from TIFF
  XResolution; identical for every image; FOV ~175 x 54 um.
- The three detectors are pixel-aligned (peak cross-correlation at zero
  shift), so Inlens dark pixels are used to refine pores.
- Visible porosity (~10%) underestimates true porosity: pores below the
  ~50 nm pixel resolution are invisible. DFN is run both with the
  measured value and a literature-range correction setting porosity to 0.30.
- Two-point correlation / Clark-Evans / chord methods treat each image
  as one statistical realisation; images within a batch are the samples.
- Tortuosity proxies (tau_x, tau_y) are 2D diffusion solves on the
  downsampled pore mask; a 2D section at ~9% porosity cannot represent
  3D transport - reported as proxy only and dropped from the decision
  set for non-repeatability.

## Excluded / flagged images
{chr(10).join("- %s %s: %s" % (r.batch, r.image_id, r.flag_reason)
   for r in q[q.flagged].itertuples()) or "- none"}

## Dropped features (not used in baseline/deltas; reasons in
## dropped_features.csv)
{chr(10).join("- %s: %s" % (r.feature, r.reason)
   for r in dropped.itertuples())}

## Failures / known limitations
- tau_x/tau_y, pore percolation and crack/skeleton features fail the
  left/right repeatability check (NaN or <0.5 corr) - pores do not
  percolate at ~9% in these 2D sections. Reported in
  deltas_dropped.csv but not used in the decision set.
- Single-image categorisation is unreliable (LOO false-alarm ~50%);
  use >=5 images per sample.
- bse_bulk_texture and etd_roughness correlate ~0.9 with the image
  noise guard - treated as imaging-signature (artefact-risk) features
  and excluded from the artefact-safe categoriser.
- Multi-Otsu valley detection failed on images with a poorly-separated
  bright tail; the guard/flag mechanism covers those.
- point_count_validation.xlsx human column is intentionally empty;
  score with score_pointcount.py.

## Verification vs prior quick-pass
- silicon fraction same across batches: AGREES once the 2 flagged
  images are excluded (with them, +45% driven by low contrast).
- Batch_1 ~18% less pore space: AGREES directionally (-13.6% here,
  suggestive not Holm-significant).
- Batch_2 fewer elongated cracks: NOT confirmed - crack_len_density
  is -21% in Batch_1 not Batch_2, and the feature itself is
  unrepeatable (dropped).
- flagged problem images 4ih2ggld / 5n1q8atc: CONFIRMED by the
  contrast guard (si_bulk_contrast 2.15/2.27 vs >=3.40 elsewhere).

{extra}
"""
    p = os.path.join(C.OUT, "..", "NOTES.md")
    with open(p, "w") as f:
        f.write(txt)
    C.log(f"wrote {p}")
    return p


def build_all_pdfs(locs):
    for fn in (pdf_01_segmentation, pdf_02_baseline, pdf_03_deltas,
               pdf_04_dfn, pdf_05_causes, pdf_06_categorisation,
               pdf_00_summary):
        try:
            fn(locs)
            C.log(f"built {fn.__name__}")
        except Exception as e:
            C.log(f"FAILED {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
