"""Phase 6 - categorise unseen samples.

categorise.py <folder>          classify every img_*_BSE.tif in a folder
categorise.py --validate        run the leave-one-out + controls on the
                                known batches (writes categorisation.xlsx
                                inputs to tables/)

Method (simple, inspectable):
  * feature set: the top SEL_K retained features ranked by discriminability
    (max |standardised delta| of B1/B2 vs B3) among features with left/right
    correlation >= 0.5 - re-fit inside every leave-one-out fold.
  * IN/OUT of the Batch_3 distribution: robust Mahalanobis distance
    (per-feature median + MAD scaling, covariance on the standardised
    features). Threshold = the 95th percentile of the distances obtained on
    the reference Batch_3 images themselves.
  * which batch: nearest batch centroid in the same standardised space;
    confidence = softmax(-distance).
  * top-3 drivers: largest per-feature |z| contributions.
  * missing detector: features that need it become NaN and are dropped from
    the distance for that sample (recomputed per sample).
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import common as C

SEL_K = 6            # 4-8 most reliable features
THRESH_Q = 0.95      # in/out threshold quantile on reference distances

# features whose measured |corr| with the image noise guard (quality_guards
# noise_mad) exceeds 0.7: bse_bulk_texture=0.91, etd_roughness=0.88. They
# track imaging noise rather than structure, so they are reported but kept
# out of the artefact-safe categoriser variant.
ARTEFACT_RISK = {"bse_bulk_texture", "etd_roughness"}

# Images excluded from model fitting and evaluation: low Si/bulk contrast +
# high noise, flagged independently by this pipeline and by a parallel one
# (see NOTES.md). Their extreme values (silicon_frac 0.17-0.18 vs <=0.10
# everywhere else) drag Batch_1's centroid toward artefact outliers - the
# direct cause of a B1<->B2 swap on organiser test images.
BAD_IMAGES = {"4ih2ggld", "5n1q8atc"}

# si_d10_um is stereologically broken: in a 2D cut through a material made
# of tightly packed particles, the smallest apparent silicon blobs are
# glancing chord-cuts of larger particles - it measures the sectioning
# geometry (~0.37 um floor on every batch), not the material. Excluded from
# every model input (organiser feedback, S. Kench).
STEREO_BROKEN = {"si_d10_um"}


# ---------------------------------------------------------------------------
def robust_stats(df, feats):
    """Median + 1.4826*MAD per feature."""
    mu = df[feats].median()
    scale = (df[feats] - mu).abs().median() * 1.4826
    scale = scale.replace(0, np.nan).fillna(df[feats].std()).replace(0, 1.0)
    return mu, scale


def select_features(full, kept, repeat):
    """Top SEL_K features by discriminability among reliable ones."""
    full = full[~full.image_id.isin(BAD_IMAGES)]
    ok = repeat.set_index("feature")["lr_corr"]
    ok = ok[ok >= 0.5].index.tolist()
    cand = [f for f in kept if f in ok and f not in STEREO_BROKEN]
    scores = {}
    b3 = full[full.batch == C.BASELINE]
    for f in cand:
        sd = b3[f].std()
        m = 0.0
        for b in ("Batch_1", "Batch_2"):
            bb = full[full.batch == b]
            d = abs(bb[f].mean() - b3[f].mean()) / max(sd, 1e-12)
            m = max(m, d)
        scores[f] = m
    return sorted(cand, key=lambda f: -scores[f])[:SEL_K]


def build_model(full, feats, batches=C.BATCHES, drop_bad=True):
    """Fit envelope + centroids on `full` (DataFrame subset)."""
    if drop_bad:
        full = full[~full.image_id.isin(BAD_IMAGES)]
    b3 = full[full.batch == C.BASELINE]
    mu, scale = robust_stats(b3, feats)
    Z3 = ((b3[feats] - mu) / scale).fillna(0).values
    cov = np.cov(Z3.T) if len(Z3) > 1 else np.eye(len(feats))
    cov += 1e-6 * np.eye(len(feats))
    cents = {}
    for b in batches:
        bb = full[full.batch == b]
        cents[b] = ((bb[feats] - mu) / scale).mean().values
    # typical distance of each batch's own images to its centroid: an unseen
    # point beyond this is 'unlike any batch', not 'most like' it.
    self_q95 = {}
    for b, c in cents.items():
        bb = full[full.batch == b]
        Zb = ((bb[feats] - mu) / scale).fillna(0).values
        dd = np.linalg.norm(Zb - c, axis=1)
        self_q95[b] = (float(np.nanpercentile(dd, 95))
                       if len(dd) else np.inf)
    return dict(mu=mu, scale=scale, cov=cov, cents=cents, feats=feats,
                self_q95=self_q95)


def dist_and_predict(row, model):
    """Distance to B3 envelope + nearest centroid for one sample (Series)."""
    feats = model["feats"]
    avail = [i for i, f in enumerate(feats) if pd.notna(row.get(f, np.nan))]
    if not avail:
        return np.nan, None, np.nan, [], []
    fsub = [feats[i] for i in avail]
    z = ((row[fsub].astype(float) - model["mu"][fsub]) /
         model["scale"][fsub]).values
    Vi = np.linalg.pinv(model["cov"][np.ix_(avail, avail)])
    d = float(np.sqrt(max(z @ Vi @ z, 0)))
    dists = {b: float(np.linalg.norm(z - c[avail])) for b, c in
             model["cents"].items()}
    ex = {b: np.exp(-v) for b, v in dists.items()}
    tot = sum(ex.values())
    probs = {b: ex[b] / tot for b in ex}
    best = min(dists, key=dists.get)
    drivers = sorted(zip(fsub, np.abs(z)), key=lambda t: -t[1])[:3]
    return d, best, probs[best], drivers, dists


def far_from_all(dists, model):
    """True when even the nearest batch centroid lies beyond that batch's
    own typical spread -> the 'most like' label is weak evidence (e.g. an
    artefact-dominated image that resembles no clean batch)."""
    if not dists:
        return False
    best = min(dists, key=dists.get)
    return dists[best] > model.get("self_q95", {}).get(best, np.inf)


# ---------------------------------------------------------------------------
# LOO validation
# ---------------------------------------------------------------------------
def loo_validate(full, kept, repeat, exclude_ids=(), tag=""):
    df = full[~full.image_id.isin(exclude_ids)].reset_index(drop=True)
    preds = []
    for i in range(len(df)):
        row = df.iloc[i]
        rest = df.drop(index=i)
        feats = select_features(rest, kept, repeat)
        model = build_model(rest, feats)
        # reference distances inside this fold (B3 images excl. held-out)
        b3rest = rest[rest.batch == C.BASELINE]
        ds = []
        for _, r in b3rest.iterrows():
            d, *_ = dist_and_predict(r, model)
            ds.append(d)
        thr = np.nanpercentile(ds, THRESH_Q * 100)
        d, best, conf, drivers, dists = dist_and_predict(row, model)
        preds.append(dict(image_id=row.image_id, batch=row.batch,
                          dist=d, threshold=thr,
                          call_inout="IN" if d <= thr else "OUT",
                          pred_batch=best, conf=conf,
                          drivers="|".join(f"{f}:{z:.2f}" for f, z in drivers),
                          feats="|".join(feats)))
    # NOTE: the `kept` arg may be pre-filtered to ARTEFACT_RISK-free lists
    P = pd.DataFrame(preds)
    sfx = f"_{tag}" if tag else ""
    P.to_csv(os.path.join(C.TABLE_DIR, f"loo_predictions{sfx}.csv"),
             index=False)

    # confusion matrix for which-batch
    cm = pd.crosstab(P.batch, P.pred_batch)
    cm.to_csv(os.path.join(C.TABLE_DIR, f"loo_confusion{sfx}.csv"))
    acc = (P.batch == P.pred_batch).groupby(P.batch).mean()
    # false-alarm: B3 called OUT
    fa = P[(P.batch == C.BASELINE) & (P.call_inout == "OUT")]
    out_acc = P.groupby("batch").apply(
        lambda g: pd.Series(dict(
            batch_acc=(g.batch == g.pred_batch).mean(),
            out_rate=(g.call_inout == "OUT").mean())))
    # shuffled-label control
    rng = np.random.default_rng(7)
    sh_acc = []
    for rep in range(20):
        lab = df.batch.values.copy()
        rng.shuffle(lab)
        right = 0
        for i in range(len(df)):
            row = df.iloc[i]
            rest = df.drop(index=i)
            feats = select_features(rest, kept, repeat)
            model = build_model(rest.assign(batch=lab[np.arange(len(df)) != i]),
                                feats)
            _, best, *_ = dist_and_predict(row, model)
            right += (best == lab[i])
        sh_acc.append(right / len(df))
    summ = dict(per_batch_accuracy=out_acc.to_dict(),
                confusion=cm.to_dict(),
                false_alarm_rate=float((fa.shape[0] /
                                        max((P.batch == C.BASELINE).sum(), 1))),
                shuffled_accuracy_mean=float(np.mean(sh_acc)),
                shuffled_accuracy_std=float(np.std(sh_acc)))
    return P, summ


def images_needed(full, kept, repeat, exclude_ids=(), ns=(1, 2, 3, 4, 5, 6, 7)):
    """How many images of a batch before the group call is reliable."""
    df = full[~full.image_id.isin(exclude_ids)]
    rng = np.random.default_rng(3)
    feats = select_features(df, kept, repeat)
    model = build_model(df, feats)
    rows = []
    for b in ("Batch_1", "Batch_2"):
        ids = df[df.batch == b].image_id.tolist()
        for n in ns:
            hits = 0; trials = 200
            for _ in range(trials):
                pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
                sub = df[df.image_id.isin(pick)]
                mean_row = sub[feats].mean()
                _, best, *_ = dist_and_predict(mean_row, model)
                hits += (best == b)
            rows.append(dict(batch=b, n_images=n,
                             group_accuracy=hits / trials))
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(C.TABLE_DIR, "images_needed.csv"), index=False)
    return G


# ---------------------------------------------------------------------------
# unseen-folder entry point
# ---------------------------------------------------------------------------
PLAIN_WORDS = {
    "pore_frac": "how much dark pore space the image shows",
    "graphite_frac": "how much of the image is graphite bulk",
    "silicon_frac": "how much of the image is bright silicon particles",
    "si_d10_um": "size of the smallest silicon particles (D10)",
    "si_d50_um": "typical silicon particle size (D50)",
    "si_solidity": "how filled-in / non-dented the silicon particles are",
    "gr_chord_h_um": "typical graphite flake length measured horizontally",
    "gr_chord_ratio_hv": "how aligned the graphite is (H vs V flake length)",
    "gr_st_coherence": "how oriented the graphite texture is (structure tensor)",
    "pore_thick_d10_um": "thickness of the finest pores (D10)",
    "pore_thick_d50_um": "typical pore thickness (D50)",
    "large_gap_frac": "share of pore area in large gaps",
    "pores_per_mm2": "number of separate pores per area",
    "ring_porosity_250nm": "porosity in a 250 nm ring around silicon particles",
    "ring_porosity_500nm": "porosity in a 500 nm ring around silicon particles",
    "corr_len_gr_um": "how far graphite regions stay alike (correlation length)",
    "si_clarkevans_R": "silicon clustering vs random (Clark-Evans R)",
    "inlens_edge_density": "amount of fine surface edges (Inlens)",
    "inlens_bulk_texture": "fine texture inside the bulk (Inlens)",
    "etd_roughness": "surface topography roughness (ETD)",
    "bse_bulk_texture": "grey-level texture inside the bulk (BSE)",
    "tau_x": "pore tortuosity proxy, horizontal",
    "tau_y": "pore tortuosity proxy, vertical",
    "elong_pore_area_frac": "share of pore space in long thin cracks",
    "elong_pore_n_mm2": "number of long thin cracks per area",
    "elong_pore_hfrac": "how horizontally the cracks run",
    "pore_aspect_p90": "how stretched the most elongated pores are",
    "pore_small_n_mm2": "number of small compact pores per area",
    "pore_perim_mm": "total pore edge length per area",
    "si_border_pore": "how close silicon particles sit to pores",
    "bse_grad_coh": "how directional the bulk texture is (BSE)",
    "inl_lbp_ent": "surface texture complexity (Inlens)",
    "inl_lbp_flat": "surface smoothness — flat texture share (Inlens)",
    "inl_grad_coh": "surface streakiness (Inlens)",
    "etd_fft_hi": "fine-scale topographic texture energy (ETD)",
}


def group_threshold(b3df, model, n, reps=300, seed=1):
    """95th percentile of distances of medians of groups of n B3 images."""
    rng = np.random.default_rng(seed)
    feats = model["feats"]
    ds = []
    nn = max(1, min(n, len(b3df)))
    for _ in range(reps):
        samp = b3df.iloc[rng.choice(len(b3df), nn, replace=False)]
        med = samp[feats].median()
        ds.append(dist_and_predict(med, model)[0])
    return float(np.nanpercentile(ds, THRESH_Q * 100))


def classify_folder(folder):
    """Classify the sample of unseen images in `folder` as ONE group:
    per-image features -> group median -> Mahalanobis distance to the B3
    envelope with a group-size-adaptive threshold, plus the nearest batch
    centroid. Writes categorise_results.csv (one sample row + one row per
    image) and a one-page PDF next to the folder."""
    import segment as S
    import features as FT
    bses = sorted(glob.glob(os.path.join(folder, "*_BSE.tif")) +
                  glob.glob(os.path.join(folder, "*_bse.tif")))
    if not bses:
        raise SystemExit(f"No *_BSE.tif images in {folder}")
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    rep = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
    full = F_[F_.subset == "full"]
    kept = pd.read_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv")
                       )["feature"].tolist()
    feats = select_features(full, kept, rep)
    model = build_model(full, feats)
    b3 = full[full.batch == C.BASELINE]
    thr1 = float(np.nanpercentile(
        [dist_and_predict(r, model)[0] for _, r in b3.iterrows()],
        THRESH_Q * 100))
    thrn = group_threshold(b3, model, len(bses))

    rows = []
    feat_rows = []
    first_rec = None
    for path in bses:
        iid = os.path.basename(path).split("img_")[1].split("_")[0]
        base = os.path.join(folder, f"img_{iid}")
        rec = dict(batch="unseen", image_id=iid, bse=path,
                   inlens=base + "_Inlens.tif", topo=None, topo_kind=None)
        for k in ("ETD", "SE"):
            if os.path.exists(base + f"_{k}.tif"):
                rec.update(topo=base + f"_{k}.tif", topo_kind=k)
        if not os.path.exists(rec["inlens"]):
            rec["inlens"] = None
        g, _ = C.load_gray(path)
        il = S.load_detector(rec)
        r = S.segment(g, il)
        tmp = os.path.join(C.MASK_DIR, f"unseen_{iid}_masks.npz")
        np.savez_compressed(tmp, pore=r["pore"], silicon=r["silicon"],
                            labels=r["labels"].astype(np.int32),
                            pore_L=r["pore"], silicon_L=r["silicon"],
                            labels_L=r["labels"].astype(np.int32),
                            pore_R=r["pore"], silicon_R=r["silicon"],
                            labels_R=r["labels"].astype(np.int32))
        fe, _, _ = FT.compute(rec, "full")
        feat_rows.append(pd.Series(fe))
        d, best, conf, drivers, _ = dist_and_predict(pd.Series(fe), model)
        rows.append(dict(level="image", image_id=iid, dist=d,
                         threshold=thr1,
                         call_inout="IN" if d <= thr1 else "OUT",
                         most_like=best, confidence=conf,
                         top_features="|".join(
                             f"{PLAIN_WORDS.get(f, f)} (z={z:.2f})"
                             for f, z in drivers)))
        if first_rec is None:
            first_rec = rec
    fdf = pd.DataFrame([dict(r) for r in feat_rows])
    med = fdf.apply(pd.to_numeric, errors="coerce").median()
    d, best, conf, drivers, dists = dist_and_predict(med, model)
    call = "IN" if d <= thrn else "OUT"
    sample = dict(level="sample", image_id=f"group_of_{len(bses)}",
                  dist=d, threshold=thrn, call_inout=call,
                  most_like=best, confidence=conf,
                  top_features="|".join(
                      f"{PLAIN_WORDS.get(f, f)} (z={z:.2f})"
                      for f, z in drivers))
    out = pd.concat([pd.DataFrame([sample]), pd.DataFrame(rows)],
                    ignore_index=True)
    out.to_csv(os.path.join(folder, "categorise_results.csv"), index=False)
    _sample_pdf(folder, len(bses), sample, first_rec, full, feats, med)
    return out


def _sample_pdf(folder, n, sample, rec, full, feats, med):
    """One-page PDF: overlay, distance vs threshold, position on key plots."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Image as RLImage, Table)
    from reports import ASSUME, ss, fig_feature_panels
    import common as C
    doc = SimpleDocTemplate(os.path.join(folder, "categorise_report.pdf"),
                            pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=12*mm, bottomMargin=12*mm)
    st = ss()
    story = [Paragraph(
        f"Sample of {n} image(s): <b>{sample['call_inout']}</b> of the "
        f"Batch_3 distribution (distance {sample['dist']:.2f} vs threshold "
        f"{sample['threshold']:.2f}); most similar to "
        f"<b>{sample['most_like']}</b> (confidence "
        f"{sample['confidence']:.2f}).", st["Answer"]),
        Paragraph(ASSUME, st["Small"]), Spacer(1, 4)]
    stats = pd.read_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv"))
    panels = fig_feature_panels(full, feats, stats,
                                extra=(med, "unseen"), suffix="_unseen")
    from PIL import Image as PILImage
    for p in panels[:2]:
        iw, ih = PILImage.open(p).size
        story.append(RLImage(p, width=180*mm, height=180*mm*ih/iw))
    tbl = [["Driving feature (plain words)", ""]]
    for t in sample["top_features"].split("|"):
        tbl.append([t, ""])
    story.append(Paragraph("Top drivers of the call", st["h3"]))
    story.append(Table(tbl))
    doc.build(story)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
        rep = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
        import features as FT
        kept, _ = FT.prune(F_, rep)
        full = F_[F_.subset == "full"]
        P, summ = loo_validate(full, kept, rep, exclude_ids=BAD_IMAGES)
        G = images_needed(full, kept, rep, exclude_ids=BAD_IMAGES)
        print(json.dumps(summ, indent=2, default=str))
    elif len(sys.argv) > 1:
        print(classify_folder(sys.argv[1]))
    else:
        print(__doc__)
