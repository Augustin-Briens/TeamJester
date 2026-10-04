"""Experiment: candidate features to improve Batch_2 discrimination.

Computes ~14 candidate features per image (full + L/R halves), applies the
same gates as Phase 2 (left/right corr >= 0.5, imaging-noise corr < 0.85),
then re-runs LOO + images-needed with the expanded pool.

Outputs: outputs/tables/exp_newfeat.csv, exp_newfeat_gates.csv,
         exp_loo_predictions.csv, exp_images_needed.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops, label
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.morphology import binary_erosion, skeletonize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C

FEAT_OUT = os.path.join(C.TABLE_DIR, "exp_newfeat.csv")


# --------------------------------------------------------------------------
# candidate feature functions — each takes what it needs, returns dict
# --------------------------------------------------------------------------
def pore_props(pore, um):
    lab = label(pore)
    return regionprops(lab), pore.size * (um / 1000.0) ** 2  # area_mm2


def f_elongated(pore, um):
    """Elongated-pore (crack-like) metrics: aspect>4 and width<5um."""
    props, area_mm2 = pore_props(pore, um)
    if not props or pore.mean() == 0:
        return dict(elong_pore_area_frac=np.nan, elong_pore_n_mm2=np.nan,
                    elong_pore_hfrac=np.nan, pore_aspect_p90=np.nan,
                    pore_small_n_mm2=np.nan, pore_perim_mm=np.nan)
    maj = np.array([p.axis_major_length * um for p in props])
    mnr = np.array([p.axis_minor_length * um for p in props])
    ar = np.array([p.area * um * um for p in props])
    diam = np.array([p.equivalent_diameter_area * um for p in props])
    aspect = maj / np.maximum(mnr, 1e-9)
    big = ar > 1.0  # >1 um^2 to skip specks
    elong = big & (aspect > 4) & (mnr < 5.0)
    # horizontal alignment: skimage orientation is angle between the row
    # axis (0) and the major axis -> |orient| > pi/4 means column-ward
    ors = np.array([abs(p.orientation) for p in props])
    hfrac = float((ors[elong] > np.pi / 4).mean()) if elong.any() else np.nan
    per = sum(p.perimeter * um for p in props)
    small = big & (diam >= 0.5) & (diam <= 5.0)
    return dict(
        elong_pore_area_frac=float(ar[elong].sum() / max(ar.sum(), 1e-9)),
        elong_pore_n_mm2=float(elong.sum() / area_mm2),
        elong_pore_hfrac=hfrac,
        pore_aspect_p90=float(np.percentile(aspect[big], 90))
        if big.any() else np.nan,
        pore_small_n_mm2=float(small.sum() / area_mm2),
        pore_perim_mm=float(per / 1000.0 / area_mm2),  # mm per mm^2
    )


def f_si_border(pore, si, um):
    """Share of silicon boundary pixels within 3px of a pore."""
    if si.sum() == 0:
        return dict(si_border_pore=np.nan)
    border = si & ~binary_erosion(si)
    d2p = ndi.distance_transform_edt(~pore)
    near = d2p <= 3
    return dict(si_border_pore=float(near[border].mean()))


def f_lbp(g):
    """Inlens texture: uniform-LBP(8,1) histogram entropy + flat share."""
    h, w = g.shape
    g2 = g[::2, ::2] if max(h, w) > 4000 else g
    lbp = local_binary_pattern(g2, 8, 1, method="uniform")
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10)
    p = hist / hist.sum()
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return dict(inl_lbp_ent=ent, inl_lbp_flat=float(p[0]))


def f_grad_orient(g, name):
    """Structure-tensor coherence of gradient orientation (texture
    directionality) — high = streaky, ~0 = isotropic."""
    gx = ndi.sobel(g.astype(np.float32), axis=1)
    gy = ndi.sobel(g.astype(np.float32), axis=0)
    mag = np.hypot(gx, gy)
    keep = mag > np.percentile(mag, 75)
    Jxx = float((gx[keep] ** 2).mean()); Jyy = float((gy[keep] ** 2).mean())
    Jxy = float((gx[keep] * gy[keep]).mean())
    lam1 = ((Jxx + Jyy) / 2 + np.hypot((Jxx - Jyy) / 2, Jxy))
    lam2 = ((Jxx + Jyy) / 2 - np.hypot((Jxx - Jyy) / 2, Jxy))
    coh = (lam1 - lam2) / max(lam1 + lam2, 1e-12)
    return {f"{name}_grad_coh": float(coh)}


def f_fft(g, name):
    """Radial-FFT texture scale + spectral anisotropy (subsampled 4x)."""
    h, w = g.shape
    g2 = g[::4, ::4]
    g2 = g2 - g2.mean()
    g2 = g2 * np.outer(np.hanning(g2.shape[0]), np.hanning(g2.shape[1]))
    F = np.abs(np.fft.fftshift(np.fft.fft2(g2))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(g2.shape[0], 1.0),
                         np.fft.fftfreq(g2.shape[1], 1.0), indexing="ij")
    fr = np.hypot(fx, fy)
    tot = F[fr > 0.02].sum()
    hi = F[fr > 0.15].sum() / max(tot, 1e-12)
    # orientation anisotropy of spectral energy
    ang = np.arctan2(fy, fx)
    band = (fr > 0.02) & (fr < 0.4)
    horiz = F[band & (np.abs(np.cos(ang)) > np.cos(np.pi / 4))].sum()
    vert = F[band & (np.abs(np.sin(ang)) >= np.cos(np.pi / 4))].sum()
    return {f"{name}_fft_hi": float(hi),
            f"{name}_spec_aniso": float((horiz - vert) /
                                        max(horiz + vert, 1e-12))}


def f_glcm_bulk(g, bulk_mask):
    """GLCM homogeneity/contrast inside bulk (32 levels, d=4px,
    up to 5 fully-bulk 256px crops)."""
    h, w = g.shape
    rng = np.random.default_rng(0)
    found, hs, cs = 0, [], []
    er = binary_erosion(bulk_mask, footprint=np.ones((17, 17)))
    ys, xs = np.where(er)
    if len(ys) < 5:
        return dict(bse_glcm_homog=np.nan, bse_glcm_contrast=np.nan)
    idx = rng.choice(len(ys), size=min(5, len(ys)), replace=False)
    for i in idx:
        y, x = ys[i], xs[i]
        y = min(max(y, 128), h - 129); x = min(max(x, 128), w - 129)
        patch = g[y - 128:y + 128, x - 128:x + 128]
        q = (patch // 8).astype(np.uint8)
        gl = graycomatrix(q, distances=[4], angles=[0, np.pi / 2],
                          levels=32, symmetric=True, normed=True)
        hs.append(graycoprops(gl, "homogeneity")[0].mean())
        cs.append(graycoprops(gl, "contrast")[0].mean())
    return dict(bse_glcm_homog=float(np.mean(hs)),
                bse_glcm_contrast=float(np.mean(cs)))


# --------------------------------------------------------------------------
def compute_one(rec, subset):
    z = C.load_masks(rec["batch"], rec["image_id"])
    um, _ = C.pixel_size_um(rec["bse"])
    pore = z["pore"] if subset == "full" else z[f"pore_{subset}"]
    si = z["silicon"] if subset == "full" else z[f"silicon_{subset}"]
    fe = {}
    fe.update(f_elongated(pore, um))
    fe.update(f_si_border(pore, si, um))
    # detector crops matching the half
    def crop(g):
        if subset == "full":
            return g
        w2 = pore.shape[1]
        return g[:, :w2] if subset == "L" else g[:, -w2:]
    if rec["bse"]:
        g, _ = C.load_gray(rec["bse"])
        g = crop(g)
        fe.update(f_fft(g, "bse"))
        fe.update(f_grad_orient(g, "bse"))
        graphite = ~pore & ~si
        if subset == "full" or subset in ("L", "R"):
            fe.update(f_glcm_bulk(g, graphite))
    if rec["inlens"]:
        g, _ = C.load_gray(rec["inlens"])
        g = crop(g)
        fe.update(f_lbp(g))
        fe.update(f_grad_orient(g, "inl"))
    if rec["topo"]:
        g, _ = C.load_gray(rec["topo"])
        g = crop(g)
        fe.update(f_fft(g, "etd"))
    return fe


def main():
    locs = C.find_locations()
    rows = []
    for rec in locs:
        for subset in ("full", "L", "R"):
            try:
                fe = compute_one(rec, subset)
            except Exception as e:
                C.log(f"{rec['image_id']} {subset}: {e}")
                fe = {}
            rows.append(dict(batch=rec["batch"], image_id=rec["image_id"],
                             subset=subset, **fe))
        C.log(f"done {rec['batch']} {rec['image_id']}")
    df = pd.DataFrame(rows)
    df.to_csv(FEAT_OUT, index=False)
    print(df[df.subset == "full"].describe().T[["mean", "std"]])

    # --- gates --------------------------------------------------------------
    full = df[df.subset == "full"].set_index("image_id")
    L = df[df.subset == "L"].set_index("image_id")
    R = df[df.subset == "R"].set_index("image_id")
    feats = [c for c in df.columns if c not in ("batch", "image_id", "subset")]
    qg = pd.read_csv(os.path.join(C.TABLE_DIR, "quality_guards.csv"))
    noise = qg.set_index("image_id")["noise_mad"]
    F0 = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv")).set_index(
        "image_id")
    rep_old = pd.read_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"))
    gate_rows = []
    for f in feats:
        x, xl, xr = full[f], L[f], R[f]
        both = xl.notna() & xr.notna()
        lr = float(np.corrcoef(xl[both], xr[both])[0, 1]) if both.sum() > 5 \
            else np.nan
        nz = float(np.corrcoef(x[noise.notna()], noise[noise.notna()])[0, 1]) \
            if x.notna().sum() > 5 else np.nan
        keep = (pd.notna(lr) and lr >= 0.5 and
                (pd.isna(nz) or abs(nz) < 0.85))
        gate_rows.append(dict(feature=f, lr_corr=lr, noise_corr=nz, keep=keep))
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(os.path.join(C.TABLE_DIR, "exp_newfeat_gates.csv"),
                 index=False)
    print(gates.to_string(index=False))

    kept_new = gates[gates.keep].feature.tolist()
    # merge into the LOO machinery -------------------------------------------
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    F_ = F_.merge(full.reset_index()[["image_id"] + feats], on="image_id")
    kept = pd.read_csv(os.path.join(C.TABLE_DIR, "baseline_stats.csv"))[
        "feature"].tolist()
    rep = rep_old.copy()
    rep = pd.concat([rep, gates[["feature", "lr_corr"]]], ignore_index=True)
    kept_ext = kept + kept_new
    import categorise as CAT
    P, summ = CAT.loo_validate(F_, kept_ext, rep, tag="exp")
    G = CAT.images_needed(F_, kept_ext, rep)
    G.to_csv(os.path.join(C.TABLE_DIR, "exp_images_needed.csv"), index=False)
    print("\nLOO acc:", (P.batch == P.pred_batch).mean())
    print(pd.crosstab(P.batch, P.pred_batch))
    print("\nsummary:", summ["false_alarm_rate"],
          summ["shuffled_accuracy_mean"])
    print("\nimages needed:\n", G[G.n_images.isin([5, 7])])
    # which features got picked in folds
    from collections import Counter
    cnt = Counter()
    for s in P.feats:
        for f in s.split("|"):
            cnt[f] += 1
    print("\nfeature pick counts:", cnt.most_common(15))


if __name__ == "__main__":
    main()
