"""Phase 1 - 3-phase segmentation (pore / assumed-graphite / assumed-silicon)
plus every validation check the spec asks for.

Method (classical, transparent, identical settings for every image):
    1. grey = median of RGB; crop columns/rows with >5% non-grey pixels
       (left/right marker stripes).
    2. normalise to [0,1] by the 1st-99.5th percentiles; gaussian sigma 1.5 px.
    3. pore  : s < t_lo where t_lo = lower cut of 3-class multi-Otsu.
               cleaned (remove < MIN_PORE_PX, silicon wins overlaps),
               then refined with Inlens: pixels within INLENS_DIL px of a
               BSE pore that are below the Inlens dark threshold join the
               pore set (fine cracks Inlens resolves better).
    4. silicon: s > t_b where t_b = valley between bulk peak and bright peak,
               fallback = upper multi-Otsu cut; opening r2, remove <150 px,
               fill holes, watershed split into particles.
    5. graphite = remainder.

Usage: python3 segment.py            # segments all + validation + overlays
"""
import os
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.morphology import (binary_opening, disk, remove_small_objects,
                                binary_dilation)
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.draw import disk as draw_disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------
def find_bright_threshold(s):
    """Valley between bulk peak and bright peak; fallback multi-Otsu upper."""
    h, edges = np.histogram(s, bins=C.HIST_BINS, range=(0, 1))
    hs = ndi.gaussian_filter1d(h.astype(float), C.HIST_SMOOTH)
    cents = (edges[:-1] + edges[1:]) / 2.0
    # bulk peak = histogram maximum in the mid-grey range (0.06 - 0.75)
    lo, hi = int(0.06 * C.HIST_BINS), int(0.75 * C.HIST_BINS)
    bulk = lo + int(np.argmax(hs[lo:hi]))
    cand = []
    for i in range(bulk + C.BRIGHT_MIN_SEP, C.HIST_BINS - 7):
        if hs[i] > hs[i - 1] and hs[i] >= hs[i + 1]:
            prom = hs[i] - hs[bulk:i + 1].min()
            if prom > C.BRIGHT_MIN_PROM * h.sum() and hs[i] > C.BRIGHT_MIN_PROM * h.sum():
                cand.append((i, prom))
    if cand:
        bp = max(cand, key=lambda c: c[1])[0]
        t = cents[bulk + int(np.argmin(hs[bulk:bp]))]
        return float(t), "valley", float(cents[bulk]), float(cents[bp])
    t = threshold_multiotsu(s, classes=3, nbins=256)[1]
    return float(t), "multiotsu_fallback", float(cents[bulk]), np.nan


def thresholds(s):
    """(t_lo pore cut, t_hi multiotsu upper, t_b bright cut, method)."""
    ss = s[::4, ::4]  # decimated for speed; threshold is stable to this
    t_lo, t_hi = threshold_multiotsu(ss, classes=3, nbins=256)
    t_b, method, bx, px = find_bright_threshold(ss)
    return float(t_lo), float(t_hi), float(t_b), method


def pores_at(s, t_lo, bright_mask):
    pm = (s < t_lo) & ~bright_mask
    return remove_small_objects(pm, C.MIN_PORE_PX)


def inlens_refine(pore_mask, inlens_gray):
    """Add Inlens-dark pixels within INLENS_DIL px of a BSE pore."""
    if inlens_gray is None:
        return pore_mask, np.nan
    s_il, _, _ = C.normalise(inlens_gray)
    t_il = threshold_otsu(s_il[::4, ::4])
    il_dark = s_il < t_il
    near = binary_dilation(pore_mask, disk(C.INLENS_DIL))
    refined = pore_mask | (il_dark & near)
    return remove_small_objects(refined, C.MIN_PORE_PX), float(t_il)


def bright_particles(s, t_b):
    bw = s > t_b
    bw = binary_opening(bw, disk(C.OPEN_RADIUS))
    bw = remove_small_objects(bw, C.MIN_AREA_PX)
    bw = ndi.binary_fill_holes(bw)
    dist = ndi.distance_transform_edt(bw)
    coords = peak_local_max(dist, min_distance=C.WS_MIN_DIST,
                            threshold_abs=C.WS_MIN_DEPTH, labels=bw)
    markers = np.zeros(bw.shape, np.int32)
    for j, (r, c) in enumerate(coords, 1):
        markers[r, c] = j
    markers = label(markers > 0)
    lab = watershed(-dist, markers, mask=bw)
    return bw, lab.astype(np.int32)


def segment(g, inlens=None, with_labels=True):
    """Full segmentation of one image. Returns dict of results."""
    s, p1, p995 = C.normalise(g)
    t_lo, t_hi, t_b, method = thresholds(s)
    bw, lab = bright_particles(s, t_b)
    pm = pores_at(s, t_lo, bw)
    t_il = np.nan
    if inlens is not None:
        pm, t_il = inlens_refine(pm, inlens)
    info = dict(t_lo=t_lo, t_hi_multiotsu=t_hi, t_bright=t_b,
                bright_method=method, t_inlens=t_il,
                p_lo=p1, p_hi=p995,
                pore_frac=float(pm.mean()), silicon_frac=float(bw.mean()),
                graphite_frac=float(1 - pm.mean() - bw.mean()))
    return dict(s=s, pore=pm, silicon=bw, labels=lab if with_labels else None,
                info=info)


# ---------------------------------------------------------------------------
# Run over the dataset
# ---------------------------------------------------------------------------
def load_detector(rec):
    if rec["inlens"] and os.path.exists(rec["inlens"]):
        return C.load_gray(rec["inlens"])[0]
    return None


def run_segmentation(locs=None, halves=False):
    """Segment every location (full image; optionally also L/R halves).
    Saves masks npz; returns per-image info DataFrame."""
    locs = locs or C.find_locations()
    rows = []
    for rec in locs:
        b, iid = rec["batch"], rec["image_id"]
        mp = C.mask_path(b, iid)
        if os.path.exists(mp):
            continue
        g, crop = C.load_gray(rec["bse"])
        il = load_detector(rec)
        r = segment(g, il)
        um, src = C.pixel_size_um(rec["bse"])
        save = dict(pore=r["pore"], silicon=r["silicon"],
                    labels=r["labels"].astype(np.int32))
        # halves, segmented independently (their own thresholds)
        if halves:
            h, w = g.shape
            for name, sl in (("L", slice(0, w // 2)), ("R", slice(w // 2, w))):
                il_h = il[:, sl] if il is not None else None
                rh = segment(g[:, sl], il_h)
                save[f"pore_{name}"] = rh["pore"]
                save[f"silicon_{name}"] = rh["silicon"]
                save[f"labels_{name}"] = rh["labels"].astype(np.int32)
                for k, v in rh["info"].items():
                    r["info"][f"{k}_{name}"] = v
        C.save_masks(b, iid, **save)
        info = dict(batch=b, image_id=iid, um_per_px=um, px_source=src,
                    crop=crop, **r["info"])
        rows.append(info)
        C.log(f"segmented {b}/{iid} pore={r['info']['pore_frac']:.3f} "
              f"si={r['info']['silicon_frac']:.3f} {r['info']['bright_method']}")
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------
def threshold_sensitivity(rec):
    """Shift each threshold +/-10%, report change in each phase fraction."""
    g, _ = C.load_gray(rec["bse"])
    s, _, _ = C.normalise(g)
    t_lo, _, t_b, _ = thresholds(s)
    rows = []
    for name, base in (("t_lo(pore)", t_lo), ("t_b(silicon)", t_b)):
        for shift in C.SENS_SHIFTS:
            t = base * (1 + shift)
            if name.startswith("t_lo"):
                bw, _ = bright_particles(s, t_b)
                m = pores_at(s, t, bw)
                rows.append(dict(param=name, shift=shift, pore=m.mean(),
                                 silicon=bw.mean(),
                                 graphite=1 - m.mean() - bw.mean()))
            else:
                bw, _ = bright_particles(s, t)
                m = pores_at(s, t_lo, bw)
                rows.append(dict(param=name, shift=shift, pore=m.mean(),
                                 silicon=bw.mean(),
                                 graphite=1 - m.mean() - bw.mean()))
    return rows


def cross_detector_agreement(rec):
    """Pore mask from Inlens alone vs the BSE(+Inlens) pore mask: IoU, and
    silicon mask vs a top-tail threshold of BSE (sanity)."""
    z = C.load_masks(rec["batch"], rec["image_id"])
    pore_bse = z["pore"]
    if rec["inlens"] is None:
        return dict(iou_pore=np.nan)
    il = load_detector(rec)
    s_il, _, _ = C.normalise(il)
    t_il = threshold_otsu(s_il[::4, ::4])
    pore_il = s_il < t_il
    pore_il = remove_small_objects(pore_il, C.MIN_PORE_PX)
    # shape may differ by crop; align to min common shape
    h = min(pore_il.shape[0], pore_bse.shape[0])
    w = min(pore_il.shape[1], pore_bse.shape[1])
    a, b = pore_bse[:h, :w], pore_il[:h, :w]
    iou = (a & b).sum() / max((a | b).sum(), 1)
    dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    return dict(iou_pore=float(iou), dice_pore=float(dice),
                pore_frac_inlens=float(pore_il.mean()))


def quality_guards(rec):
    """Contrast / noise / sharpness / shading per image."""
    g, _ = C.load_gray(rec["bse"])
    z = C.load_masks(rec["batch"], rec["image_id"])
    pore, si = z["pore"], z["silicon"]
    h = min(g.shape[0], pore.shape[0]); w = min(g.shape[1], pore.shape[1])
    g, pore, si = g[:h, :w], pore[:h, :w], si[:h, :w]
    gr = ~(pore | si)
    mu_b, sd_b = g[gr].mean(), g[gr].std()
    mu_si = g[si].mean() if si.any() else np.nan
    contrast = (mu_si - mu_b) / max(sd_b, 1e-9)      # Cohen-d style
    # noise: MAD of high-frequency residual inside graphite
    hp = g - ndi.median_filter(g, 5)
    noise = float(np.median(np.abs(hp[gr] - np.median(hp[gr]))) * 1.4826)
    # sharpness: mean gradient magnitude
    gy, gx = np.gradient(g)
    sharp = float(np.hypot(gy, gx).mean())
    # shading: peak-to-peak of a heavily smoothed field
    bg = ndi.uniform_filter(g, 256)
    shading = float(np.percentile(bg, 99) - np.percentile(bg, 1))
    return dict(si_bulk_contrast=float(contrast), noise_mad=noise,
                sharpness=sharp, shading_range=shading)


# documented absolute guards (chosen at the natural gaps in the measured
# values: contrast 2.27 -> next 3.40; noise 11.9 -> next 10.4; sharpness
# 11.4 -> next 10.5; shading gap > 55)
GUARDS = dict(si_bulk_contrast=2.8, noise_mad=11.2, sharpness=11.0,
              shading_range=55.0)


def flag_images(qdf):
    """Flag an image when any guard is on the bad side of its threshold."""
    out = qdf.copy()
    hits = []
    for i, r in out.iterrows():
        bad = []
        if r.si_bulk_contrast < GUARDS["si_bulk_contrast"]:
            bad.append("low Si/bulk contrast")
        if r.noise_mad > GUARDS["noise_mad"]:
            bad.append("high noise")
        if r.sharpness > GUARDS["sharpness"]:
            bad.append("high noise-driven sharpness")
        if r.shading_range > GUARDS["shading_range"]:
            bad.append("strong shading")
        hits.append("; ".join(bad))
    out["flag_reason"] = hits
    out["flagged"] = out.flag_reason != ""
    out["flag_score"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Overlays / contact sheets
# ---------------------------------------------------------------------------
def overlay_image(rec, scale=4):
    """Colour overlay: pore red, silicon orange edge, graphite dimmed."""
    g, _ = C.load_gray(rec["bse"])
    z = C.load_masks(rec["batch"], rec["image_id"])
    pore, si = z["pore"], z["silicon"]
    h = min(g.shape[0], pore.shape[0]); w = min(g.shape[1], pore.shape[1])
    g, pore, si = g[:h, :w], pore[:h, :w], si[:h, :w]
    base = np.repeat((g / g.max() * 255).astype(np.uint8)[..., None], 3, axis=2)
    over = base.copy()
    over[pore] = (220, 40, 40)
    # silicon: interior dim orange, boundary bright
    edge = si & ~ndi.binary_erosion(si)
    over[si] = (over[si] * 0.4 + np.array([255, 160, 0]) * 0.6).astype(np.uint8)
    over[edge] = (255, 220, 0)
    out = over[::scale, ::scale]
    return out


def write_overlays(locs):
    from PIL import Image
    for rec in locs:
        out = overlay_image(rec)
        Image.fromarray(out).save(os.path.join(
            C.OVL_DIR, f"overlay_{rec['batch']}_{rec['image_id']}.png"))
    # contact sheet per batch (PIL, labels drawn directly)
    from PIL import ImageDraw
    for b in C.BATCHES:
        ims = [r for r in locs if r["batch"] == b]
        thumbs = [(r["image_id"], overlay_image(r, scale=8)) for r in ims]
        h = max(t[1].shape[0] for t in thumbs)
        w = max(t[1].shape[1] for t in thumbs)
        cols = 4
        rows = int(np.ceil(len(thumbs) / cols))
        sheet = Image.new("RGB", (cols * w, rows * (h + 18)), "white")
        dr = ImageDraw.Draw(sheet)
        for i, (iid, o) in enumerate(thumbs):
            x = (i % cols) * w
            y = (i // cols) * (h + 18) + 18
            sheet.paste(Image.fromarray(o), (x, y))
            dr.text((x + 4, y - 14), iid, fill="black")
        sheet.save(os.path.join(C.OVL_DIR, f"contact_{b}.png"))
    C.log("overlays + contact sheets written")


# ---------------------------------------------------------------------------
# Point-count validation sheet
# ---------------------------------------------------------------------------
def point_count_sheet(locs, n_total=300, crop=64):
    """Sample n_total pixels spread evenly across images; write xlsx with a
    crop PNG per point, the predicted class, and an empty human column."""
    from PIL import Image
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    rng = np.random.default_rng(20261003)
    crops_dir = os.path.join(C.PC_DIR, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    wb = Workbook(); ws = wb.active; ws.title = "point_count"
    header = ["point_id", "batch", "image_id", "x_px", "y_px",
              "crop_png", "predicted_class", "human_class"]
    ws.append(header)
    ws.column_dimensions["F"].width = 22
    per = {r["image_id"]: max(1, n_total // len(locs)) for r in locs}
    # distribute remainder
    extra = n_total - sum(per.values())
    i = 0
    for k in sorted(per):
        if extra <= 0:
            break
        per[k] += 1; extra -= 1; i += 1
    pid = 0
    for rec in locs:
        b, iid = rec["batch"], rec["image_id"]
        z = C.load_masks(b, iid)
        g, _ = C.load_gray(rec["bse"])
        pore, si = z["pore"], z["silicon"]
        h = min(g.shape[0], pore.shape[0]); w = min(g.shape[1], pore.shape[1])
        xs = rng.integers(crop // 2, w - crop // 2, per[iid])
        ys = rng.integers(crop // 2, h - crop // 2, per[iid])
        for x, y in zip(xs, ys):
            pid += 1
            cls = "pore" if pore[y, x] else ("silicon" if si[y, x] else "graphite")
            c0, c1 = x - crop // 2, x + crop // 2
            r0, r1 = y - crop // 2, y + crop // 2
            cimg = np.repeat(g[r0:r1, c0:c1].astype(np.uint8)[..., None], 3, 2)
            # mark the central pixel with a 3x3 cross
            cc = crop // 2
            cimg[cc - 1:cc + 2, cc] = (255, 0, 255)
            cimg[cc, cc - 1:cc + 2] = (255, 0, 255)
            fn = os.path.join(crops_dir, f"pt_{pid:03d}.png")
            Image.fromarray(cimg).save(fn)
            ws.append([pid, b, iid, int(x), int(y), f"pt_{pid:03d}.png",
                       cls, ""])
            img = XLImage(fn); img.width = crop; img.height = crop
            ws.add_image(img, f"F{ws.max_row}")
            ws.row_dimensions[ws.max_row].height = crop * 0.75
    out = os.path.join(C.PC_DIR, "point_count_validation.xlsx")
    wb.save(out)
    C.log(f"point-count sheet: {out} ({pid} points)")
    return out


# ---------------------------------------------------------------------------
def main():
    locs = C.find_locations()
    C.log(f"{len(locs)} locations")
    df = run_segmentation(locs, halves=True)
    df.to_csv(os.path.join(C.TABLE_DIR, "segment_info.csv"), index=False)
    C.log("segmentation done")

    # validation ----
    sens_rows, xd_rows, q_rows = [], [], []
    for rec in locs:
        for r in threshold_sensitivity(rec):
            r.update(batch=rec["batch"], image_id=rec["image_id"])
            sens_rows.append(r)
        xd = cross_detector_agreement(rec)
        xd.update(batch=rec["batch"], image_id=rec["image_id"])
        xd_rows.append(xd)
        q = quality_guards(rec)
        q.update(batch=rec["batch"], image_id=rec["image_id"])
        q_rows.append(q)
        C.log(f"checks {rec['batch']}/{rec['image_id']}")
    pd.DataFrame(sens_rows).to_csv(
        os.path.join(C.TABLE_DIR, "threshold_sensitivity.csv"), index=False)
    pd.DataFrame(xd_rows).to_csv(
        os.path.join(C.TABLE_DIR, "cross_detector.csv"), index=False)
    qdf = flag_images(pd.DataFrame(q_rows))
    qdf.to_csv(os.path.join(C.TABLE_DIR, "quality_guards.csv"), index=False)
    C.log(f"flagged images: "
          f"{qdf[qdf.flagged][['batch','image_id']].values.tolist()}")

    # left/right agreement from half-masks
    rep_rows = []
    for rec in locs:
        z = C.load_masks(rec["batch"], rec["image_id"])
        row = dict(batch=rec["batch"], image_id=rec["image_id"])
        for ph in ("pore", "silicon"):
            row[f"{ph}_L"] = float(z[f"{ph}_L"].mean())
            row[f"{ph}_R"] = float(z[f"{ph}_R"].mean())
        rep_rows.append(row)
    rep = pd.DataFrame(rep_rows)
    rep.to_csv(os.path.join(C.TABLE_DIR, "half_agreement.csv"), index=False)

    write_overlays(locs)
    point_count_sheet(locs)
    C.log("Phase 1 complete")


if __name__ == "__main__":
    main()
