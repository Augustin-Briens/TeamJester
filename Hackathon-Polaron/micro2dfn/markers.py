"""Marker extraction — stereological + morphology measurements per image.

Conventions
-----------
* Lengths in um, areas in fractions, counts per Mpx.
* `si_*` markers are computed on OBJECT-FILTERED si_particle pixels only;
  bright_fine objects are tracked separately (guard rail, not silicon).
* skimage `orientation`: angle between ROW axis and ellipse major axis;
  angle vs foil = pi/2 - |orientation|.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, gaussian_filter)
from skimage.measure import label, regionprops

from . import config
from .io import BSEImage


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _chord_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Run lengths of True along axis (1=rows/horizontal, 0=cols/vertical)."""
    m = mask if axis == 1 else mask.T
    padded = np.pad(m.astype(np.int8), ((0, 0), (1, 1)), constant_values=0)
    diff = np.diff(padded, axis=1)
    starts = np.argwhere(diff == 1)
    ends = np.argwhere(diff == -1)
    return (ends[:, 1] - starts[:, 1]).astype(np.float64)


def _interface_transitions(seg: np.ndarray, a: int, b: int) -> int:
    """4-connected pixel-pair transitions between phases a and b."""
    def pairs(x, y):
        return int(((x == a) & (y == b)).sum() + ((x == b) & (y == a)).sum())
    return (pairs(seg[:, :-1], seg[:, 1:]) +
            pairs(seg[:-1, :], seg[1:, :]))


def _two_point_corr_length(mask: np.ndarray, um: float,
                           max_r_px: int = 400) -> float:
    """Correlation length (um): radial lag where autocovariance < 1/e."""
    f = mask.astype(np.float32)
    p = f.mean()
    if p < 1e-4 or p > 1 - 1e-4:
        return 0.0
    F = np.fft.rfft2(f)
    ac = np.fft.irfft2(F * np.conj(F), s=f.shape) / f.size
    ac = np.fft.fftshift(ac[:, :ac.shape[1]], axes=0)
    cy = f.shape[0] // 2
    ys = np.arange(f.shape[0]) - cy
    xs = np.arange(ac.shape[1])
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.hypot(yy, xx).astype(np.int32).ravel()
    keep = r <= max_r_px
    sums = np.bincount(r[keep], weights=ac.ravel()[keep])
    cnt = np.bincount(r[keep])
    cov = sums / np.maximum(cnt, 1) - p * p
    norm = cov / max(p * (1 - p), 1e-12)
    below = np.where(norm < 1.0 / np.e)[0]
    return float(below[0]) * um if len(below) else float(max_r_px) * um


def _geodesic_tau(pore: np.ndarray, axis: str = "vertical"
                  ) -> tuple[float, float]:
    """2-D tortuosity proxy through the pore network.

    axis="vertical": wavefront from the top row (through-plane direction
    in the image). axis="horizontal": wavefront from the left column
    (in-plane direction). tau = mean geodesic depth / straight-line depth.
    Returns (tau, fraction of pore pixels connected to the seed edge).
    """
    from skimage.graph import MCP_Geometric
    g = config.TORT_DOWNSAMPLE
    m = pore if axis == "vertical" else pore.T
    h = m.shape[0] // g * g
    w = m.shape[1] // g * g
    small = m[:h, :w].reshape(h // g, g, w // g, g).mean((1, 3)) > 0.5
    if small.sum() < 50 or not small[0].any():
        return float("nan"), 0.0
    costs = np.where(small, 1.0, np.inf)
    mcp = MCP_Geometric(costs, fully_connected=False)
    starts = [(0, c) for c in np.where(small[0])[0]]
    dist, _ = mcp.find_costs(starts)
    connected = small & np.isfinite(dist)
    reach = float(connected.sum() / small.sum())
    finite = dist[connected]
    if len(finite) == 0:
        return float("inf"), reach
    mean_geo = float(finite.mean())
    ideal = float(np.where(connected)[0].mean())
    return mean_geo / max(ideal, 1e-6), reach



def _swelling_map(pore, bulk, si, um) -> dict:
    """Local swelling budget: pore - 2.8*Si - 0.1*bulk per ~12.5 um window."""
    w = max(8, int(round(config.BUDGET_WINDOW_PX * config.PIXEL_UM / um)))
    s = max(4, int(round(config.BUDGET_STRIDE_PX * config.PIXEL_UM / um)))
    h, wid = pore.shape
    vals = []
    for y in range(0, h - w + 1, s):
        for x in range(0, wid - w + 1, s):
            v = (pore[y:y + w, x:x + w].mean()
                 - config.SI_EXPANSION * si[y:y + w, x:x + w].mean()
                 - config.GR_EXPANSION * bulk[y:y + w, x:x + w].mean())
            vals.append(v)
    vals = np.asarray(vals)
    return {
        "swelling_budget_min": float(np.percentile(vals, 5)),
        "swelling_deficit_area_frac": float((vals < 0).mean()),
    }


def _tile_jackknife(mask: np.ndarray, grid: int) -> float:
    """95% CI half-width of the phase fraction across tiles."""
    h, w = mask.shape
    fracs = [mask[i * h // grid:(i + 1) * h // grid,
                  j * w // grid:(j + 1) * w // grid].mean()
             for i in range(grid) for j in range(grid)]
    fracs = np.asarray(fracs)
    return float(1.96 * fracs.std(ddof=1) / np.sqrt(len(fracs)))


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    return v[np.searchsorted(cum, q)]


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def extract_markers(im: BSEImage, seg: np.ndarray,
                    objects: pd.DataFrame,
                    si_part: np.ndarray,
                    bright_fine: np.ndarray) -> dict:
    """Full marker dictionary for one image.

    objects: classified object table for THIS image.
    si_part / bright_fine: masks rebuilt from the classifier.
    """
    um = im.pixel_um
    f: dict[str, float] = {}
    n = seg.size
    pore = seg == config.PORE
    bulk = seg == config.BULK
    raw = im.raw

    f["batch"] = im.batch
    f["image_id"] = im.image_id
    f["h_px"], f["w_px"] = seg.shape
    f["n_px"] = n

    # ---------------- A. composition --------------------------------------
    f["pore_frac_resolved"] = pore.sum() / n
    f["porosity_est"] = f["pore_frac_resolved"] + (
        config.SUBRES_POROSITY_LO + config.SUBRES_POROSITY_HI) / 2
    f["si_frac_total"] = si_part.sum() / n          # classified particles
    f["bright_frac_raw"] = (seg == config.SI).sum() / n  # raw, incl. fines
    f["bright_fine_frac"] = bright_fine.sum() / n
    f["gr_frac"] = bulk.sum() / n
    solid = 1.0 - f["pore_frac_resolved"]
    f["si_frac_of_solid"] = f["si_frac_total"] / max(solid, 1e-9)

    # ---------------- B. silicon particles --------------------------------
    img_obj = objects[objects["image_id"] == im.image_id]
    sub = img_obj[img_obj["kind"] == "si_particle"]
    fines = img_obj[img_obj["kind"] == "bright_fine"]
    f["si_per_mpx"] = len(sub) / (n / 1e6)
    f["bright_fine_per_mpx"] = len(fines) / (n / 1e6)
    if len(sub):
        d = sub["eq_diam_um"].to_numpy()
        a = sub["area_um2"].to_numpy()
        f["si_d10_num_um"] = float(np.percentile(d, 10))
        f["si_d50_num_um"] = float(np.percentile(d, 50))
        f["si_d90_num_um"] = float(np.percentile(d, 90))
        f["si_d50_aw_um"] = float(_weighted_quantile(d, a, 0.5))
        f["si_d90_aw_um"] = float(_weighted_quantile(d, a, 0.9))
        f["si_dmean_aw_um"] = float(np.average(d, weights=a))
        interior = ~sub["touches_border"].to_numpy()
        f["si_dmax_um"] = float(d[interior].max()) if interior.any() \
            else float(d.max())
        for cut in config.LARGE_SI_CUTOFFS_UM:
            big = d >= cut
            f[f"si_frac_gt{int(cut)}um"] = float(
                a[big].sum() / a.sum()) if big.any() else 0.0
        # 3-D sphere-equivalent radius: mean section = 0.785 * true diam
        f["si_radius_3d_um"] = f["si_dmean_aw_um"] / (2 * config.SECTION_FACTOR)
        f["si_solidity_mean"] = float(sub["solidity"].mean())
        f["si_aspect_mean"] = float(sub["aspect"].mean())

        cents = sub[["centroid_y_um", "centroid_x_um"]].to_numpy()
        if len(sub) > 2:
            tree = spatial.KDTree(cents)
            nn = tree.query(cents, k=2)[0][:, 1]
            lam = len(sub) / (n * um * um)          # particles per um2
            exp_nn = 0.5 / np.sqrt(lam)
            # Donnelly boundary correction (spatstat convention)
            P = 2 * (seg.shape[0] + seg.shape[1]) * um
            exp_nn += (0.0514 + 0.041 / np.sqrt(len(sub))) * P / len(sub)
            f["si_clustering_R"] = float(nn.mean() / exp_nn)
            f["si_nn_dist_um"] = float(nn.mean())
        else:
            f["si_clustering_R"] = f["si_nn_dist_um"] = np.nan

        strips = np.array_split(np.arange(seg.shape[1]),
                                config.N_PATCH_STRIPS)
        sf = np.asarray([si_part[:, s].mean() for s in strips])
        f["si_patchiness"] = float(sf.std() / max(sf.mean(), 1e-9))
    else:
        for k in ("si_d10_num_um", "si_d50_num_um", "si_d90_num_um",
                  "si_d50_aw_um", "si_d90_aw_um", "si_dmean_aw_um",
                  "si_dmax_um", "si_radius_3d_um", "si_solidity_mean",
                  "si_aspect_mean", "si_clustering_R", "si_nn_dist_um",
                  "si_patchiness"):
            f[k] = np.nan
        for cut in config.LARGE_SI_CUTOFFS_UM:
            f[f"si_frac_gt{int(cut)}um"] = np.nan

    # Si <-> pore proximity (continuous — survives resolution limits)
    si_boundary = si_part & ~binary_erosion(si_part)
    near_pore = binary_dilation(pore, iterations=config.CONTACT_DILATION_PX)
    if si_boundary.any():
        f["si_pore_contact"] = float(
            (si_boundary & near_pore).sum() / si_boundary.sum())
        dist_pore = distance_transform_edt(~pore)
        f["si_pore_dist_mean_um"] = float(dist_pore[si_boundary].mean()) * um
    else:
        f["si_pore_contact"] = f["si_pore_dist_mean_um"] = np.nan
    # per-particle boundary-coverage distribution (ASSB-style coverage%):
    # continuous replacement for the binary enclosed test
    if len(sub):
        lab_si = label(si_part)
        enclosed = 0
        coverages = []
        for r in regionprops(lab_si):
            isb = si_boundary[r.coords[:, 0], r.coords[:, 1]]
            nb = int(isb.sum())
            cov = float((isb & near_pore[r.coords[:, 0],
                                         r.coords[:, 1]]).sum() / nb) \
                if nb else 0.0
            coverages.append(cov)
            if cov == 0.0:
                enclosed += 1
        coverages = np.asarray(coverages)
        f["si_enclosed_share"] = enclosed / len(sub)
        f["si_coverage_mean"] = float(coverages.mean())
        f["si_coverage_p10"] = float(np.percentile(coverages, 10))
        f["si_lowcoverage_share"] = float(
            (coverages < config.LOW_COVERAGE_CUT).mean())
    else:
        f["si_enclosed_share"] = f["si_coverage_mean"] = np.nan
        f["si_coverage_p10"] = f["si_lowcoverage_share"] = np.nan
    f["si_accessible_frac"] = 1.0 - f["si_enclosed_share"] \
        if np.isfinite(f["si_enclosed_share"]) else np.nan

    # ---------------- C. graphite matrix ----------------------------------
    A_um2 = n * um * um
    t_sb = _interface_transitions(seg, config.SI, config.BULK)
    t_sp = _interface_transitions(seg, config.SI, config.PORE)
    t_ps = (_interface_transitions(seg, config.PORE, config.BULK)
            + t_sp)
    # S_V = (4/pi) * L_A;  L_A = (pi/4) * transitions*px / area  =>
    # S_V = transitions * px / area  (boundary length per area, um^-1)
    f["si_sv_um_inv"] = (t_sb + t_sp) * um / A_um2
    f["gr_sv_um_inv"] = t_ps * um / A_um2
    f["gr_radius_eff_um"] = (3.0 * f["gr_frac"] / f["gr_sv_um_inv"]
                             if f["gr_sv_um_inv"] > 0 else np.nan)
    bh = _chord_lengths(bulk, axis=1)
    bv = _chord_lengths(bulk, axis=0)
    f["gr_chord_ratio_xz"] = ((bh.mean() if len(bh) else 0.0)
                              / max(bv.mean() if len(bv) else 1e-9, 1e-9))

    # ---------------- D. pores & transport --------------------------------
    pore_lab = label(pore)
    pore_props = [r for r in regionprops(pore_lab)
                  if r.area >= config.MIN_FEATURE_PX]
    if pore_props:
        pd_um = np.asarray([np.sqrt(4 * r.area / np.pi) * um
                            for r in pore_props])
        f["pore_d50_um"] = float(np.median(pd_um))
    else:
        f["pore_d50_um"] = np.nan
    f["pore_count_per_mpx"] = len(pore_props) / (n / 1e6)

    big_px = config.BIG_VOID_IMAGE_FRAC * n
    big_void = crack_h = crack_v = round_pore = 0
    vert_num = vert_den = 0.0
    for r in pore_props:
        maj, mnr = r.axis_major_length, max(r.axis_minor_length, 1e-9)
        aspect = maj / mnr
        theta_foil = np.pi / 2 - abs(r.orientation)   # 0=horizontal
        vert_num += abs(np.sin(theta_foil)) * r.area
        vert_den += r.area
        if r.area > big_px:
            big_void += r.area
        elif aspect > config.CRACK_ASPECT:
            if np.degrees(theta_foil) < config.CRACK_ANGLE_DEG:
                crack_h += r.area
            else:
                crack_v += r.area
        else:
            round_pore += r.area
    f["big_void_frac"] = big_void / n
    f["crack_frac_h"] = crack_h / n
    f["crack_frac_v"] = crack_v / n
    f["round_pore_frac"] = round_pore / n
    f["pore_verticality_aw"] = vert_num / max(vert_den, 1)

    ch = _chord_lengths(pore, axis=1)
    cv = _chord_lengths(pore, axis=0)
    f["pore_chord_h_um"] = ch.mean() * um if len(ch) else 0.0
    f["pore_chord_v_um"] = cv.mean() * um if len(cv) else 0.0
    f["pore_anisotropy"] = f["pore_chord_h_um"] / max(f["pore_chord_v_um"],
                                                     1e-9)

    tau_v, reach_v = _geodesic_tau(pore, "vertical")
    tau_h, _ = _geodesic_tau(pore, "horizontal")
    f["tau_vertical_proxy"] = tau_v
    f["tau_horizontal_proxy"] = tau_h
    f["tau_anisotropy"] = (tau_v / tau_h
                           if np.isfinite(tau_v) and np.isfinite(tau_h)
                           and tau_h > 0 else np.nan)
    f["percolation_reach_2d"] = reach_v

    # Largest connected pore region (direct label; the FDM diffusion solve
    # was removed — no 2-D section ever has a spanning pore path, so tau
    # always returned inf. See Limitations / archive/ARCHIVE.md.)
    lab_p, _ = label(pore)
    sizes = np.bincount(lab_p.ravel()); sizes[0] = 0
    f["pore_largest_region_frac"] = (float(sizes.max() / pore.sum())
                                     if pore.sum() else np.nan)

    eps = max(f["pore_frac_resolved"], 1e-3)
    tau_for_b = tau_v
    if np.isfinite(tau_for_b) and tau_for_b > 1.0:
        # tau = eps^(1-b)  ->  b = 1 - ln(tau)/ln(eps); clipped sane range
        f["bruggeman_b_eff"] = float(np.clip(
            1.0 - np.log(tau_for_b) / np.log(eps), 1.0, 3.0))
    else:
        f["bruggeman_b_eff"] = config.BRUGGEMAN_DEFAULT

    f["corr_len_pore_um"] = _two_point_corr_length(pore, um)
    f["corr_len_bulk_um"] = _two_point_corr_length(bulk, um)
    f["corr_len_si_um"] = _two_point_corr_length(si_part, um)

    # ---------------- E. swelling & heterogeneity --------------------------
    f["swelling_budget"] = (f["pore_frac_resolved"]
                            - config.SI_EXPANSION * f["si_frac_total"]
                            - config.GR_EXPANSION * f["gr_frac"])
    f.update(_swelling_map(pore, bulk, si_part, um))

    # ---------------- F. imaging guards -----------------------------------
    f["img_p1"] = float(np.percentile(raw, 1))
    f["img_p50"] = float(np.percentile(raw, 50))
    f["img_p99"] = float(np.percentile(raw, 99))
    i_p = raw[pore].mean() if pore.any() else np.nan
    i_b = raw[bulk].mean() if bulk.any() else np.nan
    i_s = raw[si_part].mean() if si_part.any() else np.nan
    f["si_bulk_contrast"] = ((i_s - i_p) / max(i_b - i_p, 1e-6)
                             if np.isfinite(i_s) else np.nan)
    resid = raw - gaussian_filter(raw, 2)
    bulk_resid = resid[bulk]
    f["noise_mad"] = float(1.4826 * np.median(np.abs(
        bulk_resid - np.median(bulk_resid)))) if bulk.any() else np.nan
    lap = np.abs(np.gradient(np.gradient(raw, axis=0), axis=0)
                 + np.gradient(np.gradient(raw, axis=1), axis=1))
    f["sharpness_lapvar"] = float(lap[bulk].var()) if bulk.any() else np.nan
    row_med = np.median(raw * bulk, axis=1)
    yy = np.arange(len(row_med))
    f["shading_slope"] = float(np.polyfit(yy, row_med, 1)[0]) \
        if len(yy) > 2 else np.nan
    f["has_inlens"] = int("Inlens" in im.channels)
    f["has_etd_or_se"] = int(bool(set(im.channels) & {"ETD", "SE"}))

    # ---------------- uncertainty (tile jackknife) -------------------------
    f["pore_frac_tile_err"] = _tile_jackknife(pore, config.TILE_GRID)
    f["si_frac_tile_err"] = _tile_jackknife(si_part, config.TILE_GRID)
    f["swelling_budget_tile_err"] = float(np.hypot(
        f["pore_frac_tile_err"], config.SI_EXPANSION * f["si_frac_tile_err"]))
    return f
