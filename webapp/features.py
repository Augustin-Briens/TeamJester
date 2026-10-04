"""Phase 2 - per-image features from the 3-phase segmentation.

Every feature is registered in FEATURE_META with a one-sentence definition,
units, the detector used, physical meaning, and a literature reference that
was actually opened (URLs in NOTES.md / feature_meta.csv).

The image is the independent statistical unit. Sizes in um (25 nm/px).
Writes: tables/features.csv (full + L + R rows), particles.csv, pores.csv,
feature_meta.csv, repeatability.csv, dropped_features.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse import linalg as spla
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, binary_dilation, binary_erosion
import common as C

FEATURE_META = {}
def F(name, group, units, det, definition, meaning, ref):
    FEATURE_META[name] = dict(group=group, units=units, detector=det,
                              definition=definition, meaning=meaning, ref=ref)

# ---- composition
F("pore_frac", "composition", "1", "BSE", "Area fraction of the pore/crack class.",
  "Visible pore volume proxy; lower porosity follows heavier calendering.",
  "torquato")
F("graphite_frac", "composition", "1", "BSE", "Area fraction of the grey bulk class.",
  "Assumed-graphite share of the section.", "torquato")
F("silicon_frac", "composition", "1", "BSE", "Area fraction of the bright-particle class.",
  "Assumed-silicon share of the section.", "torquato")
F("si_share_of_solid", "composition", "1", "BSE",
  "silicon_frac / (silicon_frac + graphite_frac).",
  "Si share of the active solid - the composite recipe parameter.", "chen2020")

# ---- silicon particles
F("si_per_mm2", "silicon", "mm^-2", "BSE",
  "Number of watershed-separated Si particles per mm^2.",
  "Particle number density.", "clarkevans")
F("si_d50_um", "silicon", "um", "BSE", "Median number-weighted equivalent diameter.",
  "Typical Si particle size.", "allen")
F("si_d90_um", "silicon", "um", "BSE",
  "90th percentile of number-weighted equivalent diameters.",
  "Coarse end of the Si size distribution.", "allen")
F("si_d99_um", "silicon", "um", "BSE",
  "99th percentile of number-weighted equivalent diameters.",
  "Oversize tail of the Si size distribution.", "allen")
F("si_dmax_um", "silicon", "um", "BSE", "Largest equivalent diameter.",
  "Largest Si particle - crack-initiation risk.", "profatilova")
F("si_d50_area_um", "silicon", "um", "BSE",
  "Median of the area-weighted size distribution.",
  "Size below which half of the Si area sits; weights coarse particles.",
  "allen")
F("si_large_area_frac", "silicon", "1", "BSE",
  "Share of Si area in particles with equivalent diameter > 2 um.",
  "Fraction of Si in coarse particles.", "profatilova")
F("si_solidity", "silicon", "1", "BSE",
  "Area-weighted mean solidity (area / convex-hull area) of particles.",
  "Particle angularity / surface regularity.", "allen")
F("si_aspect", "silicon", "1", "BSE",
  "Area-weighted mean major/minor axis ratio of particles.",
  "Particle elongation.", "allen")
F("si_sv", "silicon", "um^-1", "BSE",
  "Si/solid-matrix boundary length per Si area (perimeter per area).",
  "Interfacial area per volume - governs Si/electrolyte contact.", "underwood")
F("si_cracked_share", "silicon", "1", "BSE",
  "Share of particles whose interior holds dark pixels covering >2% of the "
  "particle area (inside the filled particle outline).",
  "Internally cracked Si fraction.", "profatilova")

# ---- graphite
F("gr_chord_h_um", "graphite", "um", "BSE",
  "Mean horizontal uninterrupted chord through the graphite phase.",
  "Graphite flake lateral extent proxy.", "underwood")
F("gr_chord_v_um", "graphite", "um", "BSE",
  "Mean vertical uninterrupted chord through the graphite phase.",
  "Graphite flake through-plane extent proxy.", "underwood")
F("gr_chord_ratio_hv", "graphite", "1", "BSE", "gr_chord_h_um / gr_chord_v_um.",
  "Flake alignment: >1 means flakes wider than tall (calendering signature).",
  "underwood")
F("gr_st_coherence", "graphite", "1", "BSE",
  "Structure-tensor coherence of the BSE gradient inside graphite "
  "(sigma=5 px).", "Local orientation order of the graphite texture.", "bigun")
F("gr_sv", "graphite", "um^-1", "BSE",
  "Graphite/pore boundary length per graphite area.",
  "Pore-accessible surface of the graphite phase.", "underwood")

# ---- pores
F("pore_thick_d10_um", "pores", "um", "BSE",
  "10th percentile of pore local thickness (2x distance transform).",
  "Fine end of the pore-size distribution.", "hildebrand")
F("pore_thick_d50_um", "pores", "um", "BSE", "Median pore local thickness.",
  "Typical pore size.", "hildebrand")
F("pore_thick_d90_um", "pores", "um", "BSE", "90th percentile pore local thickness.",
  "Coarse end of the pore-size distribution.", "hildebrand")
F("large_gap_frac", "pores", "1", "BSE",
  "Area share of pores occupying >0.5% of the image each.",
  "Large gaps / pull-outs between grains.", "profatilova")
F("crack_frac", "pores", "1", "BSE", "Area share of pores with aspect ratio > 3.",
  "Elongated cracks / slits.", "profatilova")
F("compact_pore_frac", "pores", "1", "BSE",
  "Area share of pores neither large nor elongated.",
  "Compact interstitial pores.", "profatilova")
F("pores_per_mm2", "pores", "mm^-2", "BSE",
  "Number of discrete pore objects per mm^2.", "Pore number density.", "underwood")
F("pore_chord_h_um", "pores", "um", "BSE", "Mean horizontal chord through pores.",
  "Lateral pore width.", "underwood")
F("pore_chord_v_um", "pores", "um", "BSE", "Mean vertical chord through pores.",
  "Through-plane pore height.", "underwood")
F("pore_anisotropy", "pores", "1", "BSE", "pore_chord_h_um / pore_chord_v_um.",
  "Pore shape anisotropy; >1 = laterally elongated (pressing signature).",
  "underwood")
F("pore_depth_cv", "pores", "1", "BSE",
  "CV of pore fraction across 8 depth bands.",
  "Porosity non-uniformity through the electrode thickness.", "profatilova")
F("pore_depth_slope", "pores", "1", "BSE",
  "Least-squares slope of pore fraction vs normalised depth band.",
  "Porosity gradient across the section.", "profatilova")
F("crack_len_density", "pores", "um^-1", "BSE",
  "Skeleton length of crack-class pores per image area.",
  "Crack density per unit area.", "hildebrand")

# ---- silicon-pore relationship
F("si_pore_dist_mean_um", "si_pore", "um", "BSE",
  "Area-weighted mean over particles of the mean distance from particle "
  "pixels to the nearest pore pixel.",
  "How far Si sits from pore space on average.", "underwood")
F("ring_porosity_250nm", "si_pore", "1", "BSE",
  "Pore fraction in the 10 px (~250 nm) ring around all Si particles.",
  "Void space immediately adjacent to Si - expansion room.", "profatilova")
F("ring_porosity_500nm", "si_pore", "1", "BSE",
  "Pore fraction in the 10-20 px (~250-500 nm) ring around all Si particles.",
  "Void space in the near neighbourhood of Si.", "profatilova")
F("si_no_pore_share", "si_pore", "1", "BSE",
  "Share of particles whose nearest pore is more than 0.5 um away.",
  "Si particles fully embedded in graphite.", "profatilova")

# ---- spatial statistics
F("corr_len_pore_um", "spatial", "um", "BSE",
  "Distance where the normalised two-point correlation of the pore phase "
  "drops below 1/e.", "Pore spatial correlation length.", "torquato")
F("corr_len_si_um", "spatial", "um", "BSE", "Same for the silicon phase.",
  "Si spatial correlation length.", "torquato")
F("corr_len_gr_um", "spatial", "um", "BSE", "Same for the graphite phase.",
  "Graphite correlation length.", "torquato")
F("lineal_path_pore_um", "spatial", "um", "BSE",
  "Mean uninterrupted pore run along horizontal lines (lineal-path integral).",
  "Characteristic connected-pore length.", "torquato")
F("si_clarkevans_R", "spatial", "1", "BSE",
  "Observed/expected nearest-centroid distance vs random, Donnelly edge "
  "correction.", "<1 clustered, ~1 random, >1 ordered Si dispersion.",
  "clarkevans")
F("pore_patchiness_cv", "spatial", "1", "BSE",
  "CV of pore fraction across 6 vertical strips.", "Lateral porosity patchiness.",
  "torquato")
F("si_patchiness_cv", "spatial", "1", "BSE",
  "CV of Si fraction across 6 vertical strips.", "Lateral Si patchiness.",
  "torquato")

# ---- texture / surface (Inlens + ETD/SE)
F("inlens_edge_density", "texture", "1", "Inlens",
  "Mean gradient magnitude of the Inlens image (normalised units/px).",
  "Surface fine-crack / edge density.", "bigun")
F("inlens_bulk_texture", "texture", "1", "Inlens",
  "Std of Inlens grey level inside the graphite class.",
  "Surface texture roughness of the bulk.", "bigun")
F("etd_roughness", "texture", "1", "ETD",
  "RMS gradient magnitude of the ETD/SE topography image.",
  "Topographic roughness proxy.", "bigun")
F("bse_bulk_texture", "texture", "1", "BSE",
  "Std of BSE grey level inside the graphite class.",
  "Compositional micro-texture of the graphite.", "bigun")

# ---- transport proxies
F("tau_x", "transport", "1", "BSE",
  "Diffusive tortuosity proxy along x: steady-state Laplace solve on the "
  "8x-downsampled pore mask; tau = phi / (D_eff/D0). NaN when pores do not "
  "percolate in 2D.",
  "PROXY ONLY - a 2D section at ~10% visible porosity cannot give a reliable "
  "3D tortuosity.", "shen")
F("tau_y", "transport", "1", "BSE", "Same along y (through-plane).",
  "PROXY ONLY - see tau_x.", "shen")
F("pore_percolates_x", "transport", "bool", "BSE",
  "Whether the pore phase connects left to right on the 8x-downsampled mask.",
  "2D pore percolation flag for the tortuosity proxy.", "shen")
F("pore_percolates_y", "transport", "bool", "BSE",
  "Whether the pore phase connects top to bottom.",
  "2D pore percolation flag.", "shen")
F("pore_largest_cc_frac", "transport", "1", "BSE",
  "Area of the largest connected pore component / total pore area.",
  "How connected the pore space is inside the 2D section.", "shen")

REFS = {
    "torquato": ("S. Torquato, 'Statistical Description of Microstructures', "
                 "Annu. Rev. Mater. Res. 32:77-111 (2002), DOI "
                 "10.1146/annurev.matsci.32.110101.155324 - abstract opened; "
                 "two-point correlation framework also opened via "
                 "Princeton-hosted Torquato paper (two-point correlation "
                 "functions, basic principles)."),
    "clarkevans": ("Clark & Evans, Ecology 35(4):445-453 (1954), DOI "
                   "10.2307/1931034 - opened ESA/Wiley page (full-access "
                   "article)."),
    "allen": ("Particle-size distribution conventions (equivalent diameter, "
              "percentiles D10/D50/D90) - opened Wikipedia 'Particle-size "
              "distribution'."),
    "underwood": ("Quantitative stereology intercept/chord and "
                  "surface-per-volume methods (Underwood convention) - "
                  "opened stereology.info methods pages."),
    "bigun": ("Structure-tensor orientation/coherence (Bigun et al.) - opened "
              "scikit-image feature.structure_tensor docs and Wikipedia "
              "'Structure tensor'."),
    "hildebrand": ("Hildebrand & Ruegsegger local thickness via distance "
                   "transform, J. Microsc. 185 (1997) - opened BoneJ "
                   "'thickness' documentation describing the method."),
    "profatilova": ("Profatilova et al., ACS Appl. Energy Mater. 3 (2020), "
                    "DOI 10.1021/acsaem.0c01999 - ACS page + free SI at "
                    "figshare 13311418; main text paywalled (noted)."),
    "chen2020": ("Chen et al., J. Electrochem. Soc. 167 080534 (2020), DOI "
                 "10.1149/1945-7111/ab9050 - opened IOP open-access article "
                 "(LGM50 NMC811 / graphite-SiOx parameterisation)."),
    "shen": ("Tortuosity definitions for porous media - opened Wikipedia "
             "'Tortuosity' (diffusive tortuosity = phi/(D_eff/D0), 2D "
             "limitations discussed)."),
}


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
def run_lengths(a):
    """Run lengths of True along the last axis of a 2D bool array."""
    d = np.diff(np.pad(a.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    starts = d == 1
    ends = d == -1
    # pair per row: end index minus start index
    si_ = np.nonzero(starts); ei_ = np.nonzero(ends)
    out = np.empty(len(si_[0]))
    # both indices are row-major sorted already
    out[:] = ei_[1] - si_[1]
    return out


def chord_lengths(mask):
    return run_lengths(mask), run_lengths(mask.T)


def wpercentile(vals, weights, qs):
    """Weighted percentiles (qs in 0-100)."""
    v = np.asarray(vals, float); w = np.asarray(weights, float)
    o = np.argsort(v); v, w = v[o], w[o]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return np.interp(np.asarray(qs) / 100.0, cw, v)


def corr_len(mask, um, max_r_px=300):
    """1/e crossing of the radial two-point correlation (mask downsampled x2,
    result back in original pixels * um)."""
    m = mask[::2, ::2].astype(np.float32)
    if m.std() < 1e-9:
        return np.nan
    m = m - m.mean()
    Ff = np.fft.rfft2(m)
    ac = np.fft.irfft2(Ff * np.conj(Ff), s=m.shape) / m.size
    ac /= ac[0, 0]
    h, w = ac.shape
    yy, xx = np.indices((h, w))
    r = np.hypot(np.minimum(yy, h - yy), np.minimum(xx, w - xx))
    rb = r.astype(int)
    cnt = np.bincount(rb.ravel())
    prof = np.bincount(rb.ravel(), ac.ravel()) / np.maximum(cnt, 1)
    lim = min(max_r_px // 2, len(prof) - 1)
    below = np.nonzero(prof[:lim] < 1 / np.e)[0]
    return float(below[0] * 2 * um) if len(below) else np.nan


def clark_evans(pts, shape):
    """Clark-Evans R with Donnelly (1978) edge correction."""
    n = len(pts)
    if n < 5:
        return np.nan
    h, w = shape
    A = float(h * w)
    d, _ = cKDTree(pts).query(pts, k=2)
    r_obs = d[:, 1].mean()
    r_exp = 0.5 / np.sqrt(n / A)
    r_exp += (0.0514 + 0.0412 / np.sqrt(n)) * (2 * (h + w)) / n
    return float(r_obs / r_exp)


def structure_tensor(g, mask):
    gy, gx = np.gradient(g)
    w = mask.astype(np.float32)
    s = 5.0
    Jxx = ndi.gaussian_filter(w * gx * gx, s)
    Jxy = ndi.gaussian_filter(w * gx * gy, s)
    Jyy = ndi.gaussian_filter(w * gy * gy, s)
    den = np.maximum(Jxx + Jyy, 1e-12)
    coh = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / den
    inside = ndi.uniform_filter(w, s) > 0.5
    ang = np.degrees(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy))
    return float(np.mean(coh[inside])), float(np.median(ang[inside]))


def tortuosity_proxy(pore_mask, ds=8):
    """Steady-state diffusion through the pore phase on a ds-downsampled mask.
    Dirichlet c=1 inlet, c=0 outlet (strong penalty), sealed other faces.
    D_eff_rel = total face flux across the domain / (porous-free flux).
    Returns (tau_x, tau_y, percolates_x, percolates_y)."""
    m = pore_mask[::ds, ::ds]
    phi = float(m.mean())
    if phi < 0.005:
        return np.nan, np.nan, False, False
    h, w = m.shape
    taus, percs = [], []
    for axis in (1, 0):
        inb = (slice(None), 0) if axis == 1 else (0, slice(None))
        outb = (slice(None), w - 1) if axis == 1 else (h - 1, slice(None))
        # connectivity across flow axis?
        ids = -np.ones(m.shape, np.int64)
        ids[m] = np.arange(m.sum())
        n = int(m.sum())
        # build Laplacian on pore pixels
        rows, cols, vals = [], [], []
        yy, xx = np.nonzero(m)
        bc_idx = np.zeros(m.shape, bool)
        bc_idx[inb] = True; bc_idx[outb] = True
        bc_idx &= m
        rhs = np.zeros(n)
        pen = 1e6
        for y, x in zip(yy, xx):
            i = ids[y, x]
            if bc_idx[y, x]:
                rows.append(i); cols.append(i); vals.append(pen)
                rhs[i] = pen * (1.0 if (x == 0 if axis == 1 else y == 0) else 0.0)
                continue
            diag = 0.0
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy2, xx2 = y + dy, x + dx
                if 0 <= yy2 < h and 0 <= xx2 < w and m[yy2, xx2]:
                    if bc_idx[yy2, xx2]:
                        # move to rhs: contribution of fixed neighbour
                        rhs[i] += (1.0 if (xx2 == 0 if axis == 1 else yy2 == 0) else 0.0)
                    else:
                        rows.append(i); cols.append(ids[yy2, xx2]); vals.append(-1.0)
                diag += 1.0
            rows.append(i); cols.append(i); vals.append(diag)
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        try:
            c = spla.spsolve(A, rhs)
        except Exception:
            taus.append(np.nan); percs.append(False); continue
        conc = np.zeros(m.shape); conc[m] = c
        # total flux crossing each face adjacent to the outlet boundary
        if axis == 1:
            faces = m[:, w - 1] & m[:, w - 2]
            q = float(np.abs(conc[:, w - 1] - conc[:, w - 2])[faces].sum())
            q0 = float(np.nonzero(m[:, w - 1])[0].size) / (w - 1)
            percol = bool((m[:, 0] & (ndi.label(m)[0][:, -1] > 0)).any()
                          if m[:, 0].any() and m[:, -1].any() else False)
        else:
            faces = m[h - 1, :] & m[h - 2, :]
            q = float(np.abs(conc[h - 1, :] - conc[h - 2, :])[faces].sum())
            q0 = float(np.nonzero(m[h - 1, :])[0].size) / (h - 1)
            percol = False
        deff_rel = q / max(q0 * (w - 1 if axis == 1 else h - 1), 1e-9)
        taus.append(phi / deff_rel if deff_rel > 1e-12 else np.inf)
        percs.append(bool(q > 1e-9))
    return taus[0], taus[1], percs[0], percs[1]


def edge_density(g):
    gy, gx = np.gradient(g)
    return float(np.hypot(gy, gx).mean())


# --------------------------------------------------------------------------
# main per-image compute
# --------------------------------------------------------------------------
def compute(rec, subset="full"):
    z = C.load_masks(rec["batch"], rec["image_id"])
    um, _ = C.pixel_size_um(rec["bse"])
    if subset == "full":
        pore, si, lab = z["pore"], z["silicon"], z["labels"]
    else:
        pore, si, lab = (z[f"pore_{subset}"], z[f"silicon_{subset}"],
                         z[f"labels_{subset}"])
    h, w = pore.shape
    npx = h * w
    area_mm2 = npx * (um / 1000.0) ** 2
    fe = {}

    pf, sf = float(pore.mean()), float(si.mean())
    gf = 1 - pf - sf
    fe["pore_frac"], fe["graphite_frac"], fe["silicon_frac"] = pf, gf, sf
    fe["si_share_of_solid"] = sf / max(sf + gf, 1e-9)

    # silicon particles ---------------------------------------------------
    d2pore = ndi.distance_transform_edt(~pore)
    parts = []
    filled_si = ndi.binary_fill_holes(si)
    for p in regionprops(lab):
        r0, c0, r1, c1 = p.bbox
        reg = lab[r0:r1, c0:c1] == p.label
        dd = d2pore[r0:r1, c0:c1][reg]
        parts.append(dict(
            label=p.label, area_px=float(p.area),
            diam_um=float(p.equivalent_diameter_area) * um,
            major_um=float(p.axis_major_length) * um,
            minor_um=float(p.axis_minor_length) * um,
            solidity=float(p.solidity),
            cy=float(p.centroid[0]), cx=float(p.centroid[1]),
            border=bool(r0 == 0 or c0 == 0 or r1 == h or c1 == w),
            d_mean_px=float(dd.mean()), d_min_px=float(dd.min()),
            dark_px=float((pore[r0:r1, c0:c1] & reg).sum()),
        ))
    P = pd.DataFrame(parts)
    if len(P):
        P["dark_frac"] = P["dark_px"] / P["area_px"]
        fe["si_per_mm2"] = len(P) / area_mm2
        fe["si_d50_um"], fe["si_d90_um"], fe["si_d99_um"] = \
            np.percentile(P["diam_um"], [50, 90, 99])
        fe["si_dmax_um"] = P["diam_um"].max()
        fe["si_d50_area_um"] = float(wpercentile(P["diam_um"], P["area_px"], 50))
        big = P["diam_um"] > 2.0
        fe["si_large_area_frac"] = float(P.loc[big, "area_px"].sum()
                                         / max(P["area_px"].sum(), 1))
        fe["si_solidity"] = float(np.average(P["solidity"], weights=P["area_px"]))
        P["aspect"] = P["major_um"] / P["minor_um"].clip(1e-9)
        fe["si_aspect"] = float(wpercentile(P["aspect"], P["area_px"], 50))
        si_boundary = si & ~ndi.binary_erosion(si)
        fe["si_sv"] = float(si_boundary.sum() * um /
                            max(P["area_px"].sum() * um ** 2, 1e-9))
        fe["si_cracked_share"] = float((P["dark_frac"] > 0.02).mean())
        fe["si_pore_dist_mean_um"] = float(
            np.average(P["d_mean_px"], weights=P["area_px"])) * um
        fe["si_no_pore_share"] = float((P["d_min_px"] * um > 0.5).mean())
        fe["si_clarkevans_R"] = clark_evans(P[["cy", "cx"]].values, (h, w))
    else:
        for k in ("si_per_mm2", "si_d50_um", "si_d90_um",
                  "si_d99_um", "si_dmax_um", "si_d50_area_um",
                  "si_large_area_frac", "si_solidity", "si_aspect", "si_sv",
                  "si_cracked_share", "si_pore_dist_mean_um",
                  "si_no_pore_share", "si_clarkevans_R"):
            fe[k] = np.nan

    d_out = ndi.distance_transform_edt(~si)
    ring1 = (d_out > 0) & (d_out <= 10)
    ring2 = (d_out > 10) & (d_out <= 20)
    fe["ring_porosity_250nm"] = float(pore[ring1].mean()) if ring1.any() else np.nan
    fe["ring_porosity_500nm"] = float(pore[ring2].mean()) if ring2.any() else np.nan

    # graphite ------------------------------------------------------------
    gr = ~(pore | si)
    lh, lv = chord_lengths(gr)
    fe["gr_chord_h_um"] = float(lh.mean()) * um if len(lh) else np.nan
    fe["gr_chord_v_um"] = float(lv.mean()) * um if len(lv) else np.nan
    fe["gr_chord_ratio_hv"] = fe["gr_chord_h_um"] / max(fe["gr_chord_v_um"], 1e-9)
    # graphite-pore boundary = graphite pixels touching a pore pixel (4-conn)
    gr_bnd = gr & ndi.binary_dilation(pore)
    fe["gr_sv"] = float(gr_bnd.sum() * um / max(gr.sum() * um ** 2, 1e-9))

    # pores ----------------------------------------------------------------
    lab_p = label(pore)
    dist_p = ndi.distance_transform_edt(pore)
    thick = 2.0 * dist_p[pore] * um
    if len(thick):
        fe["pore_thick_d10_um"], fe["pore_thick_d50_um"], \
            fe["pore_thick_d90_um"] = np.percentile(thick, [10, 50, 90])
    else:
        fe["pore_thick_d10_um"] = fe["pore_thick_d50_um"] = \
            fe["pore_thick_d90_um"] = np.nan
    prow = []
    crack_mask = np.zeros_like(pore)
    for p in regionprops(lab_p):
        area_frac = p.area / npx
        aspect = p.axis_major_length / max(p.axis_minor_length, 1e-9)
        cls = ("large_gap" if area_frac > 0.005
               else ("crack" if aspect > 3 else "compact"))
        prow.append(dict(label=p.label, area_px=float(p.area),
                         area_frac=area_frac, aspect=float(aspect), cls=cls,
                         cy=float(p.centroid[0]),
                         thick_mean_um=float(2 * dist_p[p.slice][
                             lab_p[p.slice] == p.label].mean()) * um,
                         major_um=float(p.axis_major_length) * um))
        if cls == "crack":
            crack_mask[lab_p == p.label] = True
    Pf = pd.DataFrame(prow)
    if len(Pf):
        fe["large_gap_frac"] = float(Pf.loc[Pf.cls == "large_gap", "area_frac"].sum())
        fe["crack_frac"] = float(Pf.loc[Pf.cls == "crack", "area_frac"].sum())
        fe["compact_pore_frac"] = float(Pf.loc[Pf.cls == "compact", "area_frac"].sum())
        fe["pores_per_mm2"] = len(Pf) / area_mm2
    else:
        fe.update(large_gap_frac=0.0, crack_frac=0.0, compact_pore_frac=0.0,
                  pores_per_mm2=0.0)
    ph, pv = chord_lengths(pore)
    fe["pore_chord_h_um"] = float(ph.mean()) * um if len(ph) else np.nan
    fe["pore_chord_v_um"] = float(pv.mean()) * um if len(pv) else np.nan
    fe["pore_anisotropy"] = fe["pore_chord_h_um"] / max(fe["pore_chord_v_um"], 1e-9)
    bands = np.array([pore[i * h // 8:(i + 1) * h // 8].mean() for i in range(8)])
    fe["pore_depth_cv"] = float(bands.std() / max(bands.mean(), 1e-9))
    fe["pore_depth_slope"] = float(np.polyfit(np.linspace(0, 1, 8), bands, 1)[0])
    fe["crack_len_density"] = float(skeletonize(crack_mask).sum() * um /
                                    (npx * um ** 2)) if crack_mask.any() else 0.0
    fe["lineal_path_pore_um"] = fe["pore_chord_h_um"]

    # spatial ---------------------------------------------------------------
    fe["corr_len_pore_um"] = corr_len(pore, um)
    fe["corr_len_si_um"] = corr_len(si, um)
    fe["corr_len_gr_um"] = corr_len(gr, um)
    strips_p = [pore[:, i * w // 6:(i + 1) * w // 6].mean() for i in range(6)]
    strips_s = [si[:, i * w // 6:(i + 1) * w // 6].mean() for i in range(6)]
    fe["pore_patchiness_cv"] = float(np.std(strips_p) / max(np.mean(strips_p), 1e-9))
    fe["si_patchiness_cv"] = float(np.std(strips_s) / max(np.mean(strips_s), 1e-9))

    # texture + transport (needs detector files; computed on the matching
    # slice for L/R subsets as well) ---------------------------------------
    g_full, _ = C.load_gray(rec["bse"])
    if subset == "L":
        g = g_full[:, :g_full.shape[1] // 2]
    elif subset == "R":
        g = g_full[:, g_full.shape[1] // 2:]
    else:
        g = g_full
    g = g[:h, :w]
    fe["bse_bulk_texture"] = float(g[gr].std())
    if rec["inlens"] and os.path.exists(rec["inlens"]):
        il_full, _ = C.load_gray(rec["inlens"])
        il = (il_full[:, :il_full.shape[1] // 2] if subset == "L" else
              il_full[:, il_full.shape[1] // 2:] if subset == "R" else il_full)
        il = il[:h, :w]
        fe["inlens_edge_density"] = edge_density(il / 255.0)
        fe["inlens_bulk_texture"] = float(il[gr].std())
    else:
        fe["inlens_edge_density"] = np.nan
        fe["inlens_bulk_texture"] = np.nan
    if rec["topo"] and os.path.exists(rec["topo"]):
        tp_full, _ = C.load_gray(rec["topo"])
        tp = (tp_full[:, :tp_full.shape[1] // 2] if subset == "L" else
              tp_full[:, tp_full.shape[1] // 2:] if subset == "R" else tp_full)
        tp = tp[:h, :w]
        fe["etd_roughness"] = edge_density(tp / 255.0)
    else:
        fe["etd_roughness"] = np.nan
    coh, ang = structure_tensor(g, gr)
    fe["gr_st_coherence"], fe["gr_st_orient_deg"] = coh, ang
    fe["tau_x"], fe["tau_y"], fe["pore_percolates_x"], \
        fe["pore_percolates_y"] = tortuosity_proxy(pore)
    cc = ndi.label(pore)[0]
    if cc.max() > 0:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        fe["pore_largest_cc_frac"] = float(sizes.max() / max(pore.sum(), 1))
    else:
        fe["pore_largest_cc_frac"] = 0.0
    fe.update(extra_features(
        pore, si, um, area_mm2, g=g,
        il=il if (rec["inlens"] and os.path.exists(rec["inlens"])) else None,
        tp=tp if (rec["topo"] and os.path.exists(rec["topo"])) else None))
    return fe, P, Pf


# ---- second-round additions (B2-vs-B3 discrimination)
F("elong_pore_area_frac", "pores", "1", "BSE",
  "Share of pore area in components with aspect>4 and minor axis <5 um.",
  "Elongated crack-like pores — tears from drying/handling stress.",
  "underwood")
F("elong_pore_n_mm2", "pores", "mm^-2", "BSE",
  "Count of elongated pores (aspect>4, width<5 um) per mm^2.",
  "Crack/tear number density.", "underwood")
F("elong_pore_hfrac", "pores", "1", "BSE",
  "Share of elongated pores oriented more horizontally than vertically.",
  "Crack alignment with the coating plane.", "underwood")
F("pore_aspect_p90", "pores", "1", "BSE",
  "90th percentile aspect ratio of pore components >1 um^2.",
  "How stretched the extreme pores are.", "underwood")
F("pore_small_n_mm2", "pores", "mm^-2", "BSE",
  "Count of pores with equivalent diameter 0.5-5 um per mm^2.",
  "Compact-pore number density.", "hildebrand")
F("pore_perim_mm", "pores", "mm/mm^2", "BSE",
  "Total pore perimeter per image area.",
  "Pore-solid boundary density.", "underwood")
F("si_border_pore", "si-pore", "1", "BSE",
  "Share of silicon-boundary pixels within 3 px of a pore.",
  "How pore-associated the silicon particles are.", "underwood")
F("bse_grad_coh", "texture", "1", "BSE",
  "Gradient-orientation coherence (structure tensor) of the BSE image.",
  "Directionality of the bulk texture.", "bigun")
F("inl_lbp_ent", "texture", "1", "Inlens",
  "Entropy of the uniform-LBP(8,1) histogram.",
  "Surface texture complexity.", "ojala")
F("inl_lbp_flat", "texture", "1", "Inlens",
  "Share of pixels in the flat LBP bin 0.",
  "Surface smoothness: higher means flatter.", "ojala")
F("inl_grad_coh", "texture", "1", "Inlens",
  "Gradient-orientation coherence of the Inlens image.",
  "Directionality/streakiness of the surface texture.", "bigun")
F("etd_fft_hi", "texture", "1", "ETD",
  "Radial-FFT energy fraction above 0.15 cycles/px (4x subsample).",
  "Fine-scale topographic texture energy.", "fourier")


# ---------------------------------------------------------------------------
# Second-round features (added for B2-vs-B3 discrimination; gate results in
# outputs/tables/exp_newfeat_gates.csv — all below passed lr_corr >= 0.5 and
# |corr with noise_mad| < 0.85)
# ---------------------------------------------------------------------------
def _grad_coh(g):
    """Structure-tensor coherence of gradient orientation (0=isotropic)."""
    gx = ndi.sobel(g.astype(np.float32), axis=1)
    gy = ndi.sobel(g.astype(np.float32), axis=0)
    mag = np.hypot(gx, gy)
    keep = mag > np.percentile(mag, 75)
    Jxx = float((gx[keep] ** 2).mean()); Jyy = float((gy[keep] ** 2).mean())
    Jxy = float((gx[keep] * gy[keep]).mean())
    lam1 = (Jxx + Jyy) / 2 + np.hypot((Jxx - Jyy) / 2, Jxy)
    lam2 = (Jxx + Jyy) / 2 - np.hypot((Jxx - Jyy) / 2, Jxy)
    return float((lam1 - lam2) / max(lam1 + lam2, 1e-12))


def _lbp(g):
    """Uniform-LBP(8,1) histogram entropy + share of flat points."""
    from skimage.feature import local_binary_pattern
    g2 = g[::2, ::2] if max(g.shape) > 4000 else g
    lbp = local_binary_pattern(g2, 8, 1, method="uniform")
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10)
    p = hist / hist.sum()
    return float(-(p[p > 0] * np.log(p[p > 0])).sum()), float(p[0])


def _fft_hi(g):
    """Fraction of radial-FFT energy above 0.15 cycles/px (4x subsample)."""
    g2 = g[::4, ::4] - g[::4, ::4].mean()
    g2 = g2 * np.outer(np.hanning(g2.shape[0]), np.hanning(g2.shape[1]))
    Fp = np.abs(np.fft.fftshift(np.fft.fft2(g2))) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(g2.shape[0]),
                         np.fft.fftfreq(g2.shape[1]), indexing="ij")
    fr = np.hypot(fx, fy)
    return float(Fp[fr > 0.15].sum() / max(Fp[fr > 0.02].sum(), 1e-12))


def extra_features(pore, si, um, area_mm2, g=None, il=None, tp=None):
    fe = {}
    props = regionprops(ndi.label(pore)[0]) if pore.any() else []
    if props:
        maj = np.array([p.axis_major_length * um for p in props])
        mnr = np.array([p.axis_minor_length * um for p in props])
        ar = np.array([p.area * um * um for p in props])
        aspect = maj / np.maximum(mnr, 1e-9)
        big = ar > 1.0
        elong = big & (aspect > 4) & (mnr < 5.0)
        fe["elong_pore_area_frac"] = float(ar[elong].sum() /
                                           max(ar.sum(), 1e-9))
        fe["elong_pore_n_mm2"] = float(elong.sum() / area_mm2)
        ors = np.array([abs(p.orientation) for p in props])
        fe["elong_pore_hfrac"] = float((ors[elong] > np.pi / 4).mean()) \
            if elong.any() else np.nan
        fe["pore_aspect_p90"] = float(np.percentile(aspect[big], 90)) \
            if big.any() else np.nan
        diam = np.array([p.equivalent_diameter_area * um for p in props])
        fe["pore_small_n_mm2"] = float((big & (diam >= 0.5) &
                                      (diam <= 5.0)).sum() / area_mm2)
        fe["pore_perim_mm"] = float(sum(p.perimeter * um for p in props) /
                                    1000.0 / area_mm2)
    if si.any():
        border = si & ~binary_erosion(si)
        d2p = ndi.distance_transform_edt(~pore)
        fe["si_border_pore"] = float((d2p <= 3)[border].mean())
    else:
        fe["si_border_pore"] = np.nan
    if g is not None:
        fe["bse_grad_coh"] = _grad_coh(g)
    if il is not None:
        fe["inl_lbp_ent"], fe["inl_lbp_flat"] = _lbp(il)
        fe["inl_grad_coh"] = _grad_coh(il)
    if tp is not None:
        fe["etd_fft_hi"] = _fft_hi(tp)
    return fe


def one(rec):
    try:
        rows = []
        fe, parts, pores = compute(rec, "full")
        fe.update(batch=rec["batch"], image_id=rec["image_id"], subset="full")
        rows.append(fe)
        for sub in ("L", "R"):
            fh, _, _ = compute(rec, sub)
            fh.update(batch=rec["batch"], image_id=rec["image_id"], subset=sub)
            rows.append(fh)
        for df in (parts, pores):
            if len(df):
                df["batch"] = rec["batch"]; df["image_id"] = rec["image_id"]
        return pd.DataFrame(rows), parts, pores
    except Exception as e:
        import traceback; traceback.print_exc()
        return pd.DataFrame([dict(batch=rec["batch"], image_id=rec["image_id"],
                                  subset="full", error=str(e))]), \
            pd.DataFrame(), pd.DataFrame()


def run(locs=None, nproc=6):
    from multiprocessing import Pool
    locs = locs or C.find_locations()
    feat_csv = os.path.join(C.TABLE_DIR, "features.csv")
    if os.path.exists(feat_csv):
        C.log("features.csv exists - loading")
        return pd.read_csv(feat_csv)
    with Pool(nproc) as pool:
        res = pool.map(one, locs)
    F_ = pd.concat([r[0] for r in res], ignore_index=True)
    P_ = pd.concat([r[1] for r in res if len(r[1])], ignore_index=True) \
        if any(len(r[1]) for r in res) else pd.DataFrame()
    R_ = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True) \
        if any(len(r[2]) for r in res) else pd.DataFrame()
    F_.to_csv(feat_csv, index=False)
    P_.to_csv(os.path.join(C.TABLE_DIR, "particles.csv"), index=False)
    R_.to_csv(os.path.join(C.TABLE_DIR, "pores.csv"), index=False)
    pd.DataFrame([dict(feature=k, **v) for k, v in FEATURE_META.items()]
                 ).to_csv(os.path.join(C.TABLE_DIR, "feature_meta.csv"), index=False)
    C.log(f"features: {len(F_)} rows, {len(P_)} particles, {len(R_)} pores")
    return F_


def repeatability(F_):
    wide = F_.pivot_table(index=["batch", "image_id"], columns="subset",
                          values=[c for c in F_.columns
                                  if c not in ("batch", "image_id", "subset", "error")])
    feats = [c for c in FEATURE_META if c in wide.columns.get_level_values(0)]
    feats += ["gr_st_orient_deg", "pore_percolates_x", "pore_percolates_y"]
    feats = [f for f in feats if f in wide.columns.get_level_values(0)]
    rows = []
    for f in feats:
        sub = wide[f]
        if "L" not in sub.columns or "R" not in sub.columns:
            rows.append(dict(feature=f, lr_corr=np.nan, lr_mad=np.nan,
                             rel_mad=np.nan))
            continue
        l, r = sub["L"], sub["R"]
        ok = ~(l.isna() | r.isna())
        c = float(l[ok].corr(r[ok])) if ok.sum() > 3 else np.nan
        mad = float((l[ok] - r[ok]).abs().mean())
        m = float(F_[F_.subset == "full"][f].abs().mean())
        rows.append(dict(feature=f, lr_corr=c, lr_mad=mad,
                         rel_mad=mad / max(m, 1e-12)))
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(C.TABLE_DIR, "repeatability.csv"), index=False)
    return rep


def prune(F_, rep):
    full = F_[F_.subset == "full"].set_index(["batch", "image_id"])
    feats = [c for c in FEATURE_META if c in full.columns]
    kept, dropped = [], []
    rc_map = rep.set_index("feature")["lr_corr"]
    for f in feats:
        v = full[f]
        if v.isna().mean() > 0.5 or v.std() < 1e-12:
            dropped.append(dict(feature=f, reason="constant or >50% missing"))
            continue
        rc = rc_map.get(f, np.nan)
        if pd.isna(rc) or rc < 0.5:
            rs = "nan (unrepeatable: one/both halves constant)" if pd.isna(rc) \
                else f"{rc:.2f} < 0.5"
            dropped.append(dict(feature=f, reason=f"left/right corr {rs}"))
            continue
        dup = None
        for k in kept:
            kk = full[[f, k]].dropna()
            if len(kk) > 5 and abs(kk[f].corr(kk[k])) > 0.9:
                dup = k; break
        if dup:
            dropped.append(dict(feature=f, reason=f"duplicate of {dup} (|r|>0.9)"))
            continue
        kept.append(f)
    pd.DataFrame(dropped).to_csv(
        os.path.join(C.TABLE_DIR, "dropped_features.csv"), index=False)
    C.log(f"kept {len(kept)}, dropped {len(dropped)}: "
          f"{[d['feature'] for d in dropped]}")
    return kept, dropped


if __name__ == "__main__":
    F_ = run()
    rep = repeatability(F_)
    kept, dropped = prune(F_, rep)
