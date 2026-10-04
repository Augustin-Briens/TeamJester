"""Turn batch-level image markers into PyBaMM parameter override dicts.

Marker values used directly: porosity, Si volume fraction, Bruggeman
exponent, Si mean radius, surface area. Assumed (NOT measured): electrode
thickness, sub-resolution porosity, and which fraction of the silicon is
electrochemically active.  Silicon content enters as an explicit
BRACKET: the low end counts only classified Si objects times an
'accessible' share; the high end counts ALL uncertain bright material
as active silicon.  Each batch runs 1 band-centre + N Latin-hypercube
points inside the bands, so the DFN output is a per-batch band, not a
single number.  Si Young's modulus and Poisson's ratio are swept over
their cited ranges as extra assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# batch marker bands (from the extracted markers table)
# ---------------------------------------------------------------------------
def batch_bands(markers: pd.DataFrame,
                accessible: float | None = None,
                exclude_problem: bool = False) -> pd.DataFrame:
    """Batch-level min/median/max for every DFN input.

    si_act bracket:
      lo = classified Si share of the solid x accessible share
      hi = ALL bright material (raw bright fraction) treated as active Si
    accessible=None -> measured per-image si_accessible_frac;
    accessible=1.0  -> 'all silicon active' variant (assumption, swept)
    """
    f = markers
    if exclude_problem and "is_problem_photo" in markers:
        keep = markers[markers["is_problem_photo"] == 0]
        if len(keep):
            f = keep
    pore = f["pore_frac_resolved"].median()
    subpore_lo, subpore_hi = 0.15, 0.30      # assumed, swept
    # silicon brackets per image, then batch bands
    si_class_solid = (f["si_frac_total"] / (1 - f["pore_frac_resolved"])
                      ).clip(0, 1)
    si_raw_solid = (f["bright_frac_raw"] / (1 - f["pore_frac_resolved"])
                    ).clip(0, 1)
    acc = f["si_accessible_frac"].median() if accessible is None \
        else accessible
    si_act_lo = si_class_solid.median() * acc
    si_act_med = si_class_solid.median() * acc
    si_act_hi = si_raw_solid.median()        # all uncertain bright = Si
    eps_hi = float(np.quantile(f["pore_frac_resolved"], 0.85)) + subpore_hi
    rows = [
        ("porosity", f["pore_frac_resolved"].min(), pore,
         f["pore_frac_resolved"].max(), "image", "add sub-res sweep"),
        ("porosity_total_lo", 0.0, pore + subpore_lo, 0.0, "assumed",
         "sub-resolution porosity added to measured"),
        ("porosity_total_hi", 0.0, min(eps_hi, 0.9), 0.0, "assumed", ""),
        ("si_act_lo", 0.0, float(si_act_lo), 0.0, "image+assumed",
         "classified Si objects x accessible share"),
        ("si_act_med", 0.0, float(si_act_med), 0.0, "image+assumed", ""),
        ("si_act_hi", 0.0, float(si_act_hi), 0.0, "image bracket",
         "all bright material counted as active Si"),
        ("si_radius_um", f["si_d50_aw_um"].min(),
         f["si_d50_aw_um"].median(), f["si_d50_aw_um"].max(), "image", ""),
        ("si_sv", f["si_sv_um_inv"].min(), f["si_sv_um_inv"].median(),
         f["si_sv_um_inv"].max(), "image", ""),
        ("gr_radius_um", f["gr_radius_eff_um"].min(),
         f["gr_radius_eff_um"].median(), f["gr_radius_eff_um"].max(),
         "image", "graphite diffusion-equivalent radius 3V/S"),
        ("bruggeman", f["bruggeman_b_eff"].min(),
         f["bruggeman_b_eff"].median(), f["bruggeman_b_eff"].max(),
         "image proxy", "finite-difference tortuosity -> b"),
        ("thickness_um", 65.0, 75.0, 85.0, "assumed",
         "electrode extends beyond image FOV"),
    ]
    df = pd.DataFrame(rows, columns=["param", "lo", "med", "hi", "kind",
                                     "note"])
    df = df.set_index("param")
    return df


# ---------------------------------------------------------------------------
# latin-hypercube sweep points per batch
# ---------------------------------------------------------------------------
SWEEP_PARAMS = ("porosity_total", "si_active_frac", "thickness_m",
                "bruggeman", "si_radius_m", "si_youngs_pa", "si_nu")

# cited ranges for silicon mechanics (lithiated Si); swept as assumptions
SI_YOUNGS_RANGE = (35e9, 50e9, 90e9)      # Bonkile2024 default = 50 GPa
SI_NU_RANGE = (0.22, 0.25, 0.28)


def sweep_points(bands: pd.DataFrame, n_points: int = 8,
                 seed: int = 0) -> pd.DataFrame:
    """Band-centre point + n LHS samples inside the bands."""
    from scipy.stats import qmc
    g = lambda p: (bands.at[p, "lo"], bands.at[p, "med"],
                   bands.at[p, "hi"])
    box = {"porosity_total": (g("porosity_total_lo")[1],
                              g("porosity_total_hi")[1]),
           "si_active_frac": (max(g("si_act_lo")[1], 0.02),
                              min(g("si_act_hi")[1], 0.35)),
           "thickness_m": tuple(v * 1e-6 for v in g("thickness_um")[::2]),
           "bruggeman": g("bruggeman")[::2],
           "si_radius_m": (g("si_radius_um")[0] / 2 * 1e-6,
                            g("si_radius_um")[2] / 2 * 1e-6),
           "gr_radius_m": (g("gr_radius_um")[0] * 1e-6,
                            g("gr_radius_um")[2] * 1e-6),
           "si_youngs_pa": (SI_YOUNGS_RANGE[0], SI_YOUNGS_RANGE[2]),
           "si_nu": (SI_NU_RANGE[0], SI_NU_RANGE[2])}
    centre = {"porosity_total": bands.at["porosity", "med"] + 0.20,
              "si_active_frac": bands.at["si_act_med", "med"],
              "thickness_m": bands.at["thickness_um", "med"] * 1e-6,
              "bruggeman": bands.at["bruggeman", "med"],
              "si_radius_m": bands.at["si_radius_um", "med"] / 2 * 1e-6,
              "gr_radius_m": bands.at["gr_radius_um", "med"] * 1e-6,
              "si_youngs_pa": SI_YOUNGS_RANGE[1],
              "si_nu": SI_NU_RANGE[1]}
    cols = list(box)
    pts = [centre]
    if n_points > 0:
        lhs = qmc.LatinHypercube(len(box), seed=seed, strength=1)
        u = lhs.random(n_points)
        for row in qmc.scale(u, [box[c][0] for c in cols],
                             [box[c][1] for c in cols]):
            pts.append(dict(zip(cols, row)))
    return pd.DataFrame(pts)


def to_pybamm_params(point: dict) -> dict:
    """Sweep point -> PyBaMM parameter overrides."""
    eps = point["porosity_total"]
    si = point["si_active_frac"]
    solid = 1.0 - eps
    return {
        "Negative electrode thickness [m]": point["thickness_m"],
        "Negative electrode porosity": eps,
        "Negative electrode Bruggeman coefficient (electrolyte)":
            point["bruggeman"],
        "Primary: Negative electrode active material volume fraction":
            solid * (1.0 - si),
        "Secondary: Negative electrode active material volume fraction":
            solid * si,
        "Primary: Negative electrode particle radius [m]":
            point["gr_radius_m"],
        "Secondary: Negative electrode particle radius [m]":
            point["si_radius_m"],
        "Secondary: Negative electrode Young's modulus [Pa]":
            point["si_youngs_pa"],
        "Secondary: Negative electrode Poisson's ratio":
            point["si_nu"],
    }


def run_assumption_table(bands: pd.DataFrame,
                         points: pd.DataFrame) -> pd.DataFrame:
    """Plain-language list of assumed (not measured) inputs."""
    rows = [
        ("electrode thickness", "75 um [65-85 swept]",
         "foil & coating top not in frame"),
        ("sub-resolution porosity", "+0.20 [+0.15 to +0.30]",
         "pores below ~0.35 um are invisible"),
        ("Si active share",
         f"{bands.at['si_act_med', 'med']:.2f} "
         f"[{bands.at['si_act_lo', 'med']:.2f}-"
         f"{bands.at['si_act_hi', 'med']:.2f}]",
         "lo: classified Si x share touching pore; "
         "hi: ALL uncertain bright counted (label bracket); "
         "'all-Si-active' variant sets the share to 1.0"),
        ("Si Young's modulus", "50 GPa [35-90 swept]",
         "Bonkile2024; lithiated-Si literature range"),
        ("Si Poisson's ratio", "0.25 [0.22-0.28 swept]",
         "Bonkile2024; cited range"),
        ("Si particle radius",
         f"{bands.at['si_radius_um', 'med'] / 2:.2f} um",
         "d50/2 of detected Si (area-weighted)"),
        ("graphite particle radius",
         "5.86 um FIXED, identical for every batch",
         "fixed assumed value (Chen2020) — NOT measured from images"),
        ("pore count / pore density",
         "not a DFN input",
         "resolved-pore-count differences (e.g. Batch_1 vs Batch_3) "
         "are image findings only — the homogenised model has no "
         "parameter they map to, so the DFN verdict does not cover them"),
        ("primary-phase chemistry", "graphite (NMC622 cathode)",
         "Chen2020_composite baseline cell"),
        ("Bruggeman exponent",
         f"{bands.at['bruggeman', 'med']:.2f}",
         "2-D finite-difference tortuosity proxy"),
        ("degradation constants",
         "OKane2022 (LAM prop. 2.7778e-7 s-1, SEI kinetic 1e-12 m/s)",
         "published; see dfn_output/dfn_validation.md constants table"),
    ]
    return pd.DataFrame(rows, columns=["input", "value", "why assumed"])
