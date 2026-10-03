"""Phase 4 - DFN simulation of the three batches with PyBaMM.

Model: BasicDFNComposite + Chen2020_composite parameter set (LG M50 cell,
graphite/SiOx composite negative electrode). Everything that is not
image-derived stays at the published value and is identical for all batches.

Image-derived inputs (per image):
    porosity            <- pore_frac (VISIBLE porosity; true porosity is
                           higher because sub-resolution pores are unseen -
                           we therefore run measured AND a literature-range
                           correction porosity_corr = 0.30)
    graphite fraction   <- graphite_frac / solid_frac * (1 - porosity)
    silicon fraction    <- silicon_frac / solid_frac * (1 - porosity)
    graphite radius     <- 0.75 * mean graphite chord (random-section chord
                           = 4R/3 for a sphere; documented assumption)
    silicon radius      <- si_d50_um / 2
    Bruggeman exponent  <- kept at the published 1.5 (not defensible from
                           2D images - stated as an assumption)
    thickness           <- not measurable from images; parameter-set value
                           85.2 um, swept 60/85.2/110 um for Batch_3.

Outputs per batch: discharge curves at C/10, C/2, 1C, 2C; capacity vs rate;
electrolyte and particle lithium concentration profiles at end of
discharge; >=30 bootstrap resamples of the image-derived inputs -> bands.

Plain statement: absolute values are NOT validated; only differences
between batches are meaningful; the standard model includes no swelling,
cracking or ageing.
"""
import os
import pickle
import numpy as np
import pandas as pd
import common as C

N_RESAMPLE = 30
C_RATES = [0.1, 0.5, 1.0, 2.0]
POROSITY_CORR = 0.30          # literature range for calendered anodes
THICKNESS_SWEEP = [60e-6, 85.2e-6, 110e-6]
NQ = 200                      # common capacity-grid points for curve bands


def image_params(F_, batch, exclude_ids=()):
    full = F_[(F_.subset == "full") & (F_.batch == batch) &
              (~F_.image_id.isin(exclude_ids))]
    rows = []
    for _, r in full.iterrows():
        por = float(r["pore_frac"])
        sol = max(float(r["silicon_frac"] + r["graphite_frac"]), 1e-9)
        rows.append(dict(
            image_id=r["image_id"], batch=batch,
            porosity=por,
            gr_amf=float(r["graphite_frac"]) / sol * (1 - por),
            si_amf=float(r["silicon_frac"]) / sol * (1 - por),
            gr_radius_m=0.75 * float(r["gr_chord_h_um"]) * 1e-6,
            si_radius_m=float(r["si_d50_um"]) / 2 * 1e-6))
    return pd.DataFrame(rows)


def set_params(pv, p, porosity):
    pv = pv.copy()
    scale = (1 - porosity) / max(1 - p["porosity"], 1e-9)
    pv.update({
        "Negative electrode porosity": porosity,
        "Primary: Negative electrode active material volume fraction":
            p["gr_amf"] * scale,
        "Secondary: Negative electrode active material volume fraction":
            p["si_amf"] * scale,
        "Primary: Negative particle radius [m]": p["gr_radius_m"],
        "Secondary: Negative particle radius [m]": p["si_radius_m"],
    }, check_already_exists=False)
    return pv


def simulate(pv, crate):
    """One discharge at the given C-rate. Returns dict of arrays or None."""
    import pybamm
    model = pybamm.lithium_ion.BasicDFNComposite()
    sim = pybamm.Simulation(model, parameter_values=pv, C_rate=crate)
    t_end = 3600 / crate * 1.05
    try:
        sol = sim.solve(np.linspace(0, t_end, 300))
    except Exception:
        return None
    t = sol["Time [s]"].entries
    V = sol["Voltage [V]"].entries
    Q = sol["Discharge capacity [A.h]"].entries
    out = dict(t=t, V=V, Q=Q)
    for key, name in (("c_e", "Electrolyte concentration [mol.m-3]"),
                      ("c_s_gr", "Negative primary particle "
                                 "concentration [mol.m-3]"),
                      ("c_s_si", "Negative secondary particle "
                                 "concentration [mol.m-3]")):
        try:
            e = np.asarray(sol[name].entries)
            # spatial profiles: shape (n_space, n_time); keep final column
            if e.ndim == 2 and e.shape[1] == len(t):
                out[key] = e[:, -1]
            elif e.ndim == 2 and e.shape[0] == len(t):
                out[key] = e[-1]
            else:
                out[key] = e.ravel()[-60:]
        except Exception:
            out[key] = None
    return out


def band_on_Q(sols, nq=NQ):
    """V(Q) for all solutions interpolated onto a common Q grid -> bands."""
    ok = [s for s in sols if s is not None and len(s["Q"]) > 5]
    if not ok:
        return None
    qmax = min(s["Q"][-1] for s in ok)
    qg = np.linspace(0, qmax, nq)
    V = np.stack([np.interp(qg, s["Q"], s["V"]) for s in ok])
    return dict(Q=qg, V_md=np.median(V, axis=0),
                V_lo=np.percentile(V, 5, axis=0),
                V_hi=np.percentile(V, 95, axis=0))


def run(F_, kept=None, exclude_ids=()):
    import pybamm
    pv0 = pybamm.ParameterValues("Chen2020_composite")
    rng = np.random.default_rng(0)
    results = {}
    inputs = []
    for batch in C.BATCHES:
        ip = image_params(F_, batch, exclude_ids)
        inputs.append(ip)
        for mode in ("measured", "corrected"):
            for _ in range(N_RESAMPLE):
                p = ip.iloc[rng.integers(0, len(ip))].to_dict()
                por = p["porosity"] if mode == "measured" else POROSITY_CORR
                pv = set_params(pv0, p, porosity=por)
                for rate in C_RATES:
                    s = simulate(pv, rate)
                    results.setdefault((batch, mode, rate), []).append(s)
    IP = pd.concat(inputs)
    IP.to_csv(os.path.join(C.TABLE_DIR, "dfn_inputs.csv"), index=False)

    runs = []
    for (batch, mode, rate), sols in results.items():
        for i, s in enumerate(sols):
            if s is not None:
                runs.append(dict(batch=batch, porosity_mode=mode,
                                 c_rate=rate, rep=i,
                                 capacity_ah=float(s["Q"][-1])))
    pd.DataFrame(runs).to_csv(
        os.path.join(C.TABLE_DIR, "dfn_runs.csv"), index=False)

    rows = []
    for (batch, mode, rate), sols in results.items():
        caps = np.array([s["Q"][-1] for s in sols if s is not None])
        if len(caps) == 0:
            continue
        rows.append(dict(batch=batch, porosity_mode=mode, c_rate=rate,
                         n=len(caps), cap_mean=caps.mean(),
                         cap_std=caps.std(),
                         cap_p5=np.percentile(caps, 5),
                         cap_p95=np.percentile(caps, 95)))
    cap = pd.DataFrame(rows)
    cap.to_csv(os.path.join(C.TABLE_DIR, "dfn_capacity.csv"), index=False)

    # deltas of DFN outputs vs Batch_3 (Phase 5 requirement)
    rr = pd.DataFrame(runs)
    drows = []
    for mode in ("measured", "corrected"):
        for rate in C_RATES:
            y = rr[(rr.batch == C.BASELINE) & (rr.porosity_mode == mode) &
                   (rr.c_rate == rate)].capacity_ah.values
            for b in ("Batch_1", "Batch_2"):
                x = rr[(rr.batch == b) & (rr.porosity_mode == mode) &
                       (rr.c_rate == rate)].capacity_ah.values
                if len(x) < 5 or len(y) < 5:
                    continue
                ci = C.bootstrap_ci_diff(x, y)
                d = x.mean() - y.mean()
                drows.append(dict(batch=b, porosity_mode=mode, c_rate=rate,
                                  delta_ah=d, pct_delta=100*d/y.mean(),
                                  ci_lo=ci[0], ci_hi=ci[1],
                                  sig="CI excludes 0" if ci[0]*ci[1] > 0
                                  else "CI includes 0"))
    pd.DataFrame(drows).to_csv(
        os.path.join(C.TABLE_DIR, "dfn_deltas.csv"), index=False)

    # median end-of-discharge profiles per batch (corrected porosity, 1C)
    prof_curves = {}
    for batch in C.BATCHES:
        ok = [s for s in results.get((batch, "corrected", 1.0), [])
              if s is not None]
        for key in ("c_e", "c_s_gr", "c_s_si"):
            arrs = [np.asarray(s[key]).ravel() for s in ok
                    if s.get(key) is not None]
            if arrs:
                n = min(len(a) for a in arrs)
                A = np.stack([a[:n] for a in arrs])
                prof_curves[(batch, key)] = dict(
                    md=np.median(A, axis=0),
                    lo=np.percentile(A, 5, axis=0),
                    hi=np.percentile(A, 95, axis=0))
    with open(os.path.join(C.TABLE_DIR, "dfn_profile_curves.pkl"),
              "wb") as f:
        pickle.dump(prof_curves, f)

    # assumptions sheet
    assumed = [
        dict(parameter="model", value="pybamm.lithium_ion.BasicDFNComposite",
             note="composite graphite-SiOx negative, full-cell NMC811"),
        dict(parameter="parameter set", value="Chen2020_composite (LGM50)",
             note="published, unchanged where not image-derived"),
        dict(parameter="Negative electrode porosity",
             value="image value  OR corrected to 0.30 (literature range)",
             note="two runs; visible porosity underestimates true"),
        dict(parameter="active material volume fractions",
             value="scaled from image graphite/silicon fractions",
             note="primary=graphite, secondary=SiOx"),
        dict(parameter="particle radii",
             value="graphite: 0.75*chord length; silicon: D50/2",
             note="chord->radius conversion documented"),
        dict(parameter="Negative electrode thickness",
             value="85.2 um (parameter set), swept 60/85.2/110 um",
             note="not visible in top-down SEM"),
        dict(parameter="Bruggeman exponent", value="1.5 (parameter set)",
             note="kept at published value"),
        dict(parameter="everything else",
             value="Chen2020_composite defaults", note="identical for all batches"),
    ]
    pd.DataFrame(assumed).to_csv(
        os.path.join(C.TABLE_DIR, "dfn_assumptions.csv"), index=False)

    curves = {}
    for (batch, mode, rate), sols in results.items():
        b = band_on_Q(sols)
        if b is not None:
            curves[(batch, mode, rate)] = b
    with open(os.path.join(C.TABLE_DIR, "dfn_curves.pkl"), "wb") as f:
        pickle.dump(curves, f)

    prof_rows = []
    for batch in C.BATCHES:
        ok = [s for s in results.get((batch, "corrected", 1.0), [])
              if s is not None and s.get("c_e") is not None]
        for s in ok:
            ce = np.asarray(s["c_e"]).ravel()
            cg = np.asarray(s["c_s_gr"]).ravel() if s.get("c_s_gr") is not None else None
            prof_rows.append(dict(
                batch=batch, c_e_min=ce.min(), c_e_max=ce.max(),
                c_s_gr_mean=cg.mean() if cg is not None else np.nan,
                c_s_gr_min=cg.min() if cg is not None else np.nan,
                c_s_gr_max=cg.max() if cg is not None else np.nan))
    pd.DataFrame(prof_rows).to_csv(
        os.path.join(C.TABLE_DIR, "dfn_profiles.csv"), index=False)

    sweep = []
    b3 = image_params(F_, C.BASELINE, exclude_ids)
    for th in THICKNESS_SWEEP:
        pv = set_params(pv0, b3.iloc[0].to_dict(), POROSITY_CORR)
        pv["Negative electrode thickness [m]"] = th
        for rate in C_RATES:
            s = simulate(pv, rate)
            if s is not None:
                sweep.append(dict(thickness_m=th, c_rate=rate,
                                  capacity_ah=s["Q"][-1]))
    pd.DataFrame(sweep).to_csv(
        os.path.join(C.TABLE_DIR, "dfn_thickness_sweep.csv"), index=False)
    C.log("dfn done")
    return cap


if __name__ == "__main__":
    F_ = pd.read_csv(os.path.join(C.TABLE_DIR, "features.csv"))
    run(F_)
