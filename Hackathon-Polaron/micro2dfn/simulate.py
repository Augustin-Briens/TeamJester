"""PyBaMM DFN simulation of the composite Si-graphite anode (repaired).

Repaired relative to the first version (see dfn_output/bug_reproduction.txt):

  * Parameters: Chen2020_composite base, UNCHANGED. Missing
    mechanics/LAM/SEI keys are added from published donors only
    (Bonkile2024 for the per-phase composite block, OKane2022 for shared
    SEI constants) and only when absent — silicon keeps its own OCP,
    exchange current, density and mechanics.
  * Silicon mechanics come from the swappable, cited SILICON_BLOCKS table
    (default "si" = Bonkile2024's published Si-Gr values). Swap the block
    when EDS resolves the bright phase (e.g. "siox" later).
  * Degradation constants are the published OKane2022 values
    (LAM proportional term 2.7778e-7 s-1, not 1e-3).
  * Capacity is read from the discharge step of each cycle
    (last - first of the running counter), not from cycle endpoints.
  * N/P is computed from the stoichiometry windows measured on the
    reference cell; cathode thickness is rescaled to hold the reference
    N/P when the anode geometry changes.

Outputs are CONSEQUENCE INDICATORS, not cell-test predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dfn_inputs import to_pybamm_params

def _trapezoid(y, x):
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


# ---------------------------------------------------------------------------
# parameter construction
# ---------------------------------------------------------------------------
def build_model(degradation: bool = True):
    """DFN with composite Si/Gr negative electrode (+ swelling/SEI/LAM)."""
    import pybamm
    options = {"particle phases": ("2", "1")}
    if degradation:
        options.update({
            "particle mechanics": ("swelling only", "swelling only"),
            "SEI": "solvent-diffusion limited",
            "loss of active material": "stress-driven",
        })
    return pybamm.lithium_ion.DFN(options)


# Donor whitelists — keys added ONLY if absent from Chen2020_composite.
# Bonkile2024 is the published Si-Gr composite parameterisation of the
# same LG M50 cell family (per-phase mechanics, cracking, LAM).
_BONKILE_MECH_TOKENS = (
    "Poisson", "Young", "partial molar volume", "volume change",
    "reference concentration for free of deformation",
    "critical stress", "initial crack", "number of cracks",
    "Paris' law", "cracking rate", "activation energy for cracking",
    "LAM constant", "reaction-driven LAM", "Initial SEI on cracks",
)
# OKane2022 shared SEI/degradation constants (unprefixed).
_OKANE_SHARED_TOKENS = (
    "SEI", "solvent", "interstitial", "EC ", "LAM", "Dead lithium",
    "Ratio of lithium", "reaction-driven",
)


def _si_block(name: str = "si") -> dict:
    """Swappable secondary-phase (silicon) parameter block.

    "si" — Bonkile2024 published values for lithiated Si in the LG M50
    Si-Gr composite.  All values cited; the uncertain ones are swept via
    the sweep parameters si_youngs_pa / si_nu in dfn_inputs.
    """
    import pybamm
    bonk = pybamm.ParameterValues("Bonkile2024")
    if name == "si":
        return {k: bonk[k] for k in bonk.keys()
                if k.startswith("Secondary: Negative electrode")
                and any(t in k for t in _BONKILE_MECH_TOKENS)}
    if name == "siox":
        raise NotImplementedError(
            "SiOx block: populate once EDS identifies the bright phase")
    raise KeyError(f"unknown silicon block {name!r}")


def base_parameters(si_block: str = "si"):
    """Chen2020_composite + *missing* published keys only.

    Merge rules (the repair):
      1. never overwrite an existing parameter;
      2. missing per-phase mechanics/LAM keys come from Bonkile2024;
      3. missing shared SEI constants come from OKane2022;
      4. the Secondary (silicon) phase is then deliberately set from the
         swappable, cited Si block — silicon stays silicon.
    """
    import pybamm
    param = pybamm.ParameterValues("Chen2020_composite")
    bonk = pybamm.ParameterValues("Bonkile2024")
    okane = pybamm.ParameterValues("OKane2022")

    for k in bonk.keys():
        if (k.startswith(("Primary: Negative electrode",
                          "Secondary: Negative electrode",
                          "Positive electrode"))
                and any(t in k for t in _BONKILE_MECH_TOKENS)
                and k not in param.keys()):
            param.update({k: bonk[k]}, check_already_exists=False)
    for k in okane.keys():
        if (any(t in k for t in _OKANE_SHARED_TOKENS)
                and not k.startswith(("Primary:", "Secondary:"))
                and k not in param.keys()):
            param.update({k: okane[k]}, check_already_exists=False)

    # deliberate per-phase LAM constants (published OKane2022 values)
    lam = {"LAM constant proportional term [s-1]": 2.7778e-7,
           "LAM constant exponential term": 2.0}
    for scope in ("Negative electrode", "Positive electrode",
                  "Primary: Negative electrode",
                  "Secondary: Negative electrode"):
        for suffix, v in lam.items():
            param.update({f"{scope} {suffix}": v},
                         check_already_exists=False)

    param.update(_si_block(si_block), check_already_exists=False)
    return param


def _legacy_buggy_parameters():
    """The SHIPPED (buggy) merge — kept only to produce the 'before'
    column of the ablation table.  Do not use for new results."""
    import pybamm
    param = pybamm.ParameterValues("Chen2020_composite")
    okane = pybamm.ParameterValues("OKane2022")
    for key, value in okane.items():
        if "Negative electrode" in key:
            param["Primary: " + key] = value
            param["Secondary: " + key] = value
        elif "Positive electrode" in key:
            param[key] = value
        elif ("SEI" in key or "LAM" in key or "mechanics" in key.lower()
              or "swelling" in key.lower() or "stress" in key.lower()):
            if key not in param:
                param[key] = value
    param.update({
        "SEI kinetic rate constant [m.s-1]": 1e-15,
        "Negative electrode LAM constant proportional term [s-1]": 1e-3,
        "Primary: Negative electrode LAM constant proportional term [s-1]":
            1e-3,
        "Secondary: Negative electrode LAM constant proportional term [s-1]":
            1e-3,
        "Positive electrode LAM constant proportional term [s-1]": 1e-3,
        "Negative electrode LAM constant exponential term": 2.0,
        "Primary: Negative electrode LAM constant exponential term": 2.0,
        "Secondary: Negative electrode LAM constant exponential term": 2.0,
        "Positive electrode LAM constant exponential term": 2.0,
    }, check_already_exists=False)
    return param


def constants_table() -> pd.DataFrame:
    """constant | published | ours | reason — the audit table."""
    import pybamm
    param = base_parameters()
    okane = pybamm.ParameterValues("OKane2022")
    bonk = pybamm.ParameterValues("Bonkile2024")
    shipped = _legacy_buggy_parameters()

    def val(d, k):
        try:
            v = d[k]
        except Exception:
            return "-"
        return getattr(v, "__name__", v)

    rows = [
        ("Primary: Negative electrode LAM constant proportional term "
         "[s-1]", val(okane, "Negative electrode LAM constant "
                     "proportional term [s-1]"),
         val(shipped, "Primary: Negative electrode LAM constant "
             "proportional term [s-1]"),
         val(param, "Primary: Negative electrode LAM constant "
              "proportional term [s-1]"),
         "OKane2022/Bonkile2024; shipped value was per-hour entered as "
         "per-second (3600x too fast)"),
        ("Secondary: Negative electrode LAM constant proportional term "
         "[s-1]", "2.7778e-07", val(shipped, "Secondary: Negative "
             "electrode LAM constant proportional term [s-1]"),
         val(param, "Secondary: Negative electrode LAM constant "
              "proportional term [s-1]"), "same published value"),
        ("Positive electrode LAM constant proportional term [s-1]",
         "2.7778e-07", "1e-3", val(param, "Positive electrode LAM "
            "constant proportional term [s-1]"), "published OKane2022"),
        ("SEI kinetic rate constant [m.s-1]", "1e-12",
         val(shipped, "SEI kinetic rate constant [m.s-1]"),
         val(param, "SEI kinetic rate constant [m.s-1]"),
         "OKane2022 published value (shipped was 1000x slower)"),
        ("Secondary: Negative electrode partial molar volume [m3.mol-1]",
         "1.2e-05 (Bonkile2024); task spec suggests ~9e-6 from "
         "2.8x12.06/3.75", val(shipped, "Secondary: Negative electrode "
             "partial molar volume [m3.mol-1]"),
         val(param, "Secondary: Negative electrode partial molar volume "
              "[m3.mol-1]"),
         "published coherent set; implies ~+334% volume at full "
         "lithiation via silicon_volume_change_Ai2020 (near the ~280% "
         "Li15Si4 estimate)"),
        ("Secondary: Negative electrode Young's modulus [Pa]",
         "50e9 (Bonkile2024); lithiated-Si literature range ~35-90 GPa",
         val(shipped, "Secondary: Negative electrode Young's modulus "
             "[Pa]"), val(param, "Secondary: Negative electrode "
                 "Young's modulus [Pa]"),
         "cited; swept over the cited range as si_youngs_pa"),
        ("Secondary: Negative electrode Poisson's ratio",
         "0.22 (Bonkile2024); cited range 0.22-0.28",
         val(shipped, "Secondary: Negative electrode Poisson's ratio"),
         val(param, "Secondary: Negative electrode Poisson's ratio"),
         "cited; swept as si_nu"),
        ("Secondary: Negative electrode critical stress [Pa]",
         "720e6 (Bonkile2024)",
         val(shipped, "Secondary: Negative electrode critical stress "
             "[Pa]"), val(param, "Secondary: Negative electrode "
                 "critical stress [Pa]"),
         "published Si value; shipped used graphite's 60e6"),
        ("Secondary: Negative electrode OCP [V]",
         "silicon_ocp_average_Mark2016 (Chen2020_composite)",
         getattr(shipped.get("Secondary: Negative electrode OCP [V]")
                 if hasattr(shipped, 'get') else
                 shipped["Secondary: Negative electrode OCP [V]"],
                 "__name__", "?"),
         "silicon_ocp_average_Mark2016", "left untouched by merge"),
        ("Secondary: Negative electrode density [kg.m-3]",
         "2650 (Chen2020_composite)", "1657", "2650",
         "left untouched by merge"),
        ("Inner/Outer SEI split keys", "not used by "
         "'solvent-diffusion limited'", "several hand-set values",
         "dropped", "dead parameters for a different SEI option; the "
         "model reads the unprefixed OKane2022 SEI keys"),
    ]
    return pd.DataFrame(rows, columns=["constant", "published",
                                       "shipped_buggy", "ours",
                                       "reason"])


# ---------------------------------------------------------------------------
# N/P ratio from stoichiometry windows (measured once on the reference cell)
# ---------------------------------------------------------------------------
_STO_VARS = {
    "n_pri": "X-averaged negative primary particle stoichiometry",
    "n_sec": "X-averaged negative secondary particle stoichiometry",
    "pos":   "X-averaged positive particle stoichiometry",
}


def measure_sto_windows(sol) -> dict:
    """Usable stoichiometry window per electrode phase, measured on a
    reference-cell solution."""
    out = {}
    for key, var in _STO_VARS.items():
        e = sol[var].entries
        out[key] = (float(np.min(e)), float(np.max(e)))
    return out


def np_ratio(param, sto_windows: dict) -> float:
    """N/P = usable anode capacity / usable cathode capacity.

    Q_electrode = A * L * F/3600 * sum_phases(eps_i * c_max_i * d_sto_i)
    with the stoichiometry windows measured on the reference cell.
    """
    import pybamm
    F = float(pybamm.constants.F.value) if hasattr(
        pybamm.constants.F, "value") else float(pybamm.constants.F)
    A = (param["Electrode height [m]"] * param["Electrode width [m]"])
    Ln = param["Negative electrode thickness [m]"]
    Lp = param["Positive electrode thickness [m]"]
    e1 = param["Primary: Negative electrode active material "
               "volume fraction"]
    e2 = param["Secondary: Negative electrode active material "
               "volume fraction"]
    c1 = param["Primary: Maximum concentration in negative electrode "
               "[mol.m-3]"]
    c2 = param["Secondary: Maximum concentration in negative electrode "
               "[mol.m-3]"]
    ep = param["Positive electrode active material volume fraction"]
    cp = param["Maximum concentration in positive electrode [mol.m-3]"]
    d1 = sto_windows["n_pri"][1] - sto_windows["n_pri"][0]
    d2 = sto_windows["n_sec"][1] - sto_windows["n_sec"][0]
    dp = sto_windows["pos"][1] - sto_windows["pos"][0]
    qn = A * Ln * F / 3600 * (e1 * c1 * d1 + e2 * c2 * d2)
    qp = A * Lp * F / 3600 * (ep * cp * dp)
    return float(qn / qp)


def hold_np(overrides: dict, param, sto_windows: dict,
            np_ref: float) -> dict:
    """Return overrides + a positive-electrode thickness that keeps the
    reference N/P under the new anode geometry."""
    o = dict(overrides)
    test = dict(overrides)
    # compute Q_n under overrides; scale L_p so Q_p = Q_n / np_ref
    tmp = {k: param[k] for k in param.keys()}
    # read effective params after overrides
    def get(k):
        return test[k] if k in test else tmp[k]
    import pybamm
    F = float(pybamm.constants.F.value) if hasattr(
        pybamm.constants.F, "value") else float(pybamm.constants.F)
    A = get("Electrode height [m]") * get("Electrode width [m]")
    Ln = get("Negative electrode thickness [m]")
    e1 = get("Primary: Negative electrode active material volume fraction")
    e2 = get("Secondary: Negative electrode active material "
             "volume fraction")
    c1 = get("Primary: Maximum concentration in negative electrode "
             "[mol.m-3]")
    c2 = get("Secondary: Maximum concentration in negative electrode "
             "[mol.m-3]")
    ep = get("Positive electrode active material volume fraction")
    cp = get("Maximum concentration in positive electrode [mol.m-3]")
    d1 = sto_windows["n_pri"][1] - sto_windows["n_pri"][0]
    d2 = sto_windows["n_sec"][1] - sto_windows["n_sec"][0]
    dp = sto_windows["pos"][1] - sto_windows["pos"][0]
    qn = A * Ln * F / 3600 * (e1 * c1 * d1 + e2 * c2 * d2)
    lp_needed = qn / np_ref * 3600 / (A * F * ep * cp * dp)
    o["Positive electrode thickness [m]"] = float(lp_needed)
    return o


# ---------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------
def _experiment(num_cycles: int):
    import pybamm
    return pybamm.Experiment([
        ("Discharge at 1C until 2.8V",
         "Rest for 10 minutes",
         "Charge at 1C until 4.2V",
         "Hold at 4.2V until 50mA",
         "Rest for 10 minutes")] * num_cycles)


def discharge_capacity_ah(cycle) -> float:
    """Discharge capacity of one cycle = counter last-first within the
    discharge step (steps[0]).  NOT the end-of-cycle endpoint, which is
    the running counter's post-recharge residual."""
    q = cycle.steps[0]["Discharge capacity [A.h]"].entries
    return float(q[-1] - q[0])


def discharge_capacity_integral(cycle) -> float:
    """Same quantity by integrating discharge current — used by the
    unit test to prove the readout."""
    step = cycle.steps[0]
    t = step["Time [s]"].entries
    i = step["Current [A]"].entries
    return float(_trapezoid(i, t) / 3600)


def _peak_during_discharge(sol, var: str, scale: float = 1.0) -> float:
    """Peak of `var` DURING each discharge step (max lithiation), across
    all cycles — not after the final rest."""
    try:
        peaks = [float(np.nanmax(st[var].entries) * scale)
                 for cyc in sol.cycles for st in cyc.steps[:1]]
        return float(np.nanmax(peaks)) if peaks else np.nan
    except Exception:
        return np.nan


# Declared up-front: the smallest differences we would call meaningful
# for batch comparison.  Anything smaller is reported but never read as
# a batch signal.  Rationale recorded in dfn_validation.md.
MEANINGFUL_DIFF = {
    "discharge_cap_last_Ah": 0.25,     # ~5% of ~5 Ah cell
    "thickness_change_um": 0.5,        # ~0.5% of ~75 um electrode
    "si_stress_MPa": 0.5,              # large vs ~0.005 MPa scale seen
    "min_neg_surface_v": 0.01,         # 10 mV plating-margin shift
}
# indicators compared between batches (LLI and retention excluded: they
# cannot move measurably in 5 cycles — kept as run diagnostics only)
COMPARE_INDICATORS = ("discharge_cap_last_Ah", "thickness_change_um",
                      "si_stress_MPa", "min_neg_surface_v")


def _cycle_metrics(sol, num_cycles: int) -> dict:
    caps_d, caps_c = [], []
    for cyc in sol.cycles:
        caps_d.append(discharge_capacity_ah(cyc))
        qc = 0.0
        for step in cyc.steps[1:]:
            q = step["Discharge capacity [A.h]"].entries
            qc -= (q[-1] - q[0])
        caps_c.append(qc)
    out = {"cycles_completed": len(sol.cycles)}
    for i, c in enumerate(caps_d, 1):
        out[f"discharge_cap_cycle{i}_Ah"] = c
    out["discharge_cap_last_Ah"] = caps_d[-1] if caps_d else np.nan
    ce = [d / c if c > 0 else np.nan for d, c in zip(caps_d, caps_c)]
    out["mean_coulombic_eff"] = float(np.nanmean(ce)) if ce else np.nan
    out["retention_pct"] = (float(100 * caps_d[-1] / caps_d[0])
                            if len(caps_d) > 1 and caps_d[0] > 0
                            else np.nan)
    return out


def shared_cathode_schedule(points: pd.DataFrame, param,
                            sto_windows: dict, np_ref: float) -> list:
    """Cathode thickness per sweep index, computed from ONE reference
    batch's anode points — then applied identically to every batch at
    the same sweep index, so paired runs share the same cathode."""
    from .dfn_inputs import to_pybamm_params
    return [hold_np(to_pybamm_params(row), param, sto_windows, np_ref)
            ["Positive electrode thickness [m]"]
            for _, row in points.iterrows()]


def solve_once(param_overrides: dict, num_cycles: int = 5,
               param=None, degradation: bool = True,
               sto_windows: dict | None = None,
               np_ref: float | None = None,
               cathode_thickness: float | None = None) -> dict:
    """One DFN run. Returns consequence indicators; NaNs on failure.

    cathode_thickness: if given, sets 'Positive electrode thickness [m]'
    (the shared cathode for this sweep index, from shared_cathode_schedule).
    np_ratio is still REPORTED per run — it just isn't force-held."""
    import pybamm
    model = build_model(degradation)
    if param is None:
        param = base_parameters()
    if cathode_thickness is not None:
        param_overrides = {**param_overrides,
                           "Positive electrode thickness [m]":
                           cathode_thickness}
    elif sto_windows is not None and np_ref is not None:
        param_overrides = hold_np(param_overrides, param, sto_windows,
                                  np_ref)
    param.update(param_overrides, check_already_exists=False)
    npv = np.nan
    try:
        npv = np_ratio(param, sto_windows) if sto_windows else np.nan
    except Exception:
        pass
    sim = pybamm.Simulation(model, parameter_values=param,
                            experiment=_experiment(num_cycles))
    out = {"np_ratio": npv}
    def _var(sol, name, scale=1.0):
        try:
            return float(sol[name].entries[-1] * scale)
        except Exception:
            return np.nan

    try:
        sim.solve()
        sol = sim.solution
        out["converged"] = 1
        out.update(_cycle_metrics(sol, num_cycles))
        out["loss_li_inventory_pct"] = _var(
            sol, "Loss of lithium inventory [%]")          # diagnostic
        out["thickness_change_um"] = _peak_during_discharge(
            sol, "Cell thickness change [m]", 1e6)         # in-cycle peak
        out["si_stress_MPa"] = _peak_during_discharge(
            sol, "X-averaged negative secondary particle surface "
                 "tangential stress [Pa]", 1e-6)           # in-cycle peak
        v = sol["X-averaged negative electrode surface potential "
                "difference [V]"].entries
        out["min_neg_surface_v"] = float(np.min(v))
        out["plating_flag"] = int(np.min(v) < 0.0)
        out["early_stop"] = int(len(sol.cycles) < num_cycles)
        return out
    except Exception as exc:                                   # noqa: BLE001
        out.update(converged=0, error=str(exc)[:300],
                   cycles_completed=0, early_stop=1,
                   loss_li_inventory_pct=np.nan,
                   thickness_change_um=np.nan, si_stress_MPa=np.nan,
                   discharge_cap_last_Ah=np.nan,
                   mean_coulombic_eff=np.nan, retention_pct=np.nan,
                   min_neg_surface_v=np.nan, plating_flag=np.nan)
        return out


def solve_reference(num_cycles: int = 5) -> dict:
    """Unmodified Chen2020_composite, no degradation, same protocol.
    Returns metrics + the solution for stoichiometry windows."""
    import pybamm
    param = pybamm.ParameterValues("Chen2020_composite")
    sim = pybamm.Simulation(build_model(degradation=False),
                            parameter_values=param,
                            experiment=_experiment(num_cycles))
    sim.solve()
    sol = sim.solution
    out = _cycle_metrics(sol, num_cycles)
    out["converged"] = 1
    out["min_neg_surface_v"] = float(np.min(
        sol["X-averaged negative electrode surface potential "
            "difference [V]"].entries))
    out["_sol"] = sol
    out["_param"] = param
    return out


# ---------------------------------------------------------------------------
# sweeps + paired comparison
# ---------------------------------------------------------------------------
def run_sweep(points: pd.DataFrame, num_cycles: int = 5,
              sto_windows=None, np_ref=None, progress=None,
              cathode_schedule=None) -> pd.DataFrame:
    rows = []
    for i, (_, point) in enumerate(points.iterrows()):
        if progress:
            progress(f"    sweep {i + 1}/{len(points)}")
        res = solve_once(to_pybamm_params(point), num_cycles,
                         sto_windows=sto_windows, np_ref=np_ref,
                         cathode_thickness=(cathode_schedule[i]
                                            if cathode_schedule else
                                            None))
        rows.append({**point.to_dict(), **res})
    return pd.DataFrame(rows)


INDICATOR_COLS = ("discharge_cap_last_Ah", "thickness_change_um",
                  "si_stress_MPa", "min_neg_surface_v", "np_ratio",
                  "retention_pct", "loss_li_inventory_pct",
                  "mean_coulombic_eff")


def paired_differences(sweeps: dict[str, pd.DataFrame],
                       indicators=COMPARE_INDICATORS,
                       reference="Batch_3",
                       ref_min_neg: float | None = None) -> pd.DataFrame:
    """One row per (pair, indicator): median paired difference, its range,
    the assumption-only spread (pooled IQR), the declared meaningful-
    difference threshold, and a verdict.

    Verdicts (declared up front):
      meaningful_difference — |median| > threshold AND the paired range
                              excludes 0
      below_threshold       — |median| and the whole range sit inside
                              ±threshold
      inconclusive          — anything else (e.g. CI straddles 0 or thr)
    Plating (reversed rule): a sub-zero anode potential counts as a
    plating WARNING only for runs where the reference cell under the
    same protocol stays >= 0 — if the reference also dips below zero it
    is a protocol/geometry artefact, not a batch signal.
    """
    pooled = pd.concat(sweeps.values(), ignore_index=True)
    ok = pooled[pooled["converged"] == 1]
    spread = {c: float(np.subtract(*np.nanpercentile(
        ok[c].dropna(), [75, 25]))) if ok[c].notna().any() else np.nan
        for c in indicators}

    rows = []
    pairs = [("Batch_1", "Batch_3"), ("Batch_2", "Batch_3"),
             ("Batch_1", "Batch_2")]
    for a, b in pairs:
        if a not in sweeps or b not in sweeps:
            continue
        va = sweeps[a].reset_index(drop=True)
        vb = sweeps[b].reset_index(drop=True)
        n = min(len(va), len(vb))
        # plating warnings vs reference (reversed rule)
        if ref_min_neg is not None:
            warn_a = int(((va["min_neg_surface_v"].iloc[:n] < 0)
                          & (ref_min_neg >= 0)).sum())
            warn_b = int(((vb["min_neg_surface_v"].iloc[:n] < 0)
                          & (ref_min_neg >= 0)).sum())
        else:
            warn_a = warn_b = None
        for c in indicators:
            d = (va[c].iloc[:n] - vb[c].iloc[:n]).dropna()
            if len(d) == 0:
                continue
            med = float(d.median())
            thr = MEANINGFUL_DIFF.get(c, np.nan)
            lo, hi = float(d.min()), float(d.max())
            if np.isfinite(thr) and abs(med) > thr and \
                    (lo > 0 or hi < 0):
                verdict = "meaningful_difference"
            elif np.isfinite(thr) and abs(med) <= thr \
                    and abs(lo) <= thr and abs(hi) <= thr:
                verdict = "below_threshold"
            else:
                verdict = "inconclusive"
            rows.append(dict(
                pair=f"{a}-{b}", indicator=c, n_paired=len(d),
                median_paired_diff=med,
                diff_lo=lo, diff_hi=hi,
                assumption_only_spread_iqr=spread[c],
                declared_threshold=thr,
                exceeds_assumption_spread=bool(
                    np.isfinite(spread[c]) and abs(med) > spread[c]),
                verdict=verdict,
                plating_warnings_a=warn_a,
                plating_warnings_b=warn_b))
    return pd.DataFrame(rows)


def summarise_sweep(sweep_df: pd.DataFrame) -> dict:
    ok = sweep_df[sweep_df["converged"] == 1]
    out = {"n_runs": len(sweep_df), "n_converged": len(ok)}
    for col in INDICATOR_COLS + ("discharge_cap_cycle1_Ah",
                                 "mean_coulombic_eff"):
        if col in ok and len(ok) and ok[col].notna().any():
            v = ok[col].dropna()
            out[col + "_med"] = float(v.median())
            out[col + "_lo"] = float(v.min())
            out[col + "_hi"] = float(v.max())
        else:
            out[col + "_med"] = out[col + "_lo"] = out[col + "_hi"] = \
                np.nan
    out["plating_flag_any"] = int(ok["plating_flag"].max()) \
        if len(ok) and ok["plating_flag"].notna().any() else np.nan
    return out



