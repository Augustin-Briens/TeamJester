#!/usr/bin/env python3
"""micro2dfn — image markers -> PyBaMM DFN consequence indicators.

    python run_dfn.py --data Hackathon-Polaron --out dfn_output
    python run_dfn.py --no-sim     # markers & stats only, no PyBaMM
    python run_dfn.py --fast       # dev pass: 2x downsample, 3 cycles,
                                   # 9 sweep points — NOT the published
                                   # configuration

Final configuration (default): full resolution, 5 x 1C cycles,
16 sweep points per batch per variant.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import micro2dfn.config as config
from micro2dfn import io, segment, objects as obj_, markers as mk
from micro2dfn import dfn_inputs, simulate, report

# run configuration (defaults = the published numbers)
DFN_CYCLES = 5
DFN_SWEEP_POINTS = 16
DFN_CYCLES_FAST = 3
DFN_SWEEP_FAST = 9
DOWNSAMPLE_FAST = 2
N_DEBUG_FIGS = 2


def _file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_manifest(out_dir: str, recipe_path: str, rec: dict,
                       images: dict, fast: bool, downsample: int,
                       baseline: str) -> str:
    """Identify exactly which run produced an output: recipe hash,
    code hashes, and the md5 of every input image."""
    code_files = sorted(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "micro2dfn", "*.py")) + [os.path.abspath(__file__)])
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime()),
        "baseline_batch": baseline,
        "downsample": downsample, "fast": fast,
        "recipe_path": recipe_path,
        "recipe_md5": _file_md5(recipe_path),
        "recipe_fitted_on": rec.get("fitted_on"),
        "code_md5": {os.path.basename(f): _file_md5(f)
                     for f in code_files},
        "images": {b: {im.image_id: _file_md5(im.path) for im in ims}
                   for b, ims in images.items()},
    }
    path = os.path.join(out_dir, "run_manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    return path


# DFN sweep variants, all paired on the same LHS seed:
#   accessible          — Si share bracket [classified x accessible,
#                         all-uncertain-bright]   (primary analysis)
#   all_si_active       — no accessibility discount on classified Si
#   no_problem_photos   — bands rebuilt without is_problem_photo images
VARIANTS = ("accessible", "all_si_active", "no_problem_photos")


def run_validation(n_cycles: int, batch3_median_point: dict,
                   print_) -> dict:
    """Reference-cell check + B3-median gate + ablation + unit tests."""
    import pybamm
    res = {"pybamm_version": pybamm.__version__, "n_cycles": n_cycles}
    print_("  reference cell (Chen2020_composite, no degradation)...")
    ref = simulate.solve_reference(num_cycles=n_cycles)
    res["reference"] = {k: v for k, v in ref.items()
                        if not k.startswith("_")}
    res["sto_windows"] = simulate.measure_sto_windows(ref["_sol"])
    res["np_ref"] = simulate.np_ratio(ref["_param"],
                                      res["sto_windows"])
    res["ref_plating_flag"] = int(ref["min_neg_surface_v"] < 0)

    tests = []
    cap1 = ref.get("discharge_cap_cycle1_Ah", np.nan)
    tests.append(("reference discharge capacity within 10% of "
                  "nominal 5 Ah", np.isfinite(cap1)
                  and abs(cap1 - 5.0) / 5.0 <= 0.10,
                  f"{cap1:.3f} Ah"))
    cyc = [v for k, v in sorted(ref.items())
           if k.startswith("discharge_cap_cycle")]
    if len(cyc) >= 3:
        drift = abs(cyc[-1] - cyc[0]) / cyc[0] * 100
        tests.append((f"reference drift < 0.5% over {len(cyc)} cycles",
                      drift < 0.5, f"{drift:.3f}%"))
    # capacity readout unit test: counter last-first == integral of I dt
    d_step = simulate.discharge_capacity_ah(ref["_sol"].cycles[0])
    d_int = simulate.discharge_capacity_integral(ref["_sol"].cycles[0])
    tests.append(("per-cycle discharge capacity == integral of "
                  "discharge current",
                  abs(d_step - d_int) / max(d_int, 1e-9) < 0.01,
                  f"{d_step:.4f} vs {d_int:.4f} Ah"))
    # silicon stays silicon
    p = simulate.base_parameters()
    diffs = []
    for suffix in ("OCP [V]", "exchange-current density [A.m-2]",
                   "Young's modulus [Pa]", "Poisson's ratio",
                   "partial molar volume [m3.mol-1]"):
        a = p[f"Primary: Negative electrode {suffix}"]
        b = p[f"Secondary: Negative electrode {suffix}"]
        diffs.append(getattr(a, "__name__", a)
                     != getattr(b, "__name__", b))
    tests.append(("silicon (Secondary) OCP, kinetics & mechanics differ "
                  "from graphite (Primary)", all(diffs), ""))
    # Si volume-change function present; implied full-lithiation
    # volume change = Omega_Si * c_max_Si (linear Ai2020 model)
    try:
        om = float(p["Secondary: Negative electrode partial molar "
                     "volume [m3.mol-1]"])
        cm = float(p["Secondary: Maximum concentration in negative "
                     "electrode [mol.m-3]"])
        v1 = om * cm
        tests.append(("Si volume-change function ~+300% at full "
                      "lithiation",
                      2.0 < v1 < 4.0, f"+{v1 * 100:.0f}%"))
    except Exception as e:
        tests.append(("Si volume-change function ~+300% at full "
                      "lithiation", False, str(e)[:80]))

    print_("  B3 median point on the fixed model...")
    b3 = simulate.solve_once(batch3_median_point, num_cycles=n_cycles,
                             sto_windows=res["sto_windows"],
                             np_ref=res["np_ref"])
    res["b3_median"] = b3
    ref_cap = ref.get("discharge_cap_cycle1_Ah", np.nan)
    tests.append(("B3 median capacity within 15% of reference cell",
                  np.isfinite(b3.get("discharge_cap_cycle1_Ah"))
                  and abs(b3["discharge_cap_cycle1_Ah"] - ref_cap)
                  / ref_cap <= 0.15,
                  f"{b3.get('discharge_cap_cycle1_Ah', np.nan):.3f} "
                  f"vs {ref_cap:.3f} Ah"))
    if n_cycles >= 5:
        tests.append(("B3 median retention >= 98% after 5 cycles",
                      np.isfinite(b3.get("retention_pct"))
                      and b3["retention_pct"] >= 98,
                      f"{b3.get('retention_pct', np.nan):.2f}%"))
        tests.append(("B3 median lithium loss < 2% after 5 cycles",
                      np.isfinite(b3.get("loss_li_inventory_pct"))
                      and b3["loss_li_inventory_pct"] < 2,
                      f"{b3.get('loss_li_inventory_pct', np.nan):.2f}%"))
    res["tests"] = pd.DataFrame(
        [(n, bool(ok), note) for n, ok, note in tests],
        columns=["test", "pass", "detail"])

    print_("  ablation table (before/after)...")
    res["ablation"] = _run_ablation(n_cycles, batch3_median_point,
                                    res["sto_windows"], res["np_ref"],
                                    print_)
    res["constants"] = simulate.constants_table()
    return res


def _run_ablation(n_cycles, b3_point, sto_windows, np_ref, print_):
    """reference -> +our geometry -> +degradation -> +Si mechanics
    -> full.  Stage 2 uses the SHIPPED buggy parameter set (the
    'before' column); stage 3 the repaired model ('after')."""
    rows = []

    def grab(label, stage_out):
        r = {"stage": label}
        for k in ("discharge_cap_cycle1_Ah", "discharge_cap_last_Ah",
                  "retention_pct", "loss_li_inventory_pct",
                  "min_neg_surface_v", "np_ratio", "converged",
                  "cycles_completed"):
            r[k] = stage_out.get(k, np.nan)
        return r

    print_("    0 reference (unchanged Chen2020_composite)")
    ref = simulate.solve_reference(num_cycles=n_cycles)
    rows.append(grab("0_reference", ref))

    print_("    1 + our geometry (no degradation)")
    out = simulate.solve_once(b3_point, n_cycles, degradation=False,
                              sto_windows=sto_windows, np_ref=np_ref)
    rows.append(grab("1_geometry", out))

    print_("    2 + degradation on SHIPPED (buggy) parameters")
    out = simulate.solve_once(b3_point, n_cycles,
                              param=simulate._legacy_buggy_parameters(),
                              sto_windows=sto_windows, np_ref=np_ref)
    rows.append(grab("2_degradation_shipped", out))

    print_("    3 full repaired model (Si mechanics + published consts)")
    out = simulate.solve_once(b3_point, n_cycles,
                              param=simulate.base_parameters(),
                              sto_windows=sto_windows, np_ref=np_ref)
    rows.append(grab("3_full_fixed", out))

    df = pd.DataFrame(rows)
    df["note"] = [
        "unmodified Chen2020_composite, no degradation",
        "our B3 geometry, cathode rescaled to reference N/P, "
        "no degradation",
        "BEFORE: graphite params copied onto Si, LAM 1e-3 s-1, "
        "hand-set SEI",
        "AFTER: Si keeps Si chemistry/mechanics (Bonkile2024), "
        "published LAM/SEI — the full model",
    ]
    return df


def write_validation_md(res: dict, out_dir: str, settings: dict) -> str:
    L = ["# DFN validation\n"]
    L.append(f"PyBaMM {res['pybamm_version']}; protocol: 1C "
             f"discharge->2.8 V, rest, 1C charge->4.2 V, CV hold "
             f"<50 mA, rest; {res['n_cycles']} cycles.\n")
    L.append("## Reference cell (unmodified Chen2020_composite, no "
             "degradation)\n")
    ref = res["reference"]
    caps = " -> ".join(
        f"{ref[k]:.3f}" for k in sorted(ref)
        if k.startswith("discharge_cap_cycle"))
    L.append(f"- discharge capacity per cycle: {caps} Ah")
    L.append(f"- mean coulombic efficiency: "
             f"{ref.get('mean_coulombic_eff', np.nan):.4f}")
    L.append(f"- min anode surface potential: "
             f"{ref.get('min_neg_surface_v', np.nan):+.4f} V "
             f"({'plates' if res['ref_plating_flag'] else 'no plating'}"
             " — a sweep run counts as a plating WARNING only if it "
             "dips below 0 V while this reference run stays >= 0; if "
             "the reference also dips, sub-zero is a protocol artefact "
             "and is not attributed to any batch)")
    sw = res["sto_windows"]
    L.append(f"- measured stoichiometry windows: graphite "
             f"{sw['n_pri'][0]:.3f}-{sw['n_pri'][1]:.3f}, Si "
             f"{sw['n_sec'][0]:.3f}-{sw['n_sec'][1]:.3f}, NMC "
             f"{sw['pos'][0]:.3f}-{sw['pos'][1]:.3f}")
    L.append(f"- reference N/P (from stoichiometry windows): "
             f"{res['np_ref']:.3f}\n")

    L.append("## Batch_3 median point (repaired model)\n")
    b3 = res["b3_median"]
    L.append(f"- discharge cap cycle 1: "
             f"{b3.get('discharge_cap_cycle1_Ah', np.nan):.3f} Ah; "
             f"retention {b3.get('retention_pct', np.nan):.2f}%; "
             f"LLI {b3.get('loss_li_inventory_pct', np.nan):.2f}%; "
             f"min anode V {b3.get('min_neg_surface_v', np.nan):+.4f}; "
             f"N/P {b3.get('np_ratio', np.nan):.3f}\n")

    L.append("## Ablation — before vs after the fix\n")
    ab = res["ablation"]
    hdr = ["stage"] + [c for c in ab.columns
                       if c not in ("stage", "note")] + ["note"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for _, r in ab.iterrows():
        L.append("| " + " | ".join(
            f"{r[c]:.4g}" if isinstance(r[c], float) else str(r[c])
            for c in hdr) + " |")

    L.append("\n## Constants audit (ours vs published)\n")
    ct = res["constants"]
    L.append("| " + " | ".join(ct.columns) + " |")
    L.append("|" + "---|" * len(ct.columns))
    for _, r in ct.iterrows():
        L.append("| " + " | ".join(str(r[c]) for c in ct.columns)
                 + " |")

    L.append("\n## Tests\n")
    for _, r in res["tests"].iterrows():
        L.append(f"- [{'x' if r['pass'] else ' '}] {r['test']}"
                 + (f" — {r['detail']}" if r['detail'] else ""))
    n_pass = int(res["tests"]["pass"].sum())
    L.append(f"\n**{n_pass}/{len(res['tests'])} tests pass.**\n")

    L.append("## Declared comparison thresholds (fixed BEFORE the "
             "sweeps)\n")
    L.append("A batch difference only counts as meaningful when its "
             "median paired difference exceeds these values AND the "
             "paired range excludes 0:\n")
    for k, v in simulate.MEANINGFUL_DIFF.items():
        L.append(f"- `{k}`: {v}")
    L.append("\nCompared indicators: "
             + ", ".join(simulate.COMPARE_INDICATORS)
             + ". Lithium-inventory loss and capacity retention are "
             "kept in `dfn_results.csv` as run diagnostics only — they "
             "cannot move measurably in 5 cycles and play no role in "
             "the batch comparison. Swelling (thickness change) and Si "
             "surface stress are read as the PEAK during each "
             "discharge step, not at the final rest.\n")
    L.append("## Run settings actually used\n")
    for k, v in settings.items():
        L.append(f"- {k}: {v}")
    path = os.path.join(out_dir, "dfn_validation.md")
    with open(path, "w") as fh:
        fh.write("\n".join(L))
    return path


def run(out_dir: str = "dfn_output", sim: bool = True,
        fast: bool = False, data_dir: str | None = None,
        baseline: str = "Batch_3", recipe: str | None = None,
        reuse: bool = False):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    data_dir = data_dir or "."
    batches = io.discover_batches(data_dir)
    assert batches, f"no Batch_* folders with BSE TIFFs under {data_dir}"
    print(f"Found {len(batches)} batches under {data_dir}: "
          + ", ".join(f"{b} ({len(f)} imgs)"
                      for b, f in batches.items()))

    downsample = DOWNSAMPLE_FAST if fast else 1
    if downsample > 1:
        # keep the ~0.35 um size floor: pixel cutoffs scale by
        # 1/downsample^2 (area in px shrinks quadratically)
        config.MIN_BRIGHT_PX = max(
            1, round(config.MIN_BRIGHT_PX / downsample ** 2))
        config.MIN_FEATURE_PX = max(
            1, round(config.MIN_FEATURE_PX / downsample ** 2))
        print(f"  --fast: downsample x{downsample}, pixel floors "
              f"scaled to {config.MIN_BRIGHT_PX}/"
              f"{config.MIN_FEATURE_PX} px")

    recipe_path = recipe or os.path.join(out_dir, "recipe.json")
    markers_csv = os.path.join(out_dir, "markers_all.csv")
    objects_csv = os.path.join(out_dir, "objects_all.csv")

    if reuse and os.path.exists(markers_csv):
        # cached measurements: identical recipe + images by construction
        # (manifest records both); skips load/segment/extract entirely
        markers = pd.read_csv(markers_csv)
        objects = pd.read_csv(objects_csv)
        with open(recipe_path) as fh:
            rec = json.load(fh)
        t_pore, t_si, t_core = rec["t_pore"], rec["t_si"], rec["t_core"]
        manifest_path = os.path.join(out_dir, "run_manifest.json")
        print(f"--reuse: loaded {len(markers)} cached marker rows "
              f"+ frozen recipe {recipe_path} "
              f"(fitted on {rec['fitted_on']})")
    else:
        # ---- FROZEN RECIPE ---------------------------------------------
        # The measuring rules must not move when a new batch arrives:
        # thresholds + object classifier are fitted on the BASELINE
        # batch only, saved to recipe.json, and reused verbatim on
        # later runs (or loaded from --recipe). A new batch can never
        # change how the old batches are measured.
        print("\nLoading images")
        images = {b: [io.load_bse(p, b, downsample=downsample)
                      for p in files]
                  for b, files in batches.items()}
        n_im = sum(len(v) for v in images.values())

        if os.path.exists(recipe_path):
            with open(recipe_path) as fh:
                rec = json.load(fh)
            t_pore = rec["t_pore"]
            t_si = rec["t_si"]
            t_core = rec["t_core"]
            print(f"frozen recipe loaded from {recipe_path} "
                  f"(fitted on {rec['fitted_on']}, "
                  f"{rec['n_fit_images']} images, {rec['created_utc']})")
        else:
            ref = baseline if baseline in images \
                else sorted(batches)[-1]
            ref_im = images[ref]
            print(f"fitting recipe on baseline {ref} "
                  f"({len(ref_im)} images) — frozen thereafter")
            t_pore, t_si = segment.fit_thresholds(ref_im)
            base_objects = pd.concat(
                [obj_.extract_objects(
                    im, segment.segment(im, t_pore, t_si) == config.SI)
                 for im in ref_im], ignore_index=True)
            t_core = obj_.calibrate_t_core(base_objects)
            rec = {"recipe_version": "micro2dfn-2.0", "fitted_on": ref,
                   "n_fit_images": len(ref_im),
                   "t_pore": float(t_pore), "t_si": float(t_si),
                   "t_core": float(t_core),
                   "created_utc": time.strftime(
                       "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "fit_image_md5": {im.image_id: _file_md5(im.path)
                                     for im in ref_im}}
            with open(recipe_path, "w") as fh:
                json.dump(rec, fh, indent=1)
            print(f"  recipe saved -> {recipe_path}")
        print(f"  t_pore={t_pore:.4f}  t_si={t_si:.4f}  "
              f"t_core={t_core:.4f}")
        segment.save_thresholds(
            os.path.join(out_dir, "thresholds.json"), t_pore, t_si,
            n_im)
        manifest_path = write_run_manifest(
            out_dir, recipe_path, rec, images, fast, downsample,
            baseline)
        print(f"  run manifest -> {manifest_path}")

        # segment + object extraction (frozen thresholds; classifier
        # calibrated on baseline only when the recipe was just fitted)
        print("\nSegmenting + extracting objects")
        per_image = {}
        all_objects = []
        for b, ims in images.items():
            for im in ims:
                print(f"  {b}/{im.image_id}", flush=True)
                seg = segment.segment(im, t_pore, t_si)
                ob = obj_.extract_objects(im, seg == config.SI)
                per_image[im.image_id] = (im, seg, ob)
                all_objects.append(ob)
        objects = pd.concat(all_objects, ignore_index=True)
        objects = obj_.classify_objects(objects, t_core)
        print(f"  {len(objects)} bright objects; classifier "
              f"t_core={t_core:.3f} (baseline-fitted)")

        # markers
        print("\nComputing markers")
        mk_rows = []
        for b, ims in images.items():
            for im in ims:
                seg = per_image[im.image_id][1]
                part, fine = obj_.particle_mask(im, seg == config.SI,
                                                objects)
                f = mk.extract_markers(im, seg, objects, part, fine)
                mk_rows.append(f)
                if len(mk_rows) <= N_DEBUG_FIGS:
                    report.fig_classification_overlay(
                        im.raw, part, fine, seg == config.PORE,
                        f"{b}/{im.image_id}",
                        os.path.join(out_dir, "figs",
                                     f"seg_{b}_{im.image_id}.png"))
        markers = pd.DataFrame(mk_rows)
        # problem-photo flag: Si/bulk contrast below the pooled 5th pct
        cut = markers["si_bulk_contrast"].quantile(0.05)
        markers["is_problem_photo"] = (
            markers["si_bulk_contrast"] < cut).astype(int)

        markers.to_csv(markers_csv, index=False)
        objects.to_csv(objects_csv, index=False)
    for b in sorted(batches):
        markers[markers.batch == b].to_csv(
            os.path.join(out_dir, f"markers_{b}.csv"), index=False)
        objects[objects.batch == b].to_csv(
            os.path.join(out_dir, f"objects_{b}.csv"), index=False)
    print(f"\nMarkers: {len(markers)} images, {len(objects)} objects, "
          f"{int(markers['is_problem_photo'].sum())} problem photos")

    summary = report.batch_summary(markers)
    deltas = report.batch_deltas(markers)
    summary.to_csv(os.path.join(out_dir, "batch_summary.csv"),
                   index=False)
    deltas.to_csv(os.path.join(out_dir, "batch_differences.csv"),
                  index=False)
    report.write_data_dictionary(markers, out_dir)
    report.fig_marker_distributions(markers, out_dir)

    sweep_all, dfn_summary, paired, validation = None, None, None, None
    settings = {
        # run provenance — every report identifies exactly which
        # recipe/images/code produced it (frozen-measurement rule)
        "recipe": recipe_path,
        "recipe_fitted_on": rec.get("fitted_on"),
        "recipe_md5": _file_md5(recipe_path)
        if os.path.exists(recipe_path) else None,
        "run_manifest": manifest_path
        if os.path.exists(manifest_path) else None,
        "reuse_cached_markers": bool(reuse),
    }
    if sim:
        n_cycles = DFN_CYCLES_FAST if fast else DFN_CYCLES
        n_pts = DFN_SWEEP_FAST if fast else DFN_SWEEP_POINTS
        import pybamm
        settings.update({
            "pybamm_version": pybamm.__version__,
            "downsample": downsample,
            "cycles": n_cycles,
            "sweep_points_per_variant": n_pts + 1,
            "variants": ", ".join(VARIANTS),
            "si_bracket": "lo = classified Si x accessible share; "
                          "hi = all uncertain bright counted",
            "lam_proportional_s^-1": 2.7778e-7,
            "np_hold": "one shared cathode per sweep index, sized on "
                      "the reference batch's anode points (same for all "
                      "batches at that index)",
            "seed": 0,
        })

        ref_batch = "Batch_3" if "Batch_3" in set(markers.batch) \
            else sorted(batches)[-1]
        bands_ref = dfn_inputs.batch_bands(
            markers[markers.batch == ref_batch])
        b3_point = dfn_inputs.to_pybamm_params(
            dfn_inputs.sweep_points(bands_ref, n_points=0)
            .iloc[0].to_dict())

        print("\nDFN validation stage")
        validation = run_validation(
            n_cycles, b3_point, lambda s: print(s, flush=True))
        sto_windows = validation["sto_windows"]
        np_ref = validation["np_ref"]
        n_pass = int(validation["tests"]["pass"].sum())
        settings["ref_min_neg_v"] = float(
            validation["reference"].get("min_neg_surface_v", np.nan))
        print(f"  validation: {n_pass}/{len(validation['tests'])} "
              f"tests pass; ref N/P {np_ref:.3f}")

        print("\nDFN sweeps (paired same-seed LHS; shared cathode per "
              "sweep index)")
        base_param = simulate.base_parameters()
        frames = {}
        for variant in VARIANTS:
            variant_points = {}
            for batch in sorted(batches):
                f = markers[markers.batch == batch]
                bands = dfn_inputs.batch_bands(
                    f,
                    accessible=1.0 if variant == "all_si_active"
                    else None,
                    exclude_problem=variant == "no_problem_photos")
                variant_points[batch] = dfn_inputs.sweep_points(
                    bands, n_points=n_pts)
            # same cathode at each sweep index for every batch —
            # sized on the reference batch's anode points
            schedule = simulate.shared_cathode_schedule(
                variant_points[ref_batch], base_param,
                sto_windows, np_ref)
            for batch in sorted(batches):
                points = variant_points[batch]
                print(f"  {variant} / {batch}: {len(points)} pts x "
                      f"{n_cycles} cyc", flush=True)
                df = simulate.run_sweep(
                    points, n_cycles, sto_windows, np_ref,
                    progress=lambda s: print(s, end="\r", flush=True),
                    cathode_schedule=schedule)
                df["batch"] = batch
                df["variant"] = variant
                frames[(variant, batch)] = df
                df.to_csv(os.path.join(
                    out_dir, f"dfn_sweep_{variant}_{batch}.csv"),
                    index=False)
                print(" " * 40, end="\r")
        sweep_all = pd.concat(frames.values(), ignore_index=True)
        sweep_all.to_csv(os.path.join(out_dir, "dfn_results.csv"),
                         index=False)

        main = {b: frames[("accessible", b)] for b in sorted(batches)}
        dfn_summary = pd.DataFrame([
            {"batch": b, **simulate.summarise_sweep(df)}
            for b, df in main.items()])
        dfn_summary.to_csv(
            os.path.join(out_dir, "dfn_indicator_summary.csv"),
            index=False)
        paired = simulate.paired_differences(
            main,
            ref_min_neg=validation["reference"].get(
                "min_neg_surface_v"))
        paired.to_csv(
            os.path.join(out_dir, "dfn_paired_differences.csv"),
            index=False)

        for b in sorted(batches):
            frames[("accessible", b)][
                list(dfn_inputs.SWEEP_PARAMS)].to_json(
                os.path.join(out_dir, f"dfn_params_{b}.json"),
                orient="records", indent=1)
        bands_all = pd.concat([
            dfn_inputs.batch_bands(markers[markers.batch == b])
            .assign(batch=b) for b in sorted(batches)])
        bands_all.to_csv(os.path.join(out_dir, "dfn_input_bands.csv"))
        at = pd.concat([
            dfn_inputs.run_assumption_table(
                bands_all[bands_all.batch == b].drop(columns="batch"),
                frames[("accessible", b)]
                [list(dfn_inputs.SWEEP_PARAMS)])
            .assign(batch=b) for b in sorted(batches)])
        at.to_csv(os.path.join(out_dir, "assumption_table.csv"),
                  index=False)

        report.fig_consequence_bands(dfn_summary, out_dir)
        tor = report.fig_tornado(
            sweep_all[sweep_all["variant"] == "accessible"], out_dir)
        if tor:
            print(f"  wrote {tor}")
        write_validation_md(validation, out_dir, settings)

    report.write_report(out_dir, markers, summary, deltas,
                        dfn_summary, t_pore, t_si, t_core,
                        settings=settings if settings else None,
                        paired=paired)
    report.write_readme(out_dir)
    xlsx = report.write_excel(out_dir, markers, summary, deltas,
                              sweep_all, dfn_summary)
    print(f"\nExcel workbook -> {xlsx}")
    print(f"Done in {time.time() - t0:.0f}s -> {out_dir}/report.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="dfn_output")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="dev mode: 2x downsample, 3 cycles, 9 points. "
                         "Final outputs must come from the default "
                         "full-resolution 5-cycle run.")
    ap.add_argument("--data", default=None,
                    help="root dir holding Batch_* folders")
    ap.add_argument("--baseline", default="Batch_3",
                    help="batch the measuring recipe is fitted on "
                         "(default Batch_3, the approved reference)")
    ap.add_argument("--recipe", default=None,
                    help="path to a frozen recipe.json to reuse "
                         "verbatim (default: <out>/recipe.json, created "
                         "on first run and reused thereafter)")
    ap.add_argument("--reuse", action="store_true",
                    help="skip image load/segment/extract; reuse "
                         "cached markers_all.csv + objects_all.csv "
                         "under the frozen recipe")
    a = ap.parse_args()
    try:
        run(a.out, sim=not a.no_sim, fast=a.fast, data_dir=a.data,
            baseline=a.baseline, recipe=a.recipe, reuse=a.reuse)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
