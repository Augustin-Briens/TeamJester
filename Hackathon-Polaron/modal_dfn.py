"""Modal runner for the micro2dfn pipeline — parallel full-res run.

    .venv/bin/python -m modal run modal_dfn.py            # full-res 5cyc x16pts
    .venv/bin/python -m modal run modal_dfn.py --fast     # dev pass
    .venv/bin/python -m modal run modal_dfn.py --no-sim   # markers only

Runs the two expensive stages remotely in parallel:
  * per-image extraction (segment -> objects -> classify -> markers)
  * per-point DFN solves (variants x batches x sweep points)
All statistics/recipes are still computed locally from the returned
tables, so the outputs are identical in structure to run_dfn.py.
"""
import glob
import os

import modal

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = "/data"

_DEPS = [
    "pybamm==26.9.0.0", "numpy==2.5.3", "pandas==3.0.6",
    "scipy==1.18.1", "scikit-image==0.26.0", "tifffile==2026.9.20",
    "imagecodecs==2026.8.16",   # LZW-compressed TIFFs
    "openpyxl==3.1.5", "matplotlib==3.11.2",
]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(*_DEPS)
    .add_local_python_source("micro2dfn")
    .add_local_dir(os.path.join(ROOT, "Batch_1"), f"{DATA_ROOT}/Batch_1")
    .add_local_dir(os.path.join(ROOT, "Batch_2"), f"{DATA_ROOT}/Batch_2")
    .add_local_dir(os.path.join(ROOT, "Batch_3"), f"{DATA_ROOT}/Batch_3")
)

app = modal.App("micro2dfn-dfn", image=image)


# --------------------------------------------------------------------------
# remote stages
# --------------------------------------------------------------------------
@app.function(timeout=1800, memory=8192, cpu=4)
def fit_thresholds_remote(batch_paths: dict, downsample: int):
    """Load every image, fit the ONE shared threshold pair."""
    from micro2dfn import io, segment
    images = [io.load_bse(p, b, downsample=downsample)
              for b, paths in batch_paths.items() for p in paths]
    t_pore, t_si = segment.fit_thresholds(images)
    return float(t_pore), float(t_si)


@app.function(timeout=1800)
def objects_remote(path: str, batch: str, t_pore: float, t_si: float,
                   downsample: int):
    """Per-image: load -> segment -> watershed objects (unclassified)."""
    from micro2dfn import io, segment, config, objects as obj_
    if downsample > 1:
        config.MIN_BRIGHT_PX = max(1, round(config.MIN_BRIGHT_PX
                                          / downsample ** 2))
        config.MIN_FEATURE_PX = max(1, round(config.MIN_FEATURE_PX
                                           / downsample ** 2))
    im = io.load_bse(path, batch, downsample=downsample)
    seg = segment.segment(im, t_pore, t_si)
    ob = obj_.extract_objects(im, seg == config.SI)
    return ob.to_dict("records")


@app.function(timeout=1800)
def markers_remote(path: str, batch: str, t_pore: float, t_si: float,
                   downsample: int, objects_records: list,
                   want_overlay: bool):
    """Per-image: rebuild masks from classified objects -> markers."""
    import io as _io
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from micro2dfn import io, segment, config, objects as obj_, \
        markers as mk, report
    if downsample > 1:
        config.MIN_BRIGHT_PX = max(1, round(config.MIN_BRIGHT_PX
                                          / downsample ** 2))
        config.MIN_FEATURE_PX = max(1, round(config.MIN_FEATURE_PX
                                           / downsample ** 2))
    im = io.load_bse(path, batch, downsample=downsample)
    seg = segment.segment(im, t_pore, t_si)
    ob = pd.DataFrame(objects_records)
    part, fine = obj_.particle_mask(im, seg == config.SI, ob)
    f = mk.extract_markers(im, seg, ob, part, fine)
    png = None
    if want_overlay:
        fig_path = "/tmp/overlay.png"
        report.fig_classification_overlay(
            im.raw, part, fine, seg == config.PORE,
            f"{batch}/{im.image_id}", fig_path)
        with open(fig_path, "rb") as fh:
            png = fh.read()
    # serialise: numpy -> python scalars
    f = {k: (v.item() if isinstance(v, np.generic) else v)
         for k, v in f.items()}
    return {"markers": f, "overlay_png": png,
            "image_id": im.image_id}


@app.function(timeout=1800, max_containers=32)
def solve_remote(overrides: dict, n_cycles: int, sto_windows: dict,
                 np_ref: float, cathode_thickness: float):
    """One DFN run on Modal (parallel map over sweep points).

    cathode_thickness: shared cathode for this sweep index — sized on
    the reference batch's anode points, identical across batches."""
    from micro2dfn import simulate
    return simulate.solve_once(overrides, n_cycles,
                               sto_windows=sto_windows, np_ref=np_ref,
                               cathode_thickness=cathode_thickness)


@app.function(timeout=3600)
def validate_remote(b3_point: dict, n_cycles: int):
    """Reference cell + tests + ablation + constants — one call.

    Mirrors run_dfn.run_validation but returns only picklable results
    (PyBaMM solutions are dropped after windows are measured)."""
    import numpy as np
    import pandas as pd
    import pybamm
    from micro2dfn import simulate as S

    res = {"pybamm_version": pybamm.__version__, "n_cycles": n_cycles}
    ref = S.solve_reference(num_cycles=n_cycles)
    res["reference"] = {k: v for k, v in ref.items()
                        if not k.startswith("_")}
    res["sto_windows"] = S.measure_sto_windows(ref["_sol"])
    res["np_ref"] = S.np_ratio(ref["_param"], res["sto_windows"])
    res["ref_plating_flag"] = int(ref["min_neg_surface_v"] < 0)

    tests = []
    cap1 = ref.get("discharge_cap_cycle1_Ah", np.nan)
    tests.append(("reference discharge capacity within 10% of nominal "
                  "5 Ah", np.isfinite(cap1)
                  and abs(cap1 - 5.0) / 5.0 <= 0.10, f"{cap1:.3f} Ah"))
    cyc = [v for k, v in sorted(ref.items())
           if k.startswith("discharge_cap_cycle")]
    if len(cyc) >= 3:
        drift = abs(cyc[-1] - cyc[0]) / cyc[0] * 100
        tests.append((f"reference drift < 0.5% over {len(cyc)} cycles",
                      drift < 0.5, f"{drift:.3f}%"))
    d_step = S.discharge_capacity_ah(ref["_sol"].cycles[0])
    d_int = S.discharge_capacity_integral(ref["_sol"].cycles[0])
    tests.append(("per-cycle discharge capacity == integral of "
                  "discharge current",
                  abs(d_step - d_int) / max(d_int, 1e-9) < 0.01,
                  f"{d_step:.4f} vs {d_int:.4f} Ah"))
    p = S.base_parameters()
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
    om = float(p["Secondary: Negative electrode partial molar volume "
                 "[m3.mol-1]"])
    cm = float(p["Secondary: Maximum concentration in negative "
                 "electrode [mol.m-3]"])
    v1 = om * cm
    tests.append(("Si volume-change function ~+300% at full "
                  "lithiation", 2.0 < v1 < 4.0, f"+{v1 * 100:.0f}%"))

    b3 = S.solve_once(b3_point, num_cycles=n_cycles,
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
    res["tests"] = [(n, bool(ok), note) for n, ok, note in tests]

    # ablation: reference / geometry / shipped-buggy / full-fixed
    def grab(label, o):
        r = {"stage": label}
        for k in ("discharge_cap_cycle1_Ah", "discharge_cap_last_Ah",
                  "retention_pct", "loss_li_inventory_pct",
                  "min_neg_surface_v", "np_ratio", "converged",
                  "cycles_completed"):
            r[k] = o.get(k, np.nan)
        return r

    rows = [grab("0_reference", ref)]
    rows.append(grab("1_geometry",
                     S.solve_once(b3_point, n_cycles, degradation=False,
                                  sto_windows=res["sto_windows"],
                                  np_ref=res["np_ref"])))
    rows.append(grab("2_degradation_shipped",
                     S.solve_once(b3_point, n_cycles,
                                  param=S._legacy_buggy_parameters(),
                                  sto_windows=res["sto_windows"],
                                  np_ref=res["np_ref"])))
    rows.append(grab("3_full_fixed",
                     S.solve_once(b3_point, n_cycles,
                                  param=S.base_parameters(),
                                  sto_windows=res["sto_windows"],
                                  np_ref=res["np_ref"])))
    ab = pd.DataFrame(rows)
    ab["note"] = [
        "unmodified Chen2020_composite, no degradation",
        "our B3 geometry, cathode rescaled to reference N/P, "
        "no degradation",
        "BEFORE: graphite params copied onto Si, LAM 1e-3 s-1, "
        "hand-set SEI",
        "AFTER: Si keeps Si chemistry/mechanics (Bonkile2024), "
        "published LAM/SEI — the full model",
    ]
    res["ablation"] = ab.to_dict("records")
    res["constants"] = S.constants_table().to_dict("records")
    return res


# --------------------------------------------------------------------------
# local driver
# --------------------------------------------------------------------------
@app.local_entrypoint()
def main(out: str = "dfn_output", fast: bool = False,
         no_sim: bool = False, baseline: str = "Batch_3"):
    import io as _io
    import time

    import numpy as np
    import pandas as pd

    import run_dfn as R
    import micro2dfn.config as config
    from micro2dfn import io, objects as obj_, report, dfn_inputs, \
        segment
    from micro2dfn import simulate as S

    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figs"), exist_ok=True)

    batch_paths = io.discover_batches(ROOT)
    remote_paths = {b: [p.replace(ROOT, DATA_ROOT) for p in paths]
                    for b, paths in batch_paths.items()}
    print("Batches:", {b: len(p) for b, p in batch_paths.items()})

    downsample = R.DOWNSAMPLE_FAST if fast else 1
    recipe_path = os.path.join(out, "recipe.json")
    if os.path.exists(recipe_path):
        import json
        with open(recipe_path) as fh:
            rec = json.load(fh)
        t_pore, t_si, t_core = rec["t_pore"], rec["t_si"], rec["t_core"]
        print(f"frozen recipe loaded from {recipe_path} "
              f"(fitted on {rec['fitted_on']}, {rec['n_fit_images']} "
              f"images, {rec['created_utc']})")
        rec_frozen = True
    else:
        # fit on the BASELINE batch only — frozen thereafter; a new
        # batch can never move the ruler applied to existing batches
        ref = baseline if baseline in remote_paths \
            else sorted(remote_paths)[-1]
        print(f"Fitting recipe on baseline {ref} "
              f"({len(remote_paths[ref])} images) on Modal — frozen "
              "thereafter...")
        t_pore, t_si = fit_thresholds_remote.remote(
            {ref: remote_paths[ref]}, downsample)
        t_core = None
        rec_frozen = False
    segment.save_thresholds(
        os.path.join(out, "thresholds.json"), t_pore, t_si,
        sum(len(p) for p in batch_paths.values()))

    print("Extracting objects on Modal (per image, parallel)...")
    jobs = [(p, b, t_pore, t_si, downsample)
            for b, paths in remote_paths.items() for p in paths]
    obj_frames = [pd.DataFrame(recs) for recs in
                  objects_remote.starmap(jobs)]
    objects = pd.concat(obj_frames, ignore_index=True)
    if rec_frozen is False:
        # calibrate the object classifier on baseline objects only
        t_core = obj_.calibrate_t_core(
            objects[objects.batch == ref])
        import json
        rec = {"recipe_version": "micro2dfn-2.0", "fitted_on": ref,
               "n_fit_images": len(remote_paths[ref]),
               "t_pore": float(t_pore), "t_si": float(t_si),
               "t_core": float(t_core),
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
               "fit_image_md5": {
                   os.path.basename(p).replace("_BSE.tif", ""):
                       R._file_md5(p.replace(DATA_ROOT, ROOT))
                   for p in remote_paths[ref]}}
        with open(recipe_path, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"  recipe saved -> {recipe_path}")
    objects = obj_.classify_objects(objects, t_core)
    print(f"  {len(objects)} objects; t_core={t_core:.3f} "
          "(baseline-fitted)")

    # run manifest: which recipe/code/images produced this run
    import json
    code_files = sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "micro2dfn", "*.py"))
        + [os.path.abspath(__file__), os.path.abspath("run_dfn.py")])
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime()),
        "baseline_batch": baseline, "downsample": downsample,
        "fast": fast, "compute": "modal",
        "recipe_path": recipe_path,
        "recipe_md5": R._file_md5(recipe_path),
        "recipe_fitted_on": rec.get("fitted_on"),
        "code_md5": {os.path.basename(f): R._file_md5(f)
                     for f in code_files if os.path.exists(f)},
        "images": {b: {os.path.basename(p).replace("_BSE.tif", ""):
                       R._file_md5(p.replace(DATA_ROOT, ROOT))
                       for p in paths}
                   for b, paths in batch_paths.items()},
    }
    manifest_path = os.path.join(out, "run_manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"  run manifest -> {manifest_path}")

    print("Computing markers on Modal (per image, parallel)...")
    img_ids = [os.path.basename(p).replace("_BSE.tif", "")
               for b, paths in batch_paths.items() for p in paths]
    jobs2 = []
    i = 0
    for b, paths in remote_paths.items():
        for p in paths:
            iid = img_ids[i]
            i += 1
            sub = objects[objects.image_id == iid]
            jobs2.append((p, b, t_pore, t_si, downsample,
                          sub.to_dict("records"),
                          i <= R.N_DEBUG_FIGS))
    mk_rows = []
    for res in markers_remote.starmap(jobs2):
        mk_rows.append(res["markers"])
        if res["overlay_png"]:
            with open(os.path.join(
                    out, "figs",
                    f"seg_{res['markers']['batch']}_"
                    f"{res['image_id']}.png"), "wb") as fh:
                fh.write(res["overlay_png"])
    markers = pd.DataFrame(mk_rows)
    cut = markers["si_bulk_contrast"].quantile(0.05)
    markers["is_problem_photo"] = (
        markers["si_bulk_contrast"] < cut).astype(int)

    markers.to_csv(os.path.join(out, "markers_all.csv"), index=False)
    objects.to_csv(os.path.join(out, "objects_all.csv"), index=False)
    for b in sorted(batch_paths):
        markers[markers.batch == b].to_csv(
            os.path.join(out, f"markers_{b}.csv"), index=False)
        objects[objects.batch == b].to_csv(
            os.path.join(out, f"objects_{b}.csv"), index=False)
    print(f"Markers: {len(markers)} images, {len(objects)} objects, "
          f"{int(markers['is_problem_photo'].sum())} problem photos")

    summary = report.batch_summary(markers)
    deltas = report.batch_deltas(markers)
    summary.to_csv(os.path.join(out, "batch_summary.csv"),
                   index=False)
    deltas.to_csv(os.path.join(out, "batch_differences.csv"),
                  index=False)
    report.write_data_dictionary(markers, out)
    report.fig_marker_distributions(markers, out)

    sweep_all, dfn_summary, paired, validation = None, None, None, None
    settings = {
        "recipe": recipe_path,
        "recipe_fitted_on": rec.get("fitted_on"),
        "recipe_md5": R._file_md5(recipe_path)
        if os.path.exists(recipe_path) else None,
        "run_manifest": manifest_path,
    }
    if not no_sim:
        n_cycles = R.DFN_CYCLES_FAST if fast else R.DFN_CYCLES
        n_pts = R.DFN_SWEEP_FAST if fast else R.DFN_SWEEP_POINTS
        import pybamm
        settings.update({
            "pybamm_version": pybamm.__version__,
            "downsample": downsample,
            "cycles": n_cycles,
            "sweep_points_per_variant": n_pts + 1,
            "variants": ", ".join(R.VARIANTS),
            "si_bracket": "lo = classified Si x accessible share; "
                          "hi = all uncertain bright counted",
            "lam_proportional_s^-1": 2.7778e-7,
            "np_hold": "one shared cathode per sweep index, sized on "
                      "the reference batch's anode points (same for all "
                      "batches at that index)",
            "seed": 0,
            "compute": "modal (parallel)",
        })

        ref_batch = "Batch_3" if "Batch_3" in set(markers.batch) \
            else sorted(batch_paths)[-1]
        bands_ref = dfn_inputs.batch_bands(
            markers[markers.batch == ref_batch])
        b3_point = dfn_inputs.to_pybamm_params(
            dfn_inputs.sweep_points(bands_ref, n_points=0)
            .iloc[0].to_dict())

        print("Validation on Modal...")
        validation = validate_remote.remote(b3_point, n_cycles)
        validation["tests"] = pd.DataFrame(
            validation["tests"], columns=["test", "pass", "detail"])
        validation["ablation"] = pd.DataFrame(validation["ablation"])
        validation["constants"] = pd.DataFrame(validation["constants"])
        sto_windows, np_ref = validation["sto_windows"], \
            validation["np_ref"]
        settings["ref_min_neg_v"] = float(
            validation["reference"].get("min_neg_surface_v",
                                        float("nan")))
        n_pass = int(validation["tests"]["pass"].sum())
        print(f"  validation: {n_pass}/{len(validation['tests'])} "
              f"tests pass; ref N/P {np_ref:.3f}")

        print("DFN sweeps on Modal (paired same-seed LHS; shared "
              "cathode per sweep index)...")
        solve_jobs, job_meta = [], []
        frames_pts = {}
        base_param = S.base_parameters()
        for variant in R.VARIANTS:
            for batch in sorted(batch_paths):
                f = markers[markers.batch == batch]
                bands = dfn_inputs.batch_bands(
                    f,
                    accessible=1.0 if variant == "all_si_active"
                    else None,
                    exclude_problem=variant == "no_problem_photos")
                frames_pts[(variant, batch)] = \
                    dfn_inputs.sweep_points(bands, n_points=n_pts)
            # same cathode at each sweep index for every batch —
            # sized on the reference batch's anode points
            schedule = S.shared_cathode_schedule(
                frames_pts[(variant, ref_batch)], base_param,
                sto_windows, np_ref)
            for batch in sorted(batch_paths):
                for i, (_, point) in enumerate(
                        frames_pts[(variant, batch)].iterrows()):
                    solve_jobs.append(
                        (dfn_inputs.to_pybamm_params(point.to_dict()),
                         n_cycles, sto_windows, np_ref, schedule[i]))
                    job_meta.append((variant, batch,
                                     point.to_dict()))
        print(f"  {len(solve_jobs)} solves across "
              f"{len(R.VARIANTS)} variants x {len(batch_paths)} batches")
        results = list(solve_remote.starmap(solve_jobs))
        frames = {}
        for (variant, batch, point), res in zip(job_meta, results):
            frames.setdefault((variant, batch), []).append(
                {**point, **res})
        for (variant, batch), rows in frames.items():
            df = pd.DataFrame(rows)
            df["batch"], df["variant"] = batch, variant
            frames[(variant, batch)] = df
            df.to_csv(os.path.join(
                out, f"dfn_sweep_{variant}_{batch}.csv"), index=False)
        sweep_all = pd.concat(frames.values(), ignore_index=True)
        sweep_all.to_csv(os.path.join(out, "dfn_results.csv"),
                         index=False)

        main_sw = {b: frames[("accessible", b)]
                   for b in sorted(batch_paths)}
        dfn_summary = pd.DataFrame([
            {"batch": b, **S.summarise_sweep(df)}
            for b, df in main_sw.items()])
        dfn_summary.to_csv(
            os.path.join(out, "dfn_indicator_summary.csv"),
            index=False)
        paired = S.paired_differences(
            main_sw,
            ref_min_neg=validation["reference"].get(
                "min_neg_surface_v"))
        paired.to_csv(
            os.path.join(out, "dfn_paired_differences.csv"),
            index=False)

        for b in sorted(batch_paths):
            frames_pts[("accessible", b)].to_json(
                os.path.join(out, f"dfn_params_{b}.json"),
                orient="records", indent=1)
        bands_all = pd.concat([
            dfn_inputs.batch_bands(markers[markers.batch == b])
            .assign(batch=b) for b in sorted(batch_paths)])
        bands_all.to_csv(os.path.join(out, "dfn_input_bands.csv"))
        at = pd.concat([
            dfn_inputs.run_assumption_table(
                bands_all[bands_all.batch == b].drop(columns="batch"),
                frames_pts[("accessible", b)])
            .assign(batch=b) for b in sorted(batch_paths)])
        at.to_csv(os.path.join(out, "assumption_table.csv"),
                  index=False)

        report.fig_consequence_bands(dfn_summary, out)
        tor = report.fig_tornado(
            sweep_all[sweep_all["variant"] == "accessible"], out)
        if tor:
            print(f"  wrote {tor}")
        R.write_validation_md(validation, out, settings)

    report.write_report(out, markers, summary, deltas, dfn_summary,
                        t_pore, t_si, t_core,
                        settings=settings if settings else None,
                        paired=paired)
    report.write_readme(out)
    xlsx = report.write_excel(out, markers, summary, deltas,
                              sweep_all, dfn_summary)
    print(f"\nExcel workbook -> {xlsx}")
    print(f"Done in {time.time() - t0:.0f}s -> {out}/report.md")
