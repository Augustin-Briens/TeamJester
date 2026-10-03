"""
QC pipeline on REAL images, run on Modal.

    modal run qc_real.py --baseline C:\\data\\batch_1 --batch C:\\data\\batch_2   (uses *_BSE files; --detector Inlens to change)

Each folder holds the images of one batch (.tif/.tiff/.png/.jpg).
Flow: load -> segment -> compute_kpis  (one container per image)  -> judge_batch -> report
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/tldr-group/ImageRep /root/ImageRep",
        "cd /root/ImageRep && pip install -e .",
    )
)
app = modal.App("qc-real", image=image)

# ---------------------------------------------------------------- settings (team decisions)
CROP_BOTTOM = 0.0       # fraction of image height removed (set >0 if there is a banner)
SIGMA = 1.5             # smoothing before thresholding
MIN_BRIGHT_PX = 150     # ignore bright specks smaller than this
MIN_FEATURE_PX = 20     # ignore dark specks smaller than this (noise)
BIG_VOID_FRAC = 0.005   # a dark region bigger than this share of the image = gap between particles
CRACK_ASPECT = 3.0      # long/short axis above this = crack, below = round pore
TOLERANCE = 1.5         # allowed shift in baseline SDs  (FREEZE before the new batch)
N_BOOT = 2000

KPI_LABELS = {
    "porosity": "total dark fraction",
    "big_void_frac": "gaps between particles (area frac)",
    "crack_frac": "crack area fraction",
    "round_pore_frac": "round pore area fraction",
    "cracks_per_mpx": "cracks per million px",
    "pores_per_mpx": "round pores per million px",
    "mean_pore_diam_px": "mean round pore diameter (px)",
    "interface_density": "pore-solid interface per unit area",
    "bright_frac": "bright particle area fraction",
    "bright_per_mpx": "bright particles per million px",
    "mean_bright_diam_px": "mean bright particle diameter (px)",
}


# ---------------------------------------------------------------- per-image steps
def load_image(data: bytes):
    import io
    import numpy as np
    import imageio.v3 as iio

    img = np.asarray(iio.imread(io.BytesIO(data)))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(float)


def segment(img):
    """Greyscale BSE image -> labels: 0 = pore/crack, 1 = grey bulk, 2 = bright particles."""
    import numpy as np
    from skimage import filters, morphology

    if CROP_BOTTOM:
        img = img[: int(img.shape[0] * (1 - CROP_BOTTOM))]
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-12), 0, 1)
    smooth = filters.gaussian(img, sigma=SIGMA)
    t_dark, t_bright = filters.threshold_multiotsu(smooth, classes=3)
    bright = smooth > t_bright
    bright = morphology.binary_opening(bright, morphology.disk(2))          # drop speckle
    bright = morphology.remove_small_objects(bright, MIN_BRIGHT_PX)
    labels = np.ones(smooth.shape, dtype=np.uint8)
    labels[smooth < t_dark] = 0
    labels[bright] = 2
    return labels


def compute_kpis(labels):
    import numpy as np
    from skimage import measure

    binary = (labels > 0).astype(np.uint8)
    n_px = binary.size
    pore = labels == 0
    kpis = {"porosity": float(pore.mean())}

    big = crack = rnd = 0
    n_crack = n_round = 0
    diams = []
    for r in measure.regionprops(measure.label(pore, connectivity=2)):
        if r.area < MIN_FEATURE_PX:
            continue
        if r.area > BIG_VOID_FRAC * n_px:
            big += r.area
        elif r.axis_major_length / max(r.axis_minor_length, 1.0) > CRACK_ASPECT:
            crack += r.area; n_crack += 1
        else:
            rnd += r.area; n_round += 1
            diams.append(r.equivalent_diameter_area)

    kpis["big_void_frac"] = big / n_px
    kpis["crack_frac"] = crack / n_px
    kpis["round_pore_frac"] = rnd / n_px
    kpis["cracks_per_mpx"] = n_crack / n_px * 1e6
    kpis["pores_per_mpx"] = n_round / n_px * 1e6
    kpis["mean_pore_diam_px"] = float(np.mean(diams)) if diams else 0.0
    bright = labels == 2
    b_regions = measure.regionprops(measure.label(bright, connectivity=2))
    kpis["bright_frac"] = float(bright.mean())
    kpis["bright_per_mpx"] = len(b_regions) / n_px * 1e6
    kpis["mean_bright_diam_px"] = float(np.mean([r.equivalent_diameter_area for r in b_regions])) if b_regions else 0.0
    horiz = (binary[:, 1:] != binary[:, :-1]).mean()
    vert = (binary[1:, :] != binary[:-1, :]).mean()
    kpis["interface_density"] = float((horiz + vert) / 2)

    try:  # ImageRep: how well does ONE image pin down the dark fraction?
        from representativity import core
        res = core.make_error_prediction(pore.astype(np.uint8), confidence=0.95)
        res = res if isinstance(res, dict) else vars(res)
        kpis["imagerep_abs_err"] = float(res["abs_err"])
    except Exception:
        kpis["imagerep_abs_err"] = None
    return kpis


# ---------------------------------------------------------------- batch comparison
def judge_batch(baseline_rows, batch_rows, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    rank = {"accept": 0, "investigate": 1, "reject": 2}
    per_kpi = []
    for kpi in KPI_LABELS:
        base = np.array([r[kpi] for r in baseline_rows], dtype=float)
        new = np.array([r[kpi] for r in batch_rows], dtype=float)
        sd = base.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            continue                       # KPI never varies in baseline: cannot scale it
        shift = (new.mean() - base.mean()) / sd
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            boots[i] = (rng.choice(new, new.size).mean() - rng.choice(base, base.size).mean()) / sd
        lo, hi = np.percentile(boots, [2.5, 97.5])
        if lo > -TOLERANCE and hi < TOLERANCE:
            verdict = "accept"
        elif lo > TOLERANCE or hi < -TOLERANCE:
            verdict = "reject"
        else:
            verdict = "investigate"
        pct = (new.mean() - base.mean()) / base.mean() * 100 if base.mean() else float("nan")
        per_kpi.append({
            "kpi": kpi, "verdict": verdict, "baseline_mean": float(base.mean()),
            "batch_mean": float(new.mean()), "pct_change": float(pct),
            "shift_sd": float(shift), "ci_low": float(lo), "ci_high": float(hi),
        })
    overall = max(per_kpi, key=lambda k: rank[k["verdict"]])["verdict"]
    errs = [r["imagerep_abs_err"] for r in batch_rows if r.get("imagerep_abs_err")]
    return {
        "verdict": overall,
        "per_kpi": sorted(per_kpi, key=lambda k: -abs(k["shift_sd"])),
        "imagerep_single_image_err": float(np.mean(errs)) if errs else None,
    }


# ---------------------------------------------------------------- Modal functions
@app.function(timeout=900)
def measure(job):
    try:
        return {"file": job["file"], **compute_kpis(segment(load_image(job["data"])))}
    except Exception as e:
        return {"file": job["file"], "error": repr(e)}


@app.function(timeout=600)
def judge(baseline_rows, batch_rows):
    return judge_batch(baseline_rows, batch_rows)


def _jobs(folder, detector):
    from pathlib import Path
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted(
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in exts and f.stem.lower().endswith("_" + detector.lower())
    )
    return [{"file": f.name, "data": f.read_bytes()} for f in files]


@app.local_entrypoint()
def main(baseline: str, batch: str, detector: str = "BSE"):
    import json

    base_jobs, batch_jobs = _jobs(baseline, detector), _jobs(batch, detector)
    print(f"baseline: {len(base_jobs)} images | batch: {len(batch_jobs)} images")
    rows = list(measure.map(base_jobs + batch_jobs))
    for r in rows:
        if "error" in r:
            print(f"  SKIPPED {r['file']}: {r['error']}")
    base_rows = [r for r in rows[: len(base_jobs)] if "error" not in r]
    batch_rows = [r for r in rows[len(base_jobs):] if "error" not in r]

    report = judge.remote(base_rows, batch_rows)

    print("\n" + "=" * 96)
    print(f"VERDICT: {report['verdict'].upper()}   ({len(batch_rows)} images vs {len(base_rows)} baseline)")
    print(f"  {'KPI':<36}{'baseline':>10}{'batch':>10}{'change':>9}{'shift(SD)':>11}   95% interval      verdict")
    for k in report["per_kpi"]:
        print(
            f"  {KPI_LABELS[k['kpi']]:<36}{k['baseline_mean']:>10.4g}{k['batch_mean']:>10.4g}"
            f"{k['pct_change']:>+8.0f}%{k['shift_sd']:>+11.2f}   [{k['ci_low']:+.2f}, {k['ci_high']:+.2f}]   {k['verdict']}"
        )
    top = report["per_kpi"][0]
    if report["verdict"] == "accept":
        print(f"  Why: every KPI stays within {TOLERANCE} baseline SDs, even at the edge of its interval.")
    else:
        print(f"  Why: '{KPI_LABELS[top['kpi']]}' moved {top['shift_sd']:+.1f} baseline SDs ({top['pct_change']:+.0f}%).")
    if len(batch_rows) < 8:
        print(f"  Caution: only {len(batch_rows)} images in this batch; intervals are wide.")
    if report["imagerep_single_image_err"]:
        print(f"  ImageRep: one image pins the dark fraction to +/-{report['imagerep_single_image_err']:.3f}.")

    from pathlib import Path
    out = f"{Path(batch).name}_qc.json"
    with open(out, "w") as f:
        json.dump({"report": report, "baseline_rows": base_rows, "batch_rows": batch_rows}, f, indent=2)
    print("saved", out)