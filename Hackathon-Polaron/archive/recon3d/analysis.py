"""Analysis stage shared by run_recon3d.py (local) and
modal_recon3d.py (remote-trained volumes): transport solves,
validation, figures, report."""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

from . import config, report, transport, validate


def load_volume_sets(out: str, batches: list[str], V: int, iters: int,
                     ruler: str | None = None
                     ) -> dict[str, np.ndarray]:
    """All generated volumes per batch for THIS config + ruler,
    stacked.  The ruler tag in the filename prevents mixing ensembles
    built under different measuring rulers."""
    suffix = f"__r{ruler}" if ruler else ""
    vols = {}
    for b in batches:
        vs = []
        for f in sorted(glob.glob(os.path.join(
                out, "volumes", f"{b}__s*__V{V}i{iters}{suffix}.npz"))):
            v, _meta = _load_npz(f)
            vs.append(v)
        if vs:
            vols[b] = np.concatenate(vs)
    return vols


def _load_npz(path: str):
    z = np.load(path, allow_pickle=True)
    return z["volumes"], (dict(z["meta"][0]) if "meta" in z else {})


def heldout_crops(masks: dict[str, np.ndarray], V: int, n: int,
                  seed: int = 0) -> np.ndarray:
    """n random V x V crops pooled across a batch's masks."""
    rng = np.random.default_rng(seed)
    imgs = [m for m in masks.values()
            if m.shape[0] >= V and m.shape[1] >= V]
    out = np.empty((n, V, V), np.uint8)
    for k in range(n):
        m = imgs[rng.integers(0, len(imgs))]
        y = rng.integers(0, m.shape[0] - V + 1)
        x = rng.integers(0, m.shape[1] - V + 1)
        out[k] = m[y:y + V, x:x + V]
    return out


_CROSS2D = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def acquisition_table(masks: dict[str, dict[str, np.ndarray]],
                      out: str, print_=print) -> pd.DataFrame | None:
    """Per-image mask morphology joined to acquisition groups.

    The 2-D images come from ~13 distinguishable imaging sessions and
    count/fragmentation statistics are session-dominated (see the
    paper's acquisition analysis).  recon3d trains one GAN per BATCH,
    so each model learns its batch's session mix — batch differences
    in the generated ensembles can be session differences.  This table
    is the pipeline's own control for that confound.
    """
    from scipy.ndimage import label
    acq_path = "image_acquisition.csv"
    if not os.path.exists(acq_path):
        print_("  acquisition_control: image_acquisition.csv not "
               "found — skipped")
        return None
    rows = []
    for b, imgs in masks.items():
        for iid, m in imgs.items():
            pore = m == 0
            lab, n = label(pore, structure=_CROSS2D)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            ch_z = validate.chord_lengths(pore, 0)
            ch_ip = validate.chord_lengths(pore, 1)
            idx = np.arange(1, len(ch_z) + 1)
            rows.append(dict(
                batch=b, image_id=iid,
                pore_frac=float(pore.mean()),
                bulk_frac=float((m == 1).mean()),
                bright_frac=float((m == 2).mean()),
                pore_comp_per_1000vox=float(n / m.size * 1000),
                largest_comp_share=float(
                    sizes.max() / max(pore.sum(), 1)),
                pore_chord_z=float(np.average(idx, weights=ch_z)),
                pore_chord_ip=float(np.average(idx, weights=ch_ip))))
    df = pd.DataFrame(rows)
    acq = pd.read_csv(acq_path)
    df = df.merge(acq[["image_id", "group_id", "noise_mad",
                       "si_bulk_contrast", "sharpness_lapvar"]],
                  on="image_id", how="left")
    df.to_csv(os.path.join(out, "acquisition_control.csv"),
              index=False)
    return df


def _variance_partition(df: pd.DataFrame, col: str) -> dict:
    """Variance left after removing group means vs batch means —
    lower residual = the variable explains more."""
    d = df.dropna(subset=["group_id"])
    tot = d[col].var()
    if not tot or tot <= 0:
        return dict(metric=col, resid_after_group=np.nan,
                    resid_after_batch=np.nan)
    rg = d.groupby("group_id")[col].transform(
        lambda s: s - s.mean()).var() / tot
    rb = d.groupby("batch")[col].transform(
        lambda s: s - s.mean()).var() / tot
    return dict(metric=col, resid_after_group=float(rg),
                resid_after_batch=float(rb))


def _mixed_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Per-group rows for sessions containing >1 batch."""
    n_b = df.groupby("group_id").batch.nunique()
    mixed = n_b[n_b > 1].index
    keep = ["group_id", "batch", "image_id", "pore_frac",
            "pore_comp_per_1000vox", "largest_comp_share",
            "pore_chord_ip"]
    return (df[df.group_id.isin(mixed)][keep]
            .sort_values(["group_id", "batch"]))


def run_analysis(out: str, batches: list[str],
                 masks: dict[str, dict[str, np.ndarray]],
                 settings: dict, print_=print) -> None:
    V = settings.get("V", config.VOLUME)
    vox_um = settings.get("voxel_nm", config.VOXEL_NM) / 1000.0
    figdir = os.path.join(out, "figs")
    os.makedirs(figdir, exist_ok=True)
    vols = load_volume_sets(out, batches, V,
                            settings.get("iters", config.ITERS),
                            settings.get("ruler"))
    if not vols:
        print_("no volumes found — train first")
        return

    # --- transport solves -------------------------------------------------
    rows = []
    t0 = time.time()
    for b, vs in vols.items():
        for k, v in enumerate(vs):
            m = transport.volume_metrics(v, voxel_nm=vox_um * 1000.0)
            m.update(batch=b, realization=k)
            rows.append(m)
            print_(f"  solve {b} vol {k + 1}/{len(vs)}: "
                   f"tau_tp={m['tau_tp']:.3g} spans={m['spans_tp']}"
                   f"  ({time.time() - t0:.0f}s)", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out, "transport_metrics.csv"),
                   index=False)

    # --- validation -------------------------------------------------------
    vrows, verdicts, real_crops = [], {}, {}
    for b in vols:
        real_crops[b] = heldout_crops(masks[b], V, config.N_VAL_REAL)
        v = validate.validate_batch(real_crops[b], vols[b], vox_um)
        v["batch"] = b
        verdicts[b] = validate.summary_verdict(v)
        vrows.append(v)
        print_(f"  validation {b}: {verdicts[b].split('—')[0].strip()}")
    val = pd.DataFrame(vrows)
    val.to_csv(os.path.join(out, "validation.csv"), index=False)

    # --- acquisition control ----------------------------------------------
    acq = acquisition_table(masks, out, print_)
    acq_info = None
    if acq is not None:
        acq_info = dict(
            partition=pd.DataFrame(
                [_variance_partition(acq, c) for c in
                 ("pore_frac", "pore_comp_per_1000vox",
                  "largest_comp_share", "pore_chord_ip")]),
            mixed=_mixed_groups(acq),
            n_groups=int(acq.group_id.nunique()),
            groups_per_batch=acq.groupby("batch")
            .group_id.nunique().to_dict())

    # --- figures ----------------------------------------------------------
    for b in vols:
        report.fig_slices(real_crops[b], vols[b], b,
                          os.path.join(figdir, f"slices_{b}.png"))
        report.fig_volume(vols[b], b,
                          os.path.join(figdir, f"volume_{b}.png"),
                          voxel_nm=vox_um * 1000.0)
        report.fig_s2(real_crops[b], vols[b], b,
                      os.path.join(figdir, f"s2_{b}.png"))
        report.fig_chords(real_crops[b], vols[b], b,
                          os.path.join(figdir, f"chords_{b}.png"))
    # flux-solve figure on the first SPANNING volume of the reference
    # batch (non-spanning volumes return early without a field)
    ref = "Batch_3" if "Batch_3" in vols else sorted(vols)[-1]
    for v in vols[ref]:
        tv = transport.tortuosity(v, phase=0, axis=0,
                                  return_field=True)
        if "field" in tv:
            report.fig_flux(v, tv["field"], tv["cluster"],
                            ref, os.path.join(figdir, "flux_solve.png"))
            break
    report.fig_transport(metrics,
                         os.path.join(figdir, "transport.png"))

    # --- training diagnostics --------------------------------------------
    losses = []
    for f in sorted(glob.glob(os.path.join(out, "models", "*.csv"))):
        h = pd.read_csv(f)
        losses.append(dict(
            model=os.path.basename(f)[:-4], iters=int(h["iter"].max()),
            final_w=float(h["w"].iloc[-1]),
            seconds=float(h["seconds"].iloc[0])
            if "seconds" in h else np.nan))
    losses_df = pd.DataFrame(losses) if losses else None

    path = report.write_report(out, metrics, val, verdicts,
                               settings, losses_df, acq_info)
    print_(f"report -> {path}")
