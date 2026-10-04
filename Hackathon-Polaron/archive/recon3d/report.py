"""Figures + markdown report for the recon3d branch."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import config, validate

CMAP = ListedColormap(["#1f4e9e", "#8c8c8c", "#d62728"])  # pore,bulk,bright


def fig_slices(real: np.ndarray, vols: np.ndarray, batch: str,
               path: str, seed: int = 0) -> None:
    """Real crops vs generated z-slices — visual plausibility check."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    for j in range(4):
        ax[0, j].imshow(real[rng.integers(0, len(real))], cmap=CMAP,
                        vmin=0, vmax=2, interpolation="nearest")
        v = vols[rng.integers(0, len(vols))]
        a, i = rng.choice((1, 2)), rng.integers(0, v.shape[1])
        sl = v[:, i, :] if a == 1 else v[:, :, i]
        ax[1, j].imshow(sl, cmap=CMAP, vmin=0, vmax=2,
                        interpolation="nearest")
    ax[0, 0].set_ylabel("real 2-D crops")
    ax[1, 0].set_ylabel("generated z-slices")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{batch}: training data vs SliceGAN output  "
                 f"(red=bright/Si-family, blue=pore, grey=bulk)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_volume(vols: np.ndarray, batch: str, path: str,
               voxel_nm: float = config.VOXEL_NM) -> None:
    """Orthogonal centre slices + a downsampled 3-D voxel view."""
    v = vols[0]
    V = v.shape[0]
    c = V // 2
    fig = plt.figure(figsize=(14, 4.5))
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(v[c], cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax1.set_title("in-plane (y,x) — unconstrained")
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(v[:, c, :], cmap=CMAP, vmin=0, vmax=2,
               interpolation="nearest")
    ax2.set_title("(z,x) — trained")
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(v[:, :, c], cmap=CMAP, vmin=0, vmax=2,
               interpolation="nearest")
    ax3.set_title("(z,y) — trained")
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    g = 4
    ax4.voxels(v[::g, ::g, ::g] == 0, facecolors="#1f4e9e", alpha=0.5,
               edgecolor="none")
    ax4.set_title("pore phase, 3-D")
    for a in (ax1, ax2, ax3):
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{batch}: one realisation ({V} vox = "
                 f"{V * voxel_nm / 1000:.1f} um edge)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_s2(real: np.ndarray, vols: np.ndarray, batch: str,
           path: str, seed: int = 0) -> None:
    """Two-point correlation curves per phase, real vs generated."""
    fake = validate.generated_slices(vols, 24, seed)
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(24, len(real)),
                           replace=False)]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for k, (name, ph) in enumerate(validate.PHASES.items()):
        for sl, lab in ((real, "real"), (fake, "generated")):
            s2 = np.mean([validate.two_point_corr(s == ph)
                          for s in sl], axis=0)
            ax[k].plot(s2, label=lab, lw=1.6)
        ax[k].set_title(f"{name} S2(r)")
        ax[k].set_xlabel("r (vox)")
        ax[k].legend(fontsize=8)
    fig.suptitle(f"{batch}: two-point correlation, real vs generated")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_transport(metrics: pd.DataFrame, path: str) -> None:
    """Ensemble distributions per batch — the headline figure."""
    rows = [("tau_tp", "tortuosity, through-plane"),
            ("tau_ip", "tortuosity, in-plane"),
            ("tau_anisotropy", "tau anisotropy (tp/ip)"),
            ("spanning_frac_tp", "spanning pore fraction"),
            ("si_accessible_frac", "bright/Si accessible to network")]
    fig, ax = plt.subplots(1, len(rows), figsize=(17, 4.2))
    batches = sorted(metrics.batch.unique())
    for k, (col, title) in enumerate(rows):
        data = [metrics[metrics.batch == b][col].replace(
            np.inf, np.nan).dropna() for b in batches]
        ax[k].boxplot(data, tick_labels=batches, showfliers=False)
        for j, d in enumerate(data):
            ax[k].scatter(np.full(len(d), j + 1) +
                          np.linspace(-0.12, 0.12, len(d)), d,
                          s=14, alpha=0.7)
        ax[k].set_title(title, fontsize=9)
    n_inf = int(np.isinf(metrics.tau_tp).sum())
    cap = f" (n={n_inf} non-spanning omitted)" if n_inf else ""
    fig.suptitle("SliceGAN ensemble transport metrics" + cap)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_flux(vol: np.ndarray, field: np.ndarray, cluster: np.ndarray,
             batch: str, path: str) -> None:
    """Mid-plane concentration field + spanning cluster of one solve."""
    c = vol.shape[0] // 2
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
    ax[0].imshow(vol[c], cmap=CMAP, vmin=0, vmax=2,
                 interpolation="nearest")
    ax[0].set_title("label volume (y,x mid-slice)")
    ax[1].imshow(cluster[c], cmap="gray", interpolation="nearest")
    ax[1].set_title("spanning pore cluster")
    im = ax[2].imshow(field[c], cmap="viridis", vmin=0, vmax=1,
                      interpolation="nearest")
    ax[2].set_title("concentration c (solve domain)")
    fig.colorbar(im, ax=ax[2], shrink=0.8)
    fig.suptitle(f"{batch}: steady-state diffusion solve, "
                 "through-plane")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_chords(real: np.ndarray, vols: np.ndarray, batch: str,
               path: str, seed: int = 0) -> None:
    """Pore chord-length histograms, real vs generated."""
    fake = validate.generated_slices(vols, 24, seed)
    rng = np.random.default_rng(seed)
    real = real[rng.choice(len(real), min(24, len(real)),
                           replace=False)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, axn in enumerate((0, 1)):
        hr = np.mean([validate.chord_lengths(s == 0, axn)
                      for s in real], axis=0)
        hf = np.mean([validate.chord_lengths(s == 0, axn)
                      for s in fake], axis=0)
        ax[k].plot(hr, label="real", lw=1.6)
        ax[k].plot(hf, label="generated", lw=1.6, ls="--")
        ax[k].set_xlim(0, 64)
        ax[k].set_title("pore chords along z" if axn == 0
                        else "pore chords in-plane")
        ax[k].set_xlabel("chord length (vox)")
        ax[k].legend(fontsize=8)
    fig.suptitle(f"{batch}: pore chord-length distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def write_report(out: str, metrics: pd.DataFrame, val: pd.DataFrame,
                 verdicts: dict, settings: dict,
                 losses: pd.DataFrame | None,
                 acq_info: dict | None = None) -> str:
    lines = []
    A = lines.append
    A("# recon3d — 2-D→3-D reconstruction & transport (optional "
      "sensitivity branch)\n")
    A("SliceGAN statistically reconstructs 3-D microstructures whose "
      "2-D slices match the segmented BSE cross-sections; a "
      "steady-state diffusion solve (same physics as the 2-D FDM "
      "solver / TauFactor) then measures through-plane and in-plane "
      "tortuosity, pore percolation and Si accessibility on each "
      "volume.\n")
    A("**Read this first.** Generated volumes are *statistical "
      "realisations consistent with* the 2-D data — they are not the "
      "real 3-D structure (unknowable from one section orientation) "
      "and not a measurement.  Every number below carries "
      "reconstruction uncertainty on top of the usual small-sample "
      "uncertainty.  Use the batch *comparison* of ensembles, never a "
      "single volume or an absolute value.\n")

    A("## Plain conclusion\n")
    A(_plain(metrics, verdicts))
    A("\n## Method\n")
    ri = settings.get("ruler_info") or {}
    A(f"- Measuring ruler: t_pore={ri.get('t_pore', 0):.4f} "
      f"t_si={ri.get('t_si', 0):.4f} — reference-only calibration "
      f"fitted on {ri.get('fitted_on', '?')} "
      f"({ri.get('n_fit_images', '?')} images, source "
      f"`{ri.get('source', '?')}`, ruler tag "
      f"`{settings.get('ruler', '?')}`). Same frozen recipe as "
      "micro2dfn's baseline-fitted `recipe.json`.")
    A(f"- Training data: 3-phase segmentations under that ruler, "
      f"majority-vote downsampled to "
      f"{settings.get('voxel_nm', config.VOXEL_NM):.0f} nm/voxel")
    A(f"- Volume: {settings.get('V', config.VOLUME)}³ voxels = "
      f"{settings.get('V', config.VOLUME) * settings.get('voxel_nm', config.VOXEL_NM) / 1000:.1f} "
      "µm edge (vs Si correlation length ~1.5 µm — a *marginal* RVE; "
      "see limits)")
    A(f"- Ensemble: {settings.get('n_volumes', '?')} realisations per "
      f"batch across {settings.get('n_seeds', '?')} training seeds")
    A(f"- WGAN-GP, {settings.get('iters', '?')} G-updates, "
      "discriminator sees ONLY z-containing slices → through-plane "
      "statistics from data, in-plane isotropy *assumed*")
    A("- Transport: Laplace solve on the spanning pore cluster "
      "(c=1 separator face, c=0 collector face, no-flux elsewhere; "
      "CG + Jacobi)\n")

    A("## Reconstruction validation\n")
    A("Generated z-slices vs held-out real crops — phase fractions, "
      "S2 two-point correlation, chord lengths.\n")
    cols = ["batch", "pore_frac_real", "pore_frac_fake",
            "bright_frac_real", "bright_frac_fake",
            "pore_s2_maxabsdiff", "bright_s2_maxabsdiff",
            "pore_corr_len_real_um", "pore_corr_len_fake_um",
            "pore_chord_z_l1"]
    show = val[[c for c in cols if c in val]].copy()
    A(show.round(3).to_markdown(index=False))
    A("")
    for b, vd in verdicts.items():
        A(f"- **{b}**: {vd}")

    if acq_info is not None:
        A("\n## Acquisition control — is this material or session?\n")
        A("The 31 training images come from "
          f"{acq_info['n_groups']} distinguishable acquisition groups "
          "(inferred imaging sessions), and 2-D marker analysis shows "
          "count/fragmentation statistics are dominated by session, "
          "not batch. Each batch's GAN trained on a different session "
          "mix "
          + ", ".join(f"{b}: {n} groups" for b, n in
                      sorted(acq_info['groups_per_batch'].items()))
          + " — so ensemble differences can be session differences "
          "learned faithfully by the model.\n")
        A("Variance partition on the training masks themselves "
          "(fraction of variance left after removing group means vs "
          "batch means — lower = explains more):\n")
        part = acq_info["partition"].copy()
        part["reads_mostly"] = np.where(
            part.resid_after_group < part.resid_after_batch,
            "the session", "the specimen")
        A(part.round(3).to_markdown(index=False))
        A("")
        mg = acq_info["mixed"]
        if len(mg):
            A("Sessions shared by >1 batch — the only directly "
              "comparable images (thin evidence, but "
              "session-controlled):\n")
            A(mg.round(3).to_markdown(index=False))
            A("")
        A("**Reading:** connectivity-relevant statistics on the "
          "masks (pore component density, largest-cluster share) "
          "track imaging session more closely than batch. The batch "
          "differences in the transport ensemble below are therefore "
          "*session-confounded candidates* — not established material "
          "differences. The session-robust findings are the ones that "
          "hold across all batches: resolved pores rarely span the "
          "volume anywhere, so real transport must run substantially "
          "through sub-resolution porosity.\n")

    A("## Transport results (ensemble per batch)\n")
    g = metrics.groupby("batch")
    tbl = []
    for b, d in g:
        def ci(col):
            v = d[col].replace(np.inf, np.nan).dropna()
            if not len(v):
                return "n/a"
            return (f"{v.median():.3g} [{v.quantile(.25):.3g}–"
                    f"{v.quantile(.75):.3g}]")
        n_span = int(d.spans_tp.sum())
        tbl.append(dict(
            batch=b, n=len(d), spanning_tp=f"{n_span}/{len(d)}",
            tau_tp=ci("tau_tp"), tau_ip=ci("tau_ip"),
            anisotropy=ci("tau_anisotropy"),
            spanning_frac=ci("spanning_frac_tp"),
            si_accessible=ci("si_accessible_frac"),
            resolved_porosity=ci("pore_frac")))
    A(pd.DataFrame(tbl).to_markdown(index=False))
    A("")
    A("`resolved_porosity` is the fraction of resolved pores in the "
      "generated volume — real electrodes additionally carry "
      "sub-resolution porosity (~+0.15–0.30 assumed in the DFN "
      "branch), so reconstructed τ overstates transport resistance.\n")

    A("## Batch comparison (session-confounded — see Acquisition "
      "control)\n")
    A(_comparison(metrics))
    A("\n## Assumptions & limits — read before quoting\n")
    A("0. **Acquisition confound (primary).** Each batch's training "
      "set mixes imaging sessions differently; pore-fragmentation "
      "statistics on the masks are session-dominated (see Acquisition "
      "control above). A GAN trained per batch inherits that mix — "
      "any per-batch ensemble difference is a *candidate* until "
      "session-matched imaging exists.")
    A("1. **Statistical consistency, not ground truth.** Many 3-D "
      "structures produce identical 2-D statistics; the ensemble "
      "samples that space, it does not recover *the* structure.")
    A("2. **Single-view anisotropy.** All sections share one "
      "orientation. Through-plane stats are learned; the two in-plane "
      "directions are *assumed* statistically equivalent. A calendered "
      "electrode may violate this (e.g. z-aligned pore columns "
      "invisible from any section). τ_through-plane is the most "
      "trustworthy output; τ_in-plane rests on the assumption.")
    A("3. **Resolved phases only.** Pores < ~0.35 µm are invisible to "
      "the training data, so the reconstructed network misses the "
      "fine porosity that carries much of real electrolyte transport. "
      "τ here is the *resolved-network* τ — an upper bound on the "
      "true effective τ.")
    A("4. **Marginal RVE.** 6.4 µm edge ≈ 4× the largest correlation "
      "length; per-volume values fluctuate accordingly — interpret "
      "ensemble spread, not single volumes.")
    A("5. **Bright phase = all thresholded bright material** (the "
      "'all-bright-counts' bracket of the Si/bright_fine ambiguity).")
    A("6. Validation is patch-level (all images trained) — it checks "
      "statistical fidelity, not generalisation.")
    A("7. **Generation artefacts exist.** Some realisations show "
      "periodic stripe structure along the weakly-constrained "
      "in-plane axis (visible in `figs/slices_*.png`), and phase "
      "fractions drift by up to ~4 pts from real (see the validation "
      "table). Both biases affect connectivity estimates — treat "
      "spanning-rate differences between batches as the primary, "
      "most robust signal.\n")

    A("## Outputs\n")
    A("`transport_metrics.csv` per-volume results · "
      "`validation.csv` real-vs-generated statistics · "
      "`volumes/*.npz` label volumes · `models/` training histories · "
      "`figs/` slice panels, S2 curves, chord distributions, "
      "transport boxplots, flux solve, 3-D render · "
      "`acquisition_control.csv` per-image mask stats joined to "
      "imaging groups · `masks_<ruler>_ds<n>.npz` frozen training "
      "data · `recipe.json` + `masks_manifest.json` ruler "
      "provenance\n")
    if losses is not None and len(losses):
        A("## Training diagnostics\n")
        A(losses.round(3).to_markdown(index=False))
        A("\nW-distance (D's real-fake gap) drifting towards ~0 and "
          "stable = converged; sustained growth = mismatch.\n")

    path = os.path.join(out, "report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    return path


def _plain(m: pd.DataFrame, verdicts: dict) -> str:
    ref = "Batch_3" if "Batch_3" in set(m.batch) else sorted(
        m.batch.unique())[-1]
    bits = []
    for b in sorted(m.batch.unique()):
        if b == ref:
            continue
        a = m[(m.batch == b) & np.isfinite(m.tau_tp)].tau_tp
        r = m[(m.batch == ref) & np.isfinite(m.tau_tp)].tau_tp
        na = int(m[m.batch == b].spans_tp.sum())
        nr = int(m[m.batch == ref].spans_tp.sum())
        if len(a) and len(r):
            bits.append(
                f"{b}: {na}/{int(m.batch.eq(b).sum())} realisations "
                f"span (vs {ref} {nr}/"
                f"{int(m.batch.eq(ref).sum())}), conditional median "
                f"through-plane τ {a.median():.2f} vs "
                f"{r.median():.2f} (small n — indicative only)")
    ok = all(v.startswith("PASS") or v.startswith("ACCEPTABLE")
             for v in verdicts.values())
    qual = ("Reconstructions pass statistical validation." if ok
            else "WARNING: some reconstructions failed validation — "
                 "transport numbers are illustrative.")
    body = ("; ".join(bits) if bits else
            "no finite through-plane τ in any ensemble — the "
            "resolved pore network does not percolate the 6.4 µm "
            "volume in these reconstructions")
    return ("The resolved pore network spans the 6.4 µm volume in "
            "only a minority of realisations in EVERY batch — "
            "consistent with the 2-D result that resolved pores "
            "rarely percolate a section; the connected transport "
            "network must live substantially in sub-resolution "
            "porosity. Per batch: " + body + ". "
            + qual + " Batch differences here are "
            "*model-ensemble* differences — they suggest, they do "
            "not measure; and since pore-fragmentation statistics "
            "on the training masks are dominated by imaging session "
            "(see Acquisition control), even the batch ordering is "
            "session-confounded pending session-matched imaging.")


def _comparison(m: pd.DataFrame) -> str:
    ref = "Batch_3" if "Batch_3" in set(m.batch) else sorted(
        m.batch.unique())[-1]
    L = []
    for b in sorted(m.batch.unique()):
        if b == ref:
            continue
        a_all = m[m.batch == b]
        r_all = m[m.batch == ref]
        L.append(f"- {b} vs {ref}, spanning rate: "
                 f"{int(a_all.spans_tp.sum())}/{len(a_all)} vs "
                 f"{int(r_all.spans_tp.sum())}/{len(r_all)}; mean "
                 f"spanning-pore share "
                 f"{a_all.spanning_frac_tp.mean():.3g} vs "
                 f"{r_all.spanning_frac_tp.mean():.3g}")
        for col, tag in (("tau_tp", "through-plane τ (spanning "
                                   "subset only)"),
                         ("si_accessible_frac", "accessible bright")):
            a = m[m.batch == b][col].replace(np.inf, np.nan).dropna()
            r = m[m.batch == ref][col].replace(np.inf, np.nan).dropna()
            if len(a) and len(r):
                L.append(f"- {b} vs {ref}, {tag}: "
                         f"{a.median():.3g} vs {r.median():.3g} "
                         f"(Δ {a.median() - r.median():+.3g}; "
                         f"n={len(a)}/{len(r)})")
    L.append("\nNon-spanning realisations (τ = ∞) count as "
             "evidence of a poorly connected network, not as "
             "missing data — see `spans_tp` in "
             "transport_metrics.csv.")
    return "\n".join(L)
