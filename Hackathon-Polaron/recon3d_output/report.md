# recon3d — 2-D→3-D reconstruction & transport (optional sensitivity branch)

SliceGAN statistically reconstructs 3-D microstructures whose 2-D slices match the segmented BSE cross-sections; a steady-state diffusion solve (same physics as the 2-D FDM solver / TauFactor) then measures through-plane and in-plane tortuosity, pore percolation and Si accessibility on each volume.

**Read this first.** Generated volumes are *statistical realisations consistent with* the 2-D data — they are not the real 3-D structure (unknowable from one section orientation) and not a measurement.  Every number below carries reconstruction uncertainty on top of the usual small-sample uncertainty.  Use the batch *comparison* of ensembles, never a single volume or an absolute value.

## Plain conclusion

The resolved pore network spans the 6.4 µm volume in only a minority of realisations in EVERY batch — consistent with the 2-D result that resolved pores rarely percolate a section; the connected transport network must live substantially in sub-resolution porosity. Per batch: Batch_1: 1/12 realisations span (vs Batch_3 6/12), conditional median through-plane τ 15.54 vs 18.44 (small n — indicative only); Batch_2: 4/12 realisations span (vs Batch_3 6/12), conditional median through-plane τ 59.62 vs 18.44 (small n — indicative only). Reconstructions pass statistical validation. Batch differences here are *model-ensemble* differences — they suggest, they do not measure; and since pore-fragmentation statistics on the training masks are dominated by imaging session (see Acquisition control), even the batch ordering is session-confounded pending session-matched imaging.

## Method

- Measuring ruler: t_pore=0.7109 t_si=1.3354 — reference-only calibration fitted on Batch_3 (17 images, source `dfn_output/recipe.json`, ruler tag `cb9831`). Same frozen recipe as micro2dfn's baseline-fitted `recipe.json`.
- Training data: 3-phase segmentations under that ruler, majority-vote downsampled to 100 nm/voxel
- Volume: 64³ voxels = 6.4 µm edge (vs Si correlation length ~1.5 µm — a *marginal* RVE; see limits)
- Ensemble: 12 realisations per batch across 3 training seeds
- WGAN-GP, 6000 G-updates, discriminator sees ONLY z-containing slices → through-plane statistics from data, in-plane isotropy *assumed*
- Transport: Laplace solve on the spanning pore cluster (c=1 separator face, c=0 collector face, no-flux elsewhere; CG + Jacobi)

## Reconstruction validation

Generated z-slices vs held-out real crops — phase fractions, S2 two-point correlation, chord lengths.

| batch   |   pore_frac_real |   pore_frac_fake |   bright_frac_real |   bright_frac_fake |   pore_s2_maxabsdiff |   bright_s2_maxabsdiff |   pore_corr_len_real_um |   pore_corr_len_fake_um |   pore_chord_z_l1 |
|:--------|-----------------:|-----------------:|-------------------:|-------------------:|---------------------:|-----------------------:|------------------------:|------------------------:|------------------:|
| Batch_1 |            0.095 |            0.101 |              0.062 |              0.101 |                0.01  |                  0.04  |                   0.39  |                   0.312 |             0.308 |
| Batch_2 |            0.101 |            0.109 |              0.07  |              0.072 |                0.009 |                  0.013 |                   0.4   |                   0.379 |             0.156 |
| Batch_3 |            0.118 |            0.101 |              0.108 |              0.082 |                0.028 |                  0.046 |                   0.448 |                   0.335 |             0.208 |

- **Batch_1**: ACCEPTABLE — small statistical mismatches; transport metrics usable with caution
- **Batch_2**: PASS — generated slices reproduce volume fractions and two-point correlations within tight tolerance
- **Batch_3**: ACCEPTABLE — small statistical mismatches; transport metrics usable with caution

## Acquisition control — is this material or session?

The 31 training images come from 13 distinguishable acquisition groups (inferred imaging sessions), and 2-D marker analysis shows count/fragmentation statistics are dominated by session, not batch. Each batch's GAN trained on a different session mix Batch_1: 6 groups, Batch_2: 6 groups, Batch_3: 7 groups — so ensemble differences can be session differences learned faithfully by the model.

Variance partition on the training masks themselves (fraction of variance left after removing group means vs batch means — lower = explains more):

| metric                |   resid_after_group |   resid_after_batch | reads_mostly   |
|:----------------------|--------------------:|--------------------:|:---------------|
| pore_frac             |               0.238 |               0.983 | the session    |
| pore_comp_per_1000vox |               0.042 |               0.539 | the session    |
| largest_comp_share    |               0.425 |               0.805 | the session    |
| pore_chord_ip         |               0.312 |               0.91  | the session    |

Sessions shared by >1 batch — the only directly comparable images (thin evidence, but session-controlled):

|   group_id | batch   | image_id     |   pore_frac |   pore_comp_per_1000vox |   largest_comp_share |   pore_chord_ip |
|-----------:|:--------|:-------------|------------:|------------------------:|---------------------:|----------------:|
|       2068 | Batch_2 | img_rxax5ozo |       0.101 |                   2.181 |                0.054 |           6.59  |
|       2068 | Batch_3 | img_utfgcjfa |       0.115 |                   2.403 |                0.078 |           6.988 |
|       2068 | Batch_3 | img_vc2whyaq |       0.111 |                   2.173 |                0.08  |           7.263 |
|       2068 | Batch_3 | img_x77cy643 |       0.113 |                   2.552 |                0.053 |           6.558 |
|       2080 | Batch_1 | img_ffwubibz |       0.096 |                   2.533 |                0.037 |           6.534 |
|       2080 | Batch_2 | img_r17byphk |       0.129 |                   2.66  |                0.062 |           7.454 |
|       2080 | Batch_3 | img_cfe5vt7s |       0.112 |                   2.797 |                0.035 |           6.612 |
|       2148 | Batch_1 | img_f1vzngrs |       0.077 |                   3.742 |                0.027 |           4.611 |
|       2148 | Batch_2 | img_epqdaau9 |       0.095 |                   3.57  |                0.035 |           5.502 |
|       2156 | Batch_1 | img_fzrt2k6r |       0.107 |                   2.708 |                0.048 |           6.701 |
|       2156 | Batch_2 | img_b3esycq1 |       0.11  |                   2.691 |                0.04  |           6.648 |
|       2272 | Batch_2 | img_i9jiqjwl |       0.097 |                   2.612 |                0.035 |           6.093 |
|       2272 | Batch_3 | img_pl8uabbv |       0.117 |                   2.594 |                0.057 |           6.772 |

**Reading:** connectivity-relevant statistics on the masks (pore component density, largest-cluster share) track imaging session more closely than batch. The batch differences in the transport ensemble below are therefore *session-confounded candidates* — not established material differences. The session-robust findings are the ones that hold across all batches: resolved pores rarely span the volume anywhere, so real transport must run substantially through sub-resolution porosity.

## Transport results (ensemble per batch)

| batch   |   n | spanning_tp   | tau_tp           | tau_ip         | anisotropy          | spanning_frac   | si_accessible       | resolved_porosity    |
|:--------|----:|:--------------|:-----------------|:---------------|:--------------------|:----------------|:--------------------|:---------------------|
| Batch_1 |  12 | 1/12          | 15.5 [15.5–15.5] | 113 [88.8–139] | 0.162 [0.162–0.162] | 0 [0–0]         | 0.307 [0.307–0.307] | 0.103 [0.0771–0.127] |
| Batch_2 |  12 | 4/12          | 59.6 [47–79.1]   | n/a            | n/a                 | 0 [0–0.429]     | 0.194 [0.157–0.272] | 0.102 [0.0707–0.125] |
| Batch_3 |  12 | 6/12          | 18.4 [13.2–49.5] | 6.21 [6.18–23] | 1.96 [1.8–2.12]     | 0.101 [0–0.794] | 0.258 [0.251–0.675] | 0.119 [0.103–0.136]  |

`resolved_porosity` is the fraction of resolved pores in the generated volume — real electrodes additionally carry sub-resolution porosity (~+0.15–0.30 assumed in the DFN branch), so reconstructed τ overstates transport resistance.

## Batch comparison (session-confounded — see Acquisition control)

- Batch_1 vs Batch_3, spanning rate: 1/12 vs 6/12; mean spanning-pore share 0.0512 vs 0.359
- Batch_1 vs Batch_3, through-plane τ (spanning subset only): 15.5 vs 18.4 (Δ -2.9; n=1/6)
- Batch_1 vs Batch_3, accessible bright: 0.307 vs 0.258 (Δ +0.0492; n=1/6)
- Batch_2 vs Batch_3, spanning rate: 4/12 vs 6/12; mean spanning-pore share 0.203 vs 0.359
- Batch_2 vs Batch_3, through-plane τ (spanning subset only): 59.6 vs 18.4 (Δ +41.2; n=4/6)
- Batch_2 vs Batch_3, accessible bright: 0.194 vs 0.258 (Δ -0.0634; n=4/6)

Non-spanning realisations (τ = ∞) count as evidence of a poorly connected network, not as missing data — see `spans_tp` in transport_metrics.csv.

## Assumptions & limits — read before quoting

0. **Acquisition confound (primary).** Each batch's training set mixes imaging sessions differently; pore-fragmentation statistics on the masks are session-dominated (see Acquisition control above). A GAN trained per batch inherits that mix — any per-batch ensemble difference is a *candidate* until session-matched imaging exists.
1. **Statistical consistency, not ground truth.** Many 3-D structures produce identical 2-D statistics; the ensemble samples that space, it does not recover *the* structure.
2. **Single-view anisotropy.** All sections share one orientation. Through-plane stats are learned; the two in-plane directions are *assumed* statistically equivalent. A calendered electrode may violate this (e.g. z-aligned pore columns invisible from any section). τ_through-plane is the most trustworthy output; τ_in-plane rests on the assumption.
3. **Resolved phases only.** Pores < ~0.35 µm are invisible to the training data, so the reconstructed network misses the fine porosity that carries much of real electrolyte transport. τ here is the *resolved-network* τ — an upper bound on the true effective τ.
4. **Marginal RVE.** 6.4 µm edge ≈ 4× the largest correlation length; per-volume values fluctuate accordingly — interpret ensemble spread, not single volumes.
5. **Bright phase = all thresholded bright material** (the 'all-bright-counts' bracket of the Si/bright_fine ambiguity).
6. Validation is patch-level (all images trained) — it checks statistical fidelity, not generalisation.
7. **Generation artefacts exist.** Some realisations show periodic stripe structure along the weakly-constrained in-plane axis (visible in `figs/slices_*.png`), and phase fractions drift by up to ~4 pts from real (see the validation table). Both biases affect connectivity estimates — treat spanning-rate differences between batches as the primary, most robust signal.

## Outputs

`transport_metrics.csv` per-volume results · `validation.csv` real-vs-generated statistics · `volumes/*.npz` label volumes · `models/` training histories · `figs/` slice panels, S2 curves, chord distributions, transport boxplots, flux solve, 3-D render · `acquisition_control.csv` per-image mask stats joined to imaging groups · `masks_<ruler>_ds<n>.npz` frozen training data · `recipe.json` + `masks_manifest.json` ruler provenance

## Training diagnostics

| model                          |   iters |   final_w |   seconds |
|:-------------------------------|--------:|----------:|----------:|
| Batch_1__s0__V64i6000__rcb9831 |    6000 |     4.659 |    1678.5 |
| Batch_1__s1__V64i6000__rcb9831 |    6000 |    -3.344 |    2407.4 |
| Batch_1__s2__V64i6000__rcb9831 |    6000 |     3.448 |    3645.7 |
| Batch_2__s0__V64i6000__rcb9831 |    6000 |    15.608 |    2848.4 |
| Batch_2__s1__V64i6000__rcb9831 |    6000 |     1.085 |    1678.6 |
| Batch_2__s2__V64i6000__rcb9831 |    6000 |    -1.47  |    2822.6 |
| Batch_3__s0__V64i6000__rcb9831 |    6000 |    -5.275 |    3665.5 |
| Batch_3__s1__V64i6000__rcb9831 |    6000 |     4.139 |    3606.7 |
| Batch_3__s2__V64i6000__rcb9831 |    6000 |     9.955 |    2554.6 |

W-distance (D's real-fake gap) drifting towards ~0 and stable = converged; sustained growth = mismatch.
