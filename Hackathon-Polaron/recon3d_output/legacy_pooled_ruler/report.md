# recon3d — 2-D→3-D reconstruction & transport (optional sensitivity branch)

SliceGAN statistically reconstructs 3-D microstructures whose 2-D slices match the segmented BSE cross-sections; a steady-state diffusion solve (same physics as the 2-D FDM solver / TauFactor) then measures through-plane and in-plane tortuosity, pore percolation and Si accessibility on each volume.

**Read this first.** Generated volumes are *statistical realisations consistent with* the 2-D data — they are not the real 3-D structure (unknowable from one section orientation) and not a measurement.  Every number below carries reconstruction uncertainty on top of the usual small-sample uncertainty.  Use the batch *comparison* of ensembles, never a single volume or an absolute value.

## Plain conclusion

The resolved pore network spans the 6.4 µm volume in only a minority of realisations in EVERY batch — consistent with the 2-D result that resolved pores rarely percolate a section; the connected transport network must live substantially in sub-resolution porosity. Per batch: Batch_1: 5/12 realisations span (vs Batch_3 5/12), conditional median through-plane τ 30.10 vs 14.24 (small n — indicative only); Batch_2: 3/12 realisations span (vs Batch_3 5/12), conditional median through-plane τ 13.17 vs 14.24 (small n — indicative only). Reconstructions pass statistical validation. Batch differences here are *model-ensemble* differences — they suggest, they do not measure.

## Method

- Training data: micro2dfn frozen-threshold segmentations (identical phase recipe), majority-vote downsampled to 100 nm/voxel
- Volume: 64³ voxels = 6.4 µm edge (vs Si correlation length ~1.5 µm — a *marginal* RVE; see limits)
- Ensemble: 12 realisations per batch across 3 training seeds
- WGAN-GP, 6000 G-updates, discriminator sees ONLY z-containing slices → through-plane statistics from data, in-plane isotropy *assumed*
- Transport: Laplace solve on the spanning pore cluster (c=1 separator face, c=0 collector face, no-flux elsewhere; CG + Jacobi)

## Reconstruction validation

Generated z-slices vs held-out real crops — phase fractions, S2 two-point correlation, chord lengths.

| batch   |   pore_frac_real |   pore_frac_fake |   bright_frac_real |   bright_frac_fake |   pore_s2_maxabsdiff |   bright_s2_maxabsdiff |   pore_corr_len_real_um |   pore_corr_len_fake_um |   pore_chord_z_l1 |
|:--------|-----------------:|-----------------:|-------------------:|-------------------:|---------------------:|-----------------------:|------------------------:|------------------------:|------------------:|
| Batch_1 |            0.088 |            0.104 |              0.063 |              0.092 |                0.016 |                  0.029 |                   0.388 |                   0.312 |             0.254 |
| Batch_2 |            0.093 |            0.128 |              0.071 |              0.076 |                0.035 |                  0.01  |                   0.394 |                   0.458 |             0.199 |
| Batch_3 |            0.109 |            0.106 |              0.109 |              0.087 |                0.007 |                  0.031 |                   0.442 |                   0.425 |             0.217 |

- **Batch_1**: ACCEPTABLE — small statistical mismatches; transport metrics usable with caution
- **Batch_2**: ACCEPTABLE — small statistical mismatches; transport metrics usable with caution
- **Batch_3**: ACCEPTABLE — small statistical mismatches; transport metrics usable with caution

## Transport results (ensemble per batch)

| batch   |   n | spanning_tp   | tau_tp           | tau_ip           | anisotropy          | spanning_frac   | si_accessible        | resolved_porosity     |
|:--------|----:|:--------------|:-----------------|:-----------------|:--------------------|:----------------|:---------------------|:----------------------|
| Batch_1 |  12 | 5/12          | 30.1 [22–33.9]   | 23.5 [15.2–43.7] | 0.956 [0.588–1.12]  | 0 [0–0.834]     | 0.111 [0.0885–0.364] | 0.109 [0.0799–0.127]  |
| Batch_2 |  12 | 3/12          | 13.2 [12.9–15.7] | 55.7 [33.8–77.7] | 0.615 [0.37–0.859]  | 0 [0–0.127]     | 0.491 [0.409–0.541]  | 0.117 [0.0989–0.142]  |
| Batch_3 |  12 | 5/12          | 14.2 [13.3–17.9] | 66.9 [66.9–66.9] | 0.513 [0.513–0.513] | 0 [0–0.701]     | 0.169 [0.15–0.17]    | 0.0926 [0.0802–0.116] |

`resolved_porosity` is the fraction of resolved pores in the generated volume — real electrodes additionally carry sub-resolution porosity (~+0.15–0.30 assumed in the DFN branch), so reconstructed τ overstates transport resistance.

## Batch comparison

- Batch_1 vs Batch_3, spanning rate: 5/12 vs 5/12; mean spanning-pore share 0.349 vs 0.326
- Batch_1 vs Batch_3, through-plane τ (spanning subset only): 30.1 vs 14.2 (Δ +15.9; n=5/5)
- Batch_1 vs Batch_3, accessible bright: 0.111 vs 0.169 (Δ -0.0588; n=5/5)
- Batch_2 vs Batch_3, spanning rate: 3/12 vs 5/12; mean spanning-pore share 0.19 vs 0.326
- Batch_2 vs Batch_3, through-plane τ (spanning subset only): 13.2 vs 14.2 (Δ -1.07; n=3/5)
- Batch_2 vs Batch_3, accessible bright: 0.491 vs 0.169 (Δ +0.322; n=3/5)

Non-spanning realisations (τ = ∞) count as evidence of a poorly connected network, not as missing data — see `spans_tp` in transport_metrics.csv.

## Assumptions & limits — read before quoting

1. **Statistical consistency, not ground truth.** Many 3-D structures produce identical 2-D statistics; the ensemble samples that space, it does not recover *the* structure.
2. **Single-view anisotropy.** All sections share one orientation. Through-plane stats are learned; the two in-plane directions are *assumed* statistically equivalent. A calendered electrode may violate this (e.g. z-aligned pore columns invisible from any section). τ_through-plane is the most trustworthy output; τ_in-plane rests on the assumption.
3. **Resolved phases only.** Pores < ~0.35 µm are invisible to the training data, so the reconstructed network misses the fine porosity that carries much of real electrolyte transport. τ here is the *resolved-network* τ — an upper bound on the true effective τ.
4. **Marginal RVE.** 6.4 µm edge ≈ 4× the largest correlation length; per-volume values fluctuate accordingly — interpret ensemble spread, not single volumes.
5. **Bright phase = all thresholded bright material** (the 'all-bright-counts' bracket of the Si/bright_fine ambiguity).
6. Validation is patch-level (all images trained) — it checks statistical fidelity, not generalisation.
7. **Generation artefacts exist.** Some realisations show periodic stripe structure along the weakly-constrained in-plane axis (visible in `figs/slices_*.png`), and phase fractions drift by up to ~4 pts from real (see the validation table). Both biases affect connectivity estimates — treat spanning-rate differences between batches as the primary, most robust signal.

## Outputs

`transport_metrics.csv` per-volume results · `validation.csv` real-vs-generated statistics · `volumes/*.npz` label volumes · `models/` training histories · `figs/` slice panels, S2 curves, chord distributions, transport boxplots, flux solve, 3-D render · `masks.npz` frozen training data

## Training diagnostics

| model                 |   iters |   final_w |   seconds |
|:----------------------|--------:|----------:|----------:|
| Batch_1__s0__V64i6000 |    6000 |     9.477 |    2674.9 |
| Batch_1__s1__V64i6000 |    6000 |     2.628 |    1685.2 |
| Batch_1__s2__V64i6000 |    6000 |     4.038 |    1515.1 |
| Batch_2__s0__V64i6000 |    6000 |     8.417 |    3271.2 |
| Batch_2__s1__V64i6000 |    6000 |     2.734 |    3449.7 |
| Batch_2__s2__V64i6000 |    6000 |     0.186 |    2802.5 |
| Batch_3__s0__V64i6000 |    6000 |     3.545 |    1687.7 |
| Batch_3__s1__V64i6000 |    6000 |    -2.844 |    2925.3 |
| Batch_3__s2__V64i6000 |    6000 |     4.081 |    2885.7 |

W-distance (D's real-fake gap) drifting towards ~0 and stable = converged; sustained growth = mismatch.
