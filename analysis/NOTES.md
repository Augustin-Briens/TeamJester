# NOTES - assumptions, failures, exclusions

## Assumptions
- Assumption (applies throughout): the grey bulk is assumed graphite, bright particles are assumed silicon-based, black is pore/crack. This comes from image appearance and is unconfirmed.
- Pixel size 25.0 nm/px read from TIFF
  XResolution; identical for every image; FOV ~175 x 54 um.
- The three detectors are pixel-aligned (peak cross-correlation at zero
  shift), so Inlens dark pixels are used to refine pores.
- Visible porosity (~10%) underestimates true porosity: pores below the
  ~50 nm pixel resolution are invisible. DFN is run both with the
  measured value and a literature-range correction setting porosity to 0.30.
- Two-point correlation / Clark-Evans / chord methods treat each image
  as one statistical realisation; images within a batch are the samples.
- Tortuosity proxies (tau_x, tau_y) are 2D diffusion solves on the
  downsampled pore mask; a 2D section at ~9% porosity cannot represent
  3D transport - reported as proxy only and dropped from the decision
  set for non-repeatability.

## Excluded / flagged images
- Batch_1 4ih2ggld: low Si/bulk contrast; high noise; high noise-driven sharpness
- Batch_1 5n1q8atc: low Si/bulk contrast; high noise; high noise-driven sharpness

## Dropped features (not used in baseline/deltas; reasons in
## dropped_features.csv)
- si_share_of_solid: duplicate of silicon_frac (|r|>0.9)
- si_per_mm2: duplicate of silicon_frac (|r|>0.9)
- si_d90_um: left/right corr 0.19 < 0.5
- si_d99_um: left/right corr -0.10 < 0.5
- si_dmax_um: left/right corr 0.01 < 0.5
- si_d50_area_um: left/right corr 0.08 < 0.5
- si_large_area_frac: left/right corr 0.20 < 0.5
- si_aspect: left/right corr -0.00 < 0.5
- si_sv: left/right corr 0.42 < 0.5
- si_cracked_share: left/right corr nan (unrepeatable: one/both halves constant)
- gr_chord_v_um: duplicate of gr_chord_h_um (|r|>0.9)
- gr_sv: left/right corr 0.36 < 0.5
- pore_thick_d90_um: duplicate of pore_thick_d50_um (|r|>0.9)
- crack_frac: left/right corr 0.26 < 0.5
- compact_pore_frac: left/right corr 0.21 < 0.5
- pore_chord_h_um: duplicate of pore_frac (|r|>0.9)
- pore_chord_v_um: duplicate of pore_frac (|r|>0.9)
- pore_anisotropy: left/right corr 0.48 < 0.5
- pore_depth_cv: left/right corr -0.05 < 0.5
- pore_depth_slope: left/right corr 0.08 < 0.5
- crack_len_density: left/right corr 0.47 < 0.5
- si_pore_dist_mean_um: left/right corr 0.16 < 0.5
- si_no_pore_share: left/right corr 0.34 < 0.5
- corr_len_pore_um: duplicate of pore_thick_d50_um (|r|>0.9)
- corr_len_si_um: left/right corr 0.02 < 0.5
- lineal_path_pore_um: duplicate of pore_frac (|r|>0.9)
- pore_patchiness_cv: left/right corr 0.33 < 0.5
- si_patchiness_cv: left/right corr 0.27 < 0.5
- tau_x: left/right corr nan (unrepeatable: one/both halves constant)
- tau_y: left/right corr nan (unrepeatable: one/both halves constant)
- pore_percolates_x: constant or >50% missing
- pore_percolates_y: constant or >50% missing
- pore_largest_cc_frac: duplicate of large_gap_frac (|r|>0.9)

## Failures / known limitations
- tau_x/tau_y, pore percolation and crack/skeleton features fail the
  left/right repeatability check (NaN or <0.5 corr) - pores do not
  percolate at ~9% in these 2D sections. Reported in
  deltas_dropped.csv but not used in the decision set.
- Single-image categorisation is unreliable (LOO false-alarm ~50%);
  use >=5 images per sample.
- bse_bulk_texture and etd_roughness correlate ~0.9 with the image
  noise guard - treated as imaging-signature (artefact-risk) features
  and excluded from the artefact-safe categoriser.
- Multi-Otsu valley detection failed on images with a poorly-separated
  bright tail; the guard/flag mechanism covers those.
- point_count_validation.xlsx human column is intentionally empty;
  score with score_pointcount.py.

## Verification vs prior quick-pass
- silicon fraction same across batches: AGREES once the 2 flagged
  images are excluded (with them, +45% driven by low contrast).
- Batch_1 ~18% less pore space: AGREES directionally (-13.6% here,
  suggestive not Holm-significant).
- Batch_2 fewer elongated cracks: NOT confirmed - crack_len_density
  is -21% in Batch_1 not Batch_2, and the feature itself is
  unrepeatable (dropped).
- flagged problem images 4ih2ggld / 5n1q8atc: CONFIRMED by the
  contrast guard (si_bulk_contrast 2.15/2.27 vs >=3.40 elsewhere).



## Follow-up: improving Batch_2 discrimination (2026-10-03)

Requested: improve B2 accuracy; try new pointers and/or DINOv2.

### New explainable features (experiment_newfeat.py)
14 candidates computed full + L/R halves; same gates as Phase 2 (lr_corr >=
0.5, |corr with noise_mad| < 0.85). Passed: elong_pore_n_mm2 (lr 0.67),
si_border_pore (0.86), bse_grad_coh (0.54), inl_lbp_ent (0.88),
inl_lbp_flat (0.98), inl_grad_coh (0.65), etd_fft_hi (0.76).
LOO with the expanded pool: 48% single-image (vs 42%) — B2 still overlaps
B3; si_border_pore picked in every fold. Failed gates (unrepeatable or
noise-correlated): elong_pore_area_frac, elong_pore_hfrac, pore_aspect_p90,
pore_small_n_mm2, pore_perim_mm, bse_fft_hi, bse_spec_aniso, bse_glcm_*,
etd_spec_aniso. Tables: exp_newfeat.csv, exp_newfeat_gates.csv,
loo_predictions_exp_full.csv, exp_images_needed.csv.

### DINOv2 embeddings — REMOVED (2026-10-04)
A dinov2-small embedding variant (ETD topography, nearest-centroid on PCA
of CLS+patch-mean) was evaluated: group-of-5 accuracy ~100% for B1/B2, not
noise-correlated (r=0.17). Removed from the product per user request —
code and artifacts deleted; the explainable classical calls remain the
whole product.

## Follow-up: B2-vs-B3 separation options (2026-10-03)

Dedicated two-class discrimination tested on the 24 B2+B3 images:
- logreg + in-fold top-5 feature selection: **87.5% LOO** (B2 86%, B3 88%)
  but the picks lean on bse_bulk_texture + etd_roughness (the
  noise-correlated imaging features) in every fold.
- artefact-safe pool only: 66.7% LOO (B2 14%) — most of the single-image
  separation rides on imaging signatures, material features alone are weak.
- depth-2 tree 58%, LDA 54% — none better.
- MMD permutation test: MMD^2=0.018, p=0.19 — with n=7
  the whole-distribution difference is NOT provable; supervised separation
  remains valid evidence, group-level calls are the reliable unit.
- Shipped: a "B2-vs-B3 focused" logreg column in the webapp per-image table
  (all-features + artefact-safe variants, honestly labelled).
- New features merged into features.py compute() (extra_features) with
  FEATURE_META entries; gates recorded in exp_newfeat_gates.csv.

## Follow-up: metric removals (2026-10-04)

- **DINOv2 embedding call removed** from webapp + analysis (files deleted,
  torch/transformers/safetensors out of webapp deps).
- **si_d10_um removed** ("size of the smallest silicon particles"):
  stereologically broken — a 2D section through tightly packed particles
  reports glancing chord-cuts as small particles (~0.37 um floor on every
  batch = the segmentation floor, not a material property). Removed from
  features.py compute/meta, categorise.py selection (STEREO_BROKEN) +
  PLAIN_WORDS, and all committed tables (features.csv column,
  baseline_stats, repeatability, feature_meta, deltas*, envelope.json).
  LOO tables regenerated.
