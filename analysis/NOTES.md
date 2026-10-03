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


