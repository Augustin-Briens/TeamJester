# Branch: polaron-acquisition-control

Acquisition/session-confound analysis + the clean-room paper and the
repository cleanup that followed it (second commit adds the cleanup).

## What's here

- `image_acquisition.csv` / `image_acquisition_groups.csv` — 13
  inferred acquisition groups (frame size, pixel size, detector set,
  noise, contrast) for all 31 images + per-group summary
- `marker_session_partition.csv` — variance partition per marker:
  residual after group means vs after batch means
- `unblocked_permutation.csv` / `blocked_permutation.csv` — the 42-
  marker declared family: 3 BH survivors unblocked (all count-type),
  0 survive within-session blocking
- `report_numbers.json` + `build_report.py` → `build_paper.py` →
  `check_report.py` — the paper's reproducible number chain; every
  value in the tex is computed from the CSVs, and `check_report.py`
  fails on stale/archived strings
- `paper/micro2dfn_paper.tex` + `.pdf` — "clean-room edition": primary
  fraction markers only, acquisition confound, blocked comparisons,
  candidate bright fields, DFN demoted to appendix
- `archive/` — ARCHIVE.md documents each withdrawn route
  (FFT direction/`fft_vh`, FDM tortuosity, swelling budget, recon3d,
  `orientation_features.csv`); archived code stays runnable under
  `archive/`
- `micro2dfn/` — cleaned pipeline: `markers.py` without the removed
  FDM/swelling code, `report.py` now emits permutation + BH + MDD,
  `dfn_inputs.py` with the corrected particle-radius keys
  (`Primary/Secondary: Negative particle radius [m]`, graphite radius
  now measured via `gr_radius_eff_um`)
- `analyze_channels.py` / `channel_report.md` — InLens/ETD layer
  demoted to exploratory; curtaining test removed
- `analyze_spatial.py` / `spatial_report.md` — spatial markers labelled
  diagnostic/exploratory; physical-axis interpretation withdrawn
- `assign_new_images.py` + `new_image_assignment/` — all three calls
  relabelled "can't tell" (session-matched); original blind calls kept
  in `assigned_batch_original`
- `dfn_output/` — current outputs incl. `report.md`, `recipe.json`,
  `run_manifest.json`, `assumption_table.csv`, `dfn_validation.md`,
  `markers_all.csv`, `objects_all.csv`, `batch_differences.csv`;
  radius-key caveat documented in AGENTS.md (saved runs used default
  radii — radius-channel sensitivity untested)
- `dfn_smoke/` — early smoke output kept for provenance (superseded:
  contains withdrawn swelling-budget/old claims wording)
- `reproduce_si_accessible.py` — open defect: accessible-Si metric
  (enclosed-share denominator / per-component vs per-particle labels)
  under audit; those metrics are provisional until fixed
- `qc_output/{qc_report,PLAIN_ENGLISH_REPORT,z_scores}` — QC reports
  with corrected framing
- `teaching_micro2dfn/` — self-updating teaching pack: 41-slide deck
  (PPTX+PDF+rendered), 39 figures, claim_evidence.csv (C42 now
  WITHDRAWN), audit_claims.py (73 checks), sync.py (fingerprint-driven
  incremental rebuilds)
- `recon3d_output/` — generated results incl. `acquisition_control.csv`
- `AGENTS.md` — project notes, session-corrected results

## Headline of this work

No detectable composition difference (~6% candidate-Si, ~10-11%
resolved pores in all batches) and no batch difference separable from
imaging session: every unblocked signal is count-type and collapses
under within-session blocking. Two Batch_1 fields exceed the
reference bright-area range (12.5%, 11.9% vs ref max 10.7%) — solid
as an image observation, unresolved as material. Fractions are the
primary markers; counts are within-session diagnostics only.
"Can't tell" ≠ "identical" — the blocked test is underpowered.

Raw images (Batch_1/2/3, New_Images_Batch) are intentionally not on
this branch — they're on `main` under `Hackathon-Polaron/`.
