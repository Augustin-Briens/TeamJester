# Branch: polaron-acquisition-control

Acquisition/session-confound analysis + the revised paper that reports it,
plus the recon3d acquisition-control code (now under `archive/`).

## What's here

- `image_acquisition.csv` — 13 inferred acquisition groups (frame size,
  pixel size, detector set, noise, contrast) for all 31 images
- `marker_session_partition.csv` — variance partition per marker:
  residual after group means vs after batch means
- `report_numbers.json` + `build_report.py` → `build_paper.py` →
  `check_report.py` — the paper's reproducible number chain; every
  value in the tex is computed from the CSVs, and `check_report.py`
  fails on stale/archived strings
- `paper/micro2dfn_paper.tex` + `.pdf` — revised paper: four
  "differences" demoted to candidates, acquisition-confound section,
  within-session blocking tests, corrected DFN section
- `archive/recon3d/` + `archive/run_recon3d.py` +
  `archive/modal_recon3d.py` + `archive/ARCHIVE.md` — archived
  optional 3-D branch incl. the acquisition-control analysis
  (`acquisition_table` → `acquisition_control.csv`, variance
  partition, mixed-group listing)
- `recon3d_output/` — generated results: `report.md` (Acquisition
  control section), `acquisition_control.csv`, `recipe.json` +
  `masks_manifest.json` (ruler provenance), transport metrics,
  validation, volumes, figures
- `micro2dfn/` + `run_dfn.py` + `modal_dfn.py` — pipeline dependency
  (recon3d/data.py reuses `dfn_output/recipe.json`; baseline-fitted
  frozen ruler)
- `dfn_output/{recipe,run_manifest,thresholds}.json`,
  `dfn_output/dfn_paired_differences.csv`,
  `validated_comparison/{per_image_features,objects_classified}.csv`
  — data inputs the paper chain reads
- `AGENTS.md` — project notes incl. the acquisition-control
  documentation

## Headline of this work

Count/fragmentation markers are dominated by imaging session, not
batch (pore-count residual after group ≈ 4.3% vs ≈ 52.7% after
batch); area fractions are robust. Within-session blocking collapses
every previously reported batch difference. The defensible statements
are measurement-level only — see the paper's acquisition section.

Raw images (Batch_1/2/3, New_Images_Batch) are intentionally not on
this branch — they're on `main` under `Hackathon-Polaron/`.
