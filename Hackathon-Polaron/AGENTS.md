# Polaron QC — Track 4 (Hackathon)

Uncertainty-aware QC system comparing incoming anode-material batches
against an approved baseline (Batch_3) on BSE cross-sections.

## Run

```bash
.venv/bin/python run_qc.py                          # baseline=Batch_3, assess all other Batch_* dirs
.venv/bin/python run_qc.py --incoming Batch_4       # only assess the new 8h drop
.venv/bin/python run_qc.py --baseline Batch_3 --out qc_output2
```

Outputs land in `qc_output/`: `qc_report.md` (technical deliverable),
`PLAIN_ENGLISH_REPORT.md` (lay one-pager), `polaron_qc_results.xlsx`
(9-sheet workbook), `verdicts.csv`, `z_scores.csv`, `scorecard.csv`,
`consequences.csv`, `features_*.csv`, figures (`fig_image_scores` control
chart, `fig_fingerprints`, `fig_z_heatmap`, `fig_scorecard`,
`fig_raw_compare_*`, `fig_budget_*` swelling maps, `fig_dist_*`,
`seg_*` overlays), `cache/` (per-image feature cache — auto-invalidated on
recipe changes via FEATURE_VERSION + frozen thresholds).

## Architecture (`polaron_qc/`)

- `io.py` — TIFF load (25 nm/px from resolution tags), flat-field
  correction (sigma-200 Gaussian on 16x-downsampled image), `load_inlens`
  for the surface-sensitive channel.
- `segmentation.py` — ONE multi-Otsu threshold pair fitted on pooled
  BASELINE pixels and frozen for every image (the teammate per-image
  recipe is kept as `floating_thresholds` — diagnostic only; a moving
  ruler absorbs composition shifts). `intensity_signature` = raw-image
  percentiles for the instrument-drift guard. Sub-resolution objects
  (Si<150px, pore<20px) merged to bulk.
- `features.py` — F01–F27 + P-features: composition, watershed-split Si
  stats, Clark–Evans clustering, orientation-split pores/cracks, chord
  lengths, Crofton boundaries, Si–pore proximity, FFT correlation lengths,
  swelling budget (global + sliding-window map), MCP-geodesic tortuosity
  proxy, InLens Meijering-ridge crack metrics, `si_specific_surface`,
  tile-jackknife error bars.
- `detect.py` — robust z (median/MAD), trimmed-mean permutation tests +
  Benjamini–Hochberg (FDR 0.10), E1–E7 fingerprints with must-not-move
  guards and scorecard-derived weights, LOO calibration (measured
  false-alarm), per-image flags vs baseline envelope + consensus-
  exceedance binomial test (heterogeneous batches), bootstrap T² drift.
- `scorecard.py` / `weights.py` — per-feature tickets (plain-language
  meaning, unit, fingerprint membership) + empirical grading: baseline
  stability, discrimination, redundancy → tier + weight multiplier
  (scoring weights never see the batch under test).
- `consequence.py` — cited risk bands: swelling interval (M3/R8), SEI
  surface proxy (R1/R4/R17), plating multiplier (M1/M2/R10), dead-Si
  penalty (R8), damage exposure; MAD-based direction calls; computed on
  anomalous regions for heterogeneous batches.
- `excel_export.py` — workbook: Summary / KPI / All_features /
  Per_image_scores / Significant / Watch_list / Scorecard / Consequences /
  Diagnostics.
- `report.py` / `pipeline.py` — A→Z markdown report, plain-English
  summary, all figures, CLI driver.

## Verdict logic (three evidence paths, worst wins)

- **Uniform shift**: batch fingerprint on trimmed-mean z + BH-significant
  features; ≥2 unpatterned BH-significant features → INVESTIGATE.
- **Heterogeneous batch**: ≥2 strict flags same signature → REJECT; 1
  strict flag + ≥3 same-signature regions above baseline p90 (consensus
  exceedance, binomial) → REJECT; weaker consensus → INVESTIGATE.
- **Subtle drift**: mean T² vs bootstrap null p<0.05 → INVESTIGATE.

Verdict semantics: ACCEPT = indistinguishable within noise; INVESTIGATE =
partial/unattributable evidence; REJECT = beyond the baseline's own
envelope matching a known defect → quarantine + root-cause (NOT "battery
will fail").

## Current state (supersedes the "Results" block below)

The original `polaron_qc` verdicts below are superseded by the
`vcompare/` frozen-recipe comparison (`validated_comparison/`) and the
repaired `micro2dfn` DFN stage (`dfn_output/`):

- No silicon-fraction difference survives a frozen recipe (~6%
  classified Si in all batches; Si:Gr ratio ~0.073 everywhere).
- Batch_1's resolved-pore count is a batch-level shift vs Batch_3 (all
  7 images above the B3 median), surviving noise adjustment — but pore
  count is NOT a DFN input; the model verdict does not cover it.
- DFN comparison discipline (dfn_output/dfn_validation.md): same
  cathode for every batch at each paired sweep index (sized on the
  reference batch's anode), declared meaningful-difference thresholds,
  swelling/stress read as in-cycle discharge peaks, LLI + retention
  diagnostics-only, corrected plating rule (warning counts only where
  the reference stays >= 0 V). Result: peak Si stress resolves batch
  differences (B2 < B3 < ... see dfn_paired_differences.csv); capacity
  and swelling stay below threshold.
- New_Images_Batch: `assign_new_images.py` scores similarity to the
  batches; **all calls are "can't tell"** — each photo's acquisition
  group occurs in only one batch, so assignment cannot separate
  session-match from batch-match (protocol kept: predictions saved
  blind before any ground truth).

## Results (this dataset) — CURRENT, session-corrected

**The dominant fact:** the 31 photos fall into 13 inferred acquisition
groups (frame size / pixel size / detectors / noise / contrast —
`image_acquisition.csv`), unevenly spread across batches; B1–B3 share
one session once. Variance partition (`marker_session_partition.csv`,
chance levels ~0.60 group / ~0.93 batch): **count-type markers read the
session** (residual 0.04–0.06), **fractions are least session-sensitive**
(0.19–0.31) — every marker is still more group- than batch-organized.
Rule: fractions are primary; counts are within-session diagnostics only.

- **Composition: no detectable difference** — candidate-Si ≈6%,
  resolved pores ≈10–11% everywhere; ~2-area-point gaps invisible at
  n=7. This is the only solid batch-comparative result.
- **Unblocked vs blocked:** declared-family exact-permutation + BH gives
  3 survivors — all count-type markers (session-dominated class);
  stratified within-session permutation: **zero survivors** (B1–B3
  untestable — 1 shared photo). Every batch-difference claim from
  counts is therefore "can't tell", not "identical".
- **Batch_1: 2 photos clearly above the reference bright range**
  (12.5%, 11.9% vs ref max 10.7%) + 1 at/just above it (10.74%);
  mostly uncertain-bright material. Solid observation, unresolved as
  material — see the paper §2.
- **DFN (appendix/diagnostics):** one threshold-exceeding indicator —
  peak Si stress ordering (B3 highest) — but model-conditional
  (reachable-Si proxy, ρ≈−0.93), assumption-dominated, and
  session-inheriting. A hypothesis, not a verdict.
- Superseded: the earlier "Batch_1 → REJECT" QC verdict and
  "genuinely different / process-variation" readings — built on
  count-type markers before the acquisition analysis; see
  `archive/ARCHIVE.md`.
- Baseline LOO envelope: flag ≥2.73 (self max 3.36 → zero false alarms)
  — the detection machinery stands; its batch interpretation is
  session-limited.

## Physics interpretation notes

- Si particles sit ~25 px inside bulk, so 1-px contact tests are
  degenerate: contact uses 3 px (~75 nm) dilation + continuous
  `si_pore_dist_mean`.
- `bright_bulk_ratio` is normalised (I_Si − I_pore)/(I_bulk − I_pore) —
  a drift hints at Si vs SiOx grade changes (confirm with EDS).
- Not measurable on these images: binder migration, 3-D pore connectivity,
  SEI, Li inventory (see dictionary "Not visible" section).

## Legacy files (superseded, kept for reference only)

`extract_markers.py`, `segment_images.py`, `agent_core.py`,
`master_pipeline.py`, `build_dictionary.py`, `create_excel.py`,
`simple_tracker.py`, `modal_simulation.py`, `validate_features.py`.

## Modal (cloud compute)

- `modal 1.6.1` is installed in `.venv` — use `.venv/bin/modal` or
  `.venv/bin/python -m modal`.
- Authenticated locally via `~/.modal.toml` (profile `vvadanici`);
  verified working end-to-end (remote smoke test OK).
- For Devin Cloud sessions: `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`
  must be uploaded to the Devin secrets manager (requires
  `devin auth login` first).

## micro2dfn — interpretable marker → DFN pipeline (separate from QC)

Standalone explainable pipeline, independent of `polaron_qc/`:

```bash
# THE one-command clean report (minutes) — the deliverable:
.venv/bin/python build_report.py     # every number -> report_numbers.json
.venv/bin/python build_paper.py      # paper/micro2dfn_paper.tex from CSVs
.venv/bin/python check_report.py     # fails on stale/hand-typed numbers

.venv/bin/python run_dfn.py --fast --out dfn_output   # downsampled, ~6 min
.venv/bin/python run_dfn.py --out dfn_output          # full res
.venv/bin/python run_dfn.py --fast --reuse            # skip marker extraction, redo DFN+report
```

Outputs in `dfn_output/` — `report.md` (start here), `micro2dfn_results.xlsx`,
`markers_*.csv`, `objects_*.csv`, `data_dictionary.csv`, `thresholds.json`
(frozen segmentation cutoffs — reuse on future batches), `dfn_params_*.json`
(exact PyBaMM inputs), `figs/` (overlays: red=Si particle, yellow=bright-fine,
blue=pore).

Key facts:
- Shared multi-Otsu + watershed; object classifier splits si_particle vs
  bright_fine (fixes Batch_1 low-contrast Si overcount — both known bad
  photos auto-flagged).
- ~30 markers per image: composition, Si geometry (Wicksell 3-D radius,
  Crofton surface), pore/transport, Si–pore proximity, imaging guards.
  explained in `data_dictionary.csv`.
- Tortuosity: resolved pores never span any 2-D section → a real
  diffusion solve was removed as unmeasurable (archive/ARCHIVE.md);
  the geodesic proxy feeds the Bruggeman estimate instead.
- PyBaMM DFN (Chen2020_composite + OKane2022, swelling+SEI+stress-LAM):
  17-pt paired sweep per batch over marker bands, 5 × 1C cycles, shared
  cathode per index; indicators read at in-cycle peaks: discharge
  capacity, peak swelling, peak Si stress, min anode potential.
  Deltas are the defensible signal — absolutes illustrative.
  Demoted to appendix in the paper.
  **Saved-run caveat (2026-10-04):** the outputs in `dfn_output/` were
  generated while the particle-radius overrides used invalid PyBaMM
  key names (`...electrode particle radius [m]`), so both radii ran
  at Chen2020 defaults (Si 1.52 µm, graphite 5.86 µm) for every
  batch/point — the swept si/gr radius channels were dead. Fixed in
  `dfn_inputs.py` (`Primary/Secondary: Negative particle radius [m]`,
  graphite now read from measured `gr_radius_eff_um`); saved outputs
  stand as the record — radius-channel sensitivity is untested until
  the next run.
- Built: optional 3-D reconstruction branch — see `recon3d` below.
- `run_dfn.py` now uses reference-only calibration by default:
  thresholds + classifier t_core are fitted on `--baseline Batch_3`
  only, saved to `recipe.json`, and reused verbatim on later runs —
  a new batch can never change how old batches are measured.
  `run_manifest.json` records recipe/code/image md5s per run.

## reconcile — cross-pipeline adjudication (polaron_qc × micro2dfn)

```bash
.venv/bin/python run_reconcile.py    # -> reconcile_output/ (~1 min)
```

Compares the two systems on the same 31 images: metric crosswalk,
6-test adjudication of Batch_1's extra bright material, per-image
verdict-agreement matrix. Outputs: `metric_crosswalk.csv`,
`adjudication.csv`, `verdict_matrix.csv`, `figs/`; technical summary in
`RECONCILIATION.md`; paper in `paper/reconciliation_paper.tex`.

**Headline result — SUPERSEDED framing.** Both systems flag the SAME
Batch_1 regions (detection converges) but disagree on interpretation —
polaron_qc says excess Si, micro2dfn's classifier demotes most of it to
uncertain-bright. The earlier adjudication ("likely fine silicon")
depended on count/texture evidence that is session-sensitive, and the
REJECT was built pre-acquisition-analysis: current status = "can't
tell" until sessions are controlled. What survives: the detection
agreement (both find the same bright fields) and the classifier's role
as the honest bound between them. See Results section above.

polaron_qc report now prints a contrast caveat when an E1 driver batch
shows Si:bulk contrast z<−2 on driver regions.

## recon3d — ARCHIVED optional 2-D→3-D reconstruction branch

Moved to `archive/` (not ground truth, not needed for the core
deliverable; see `archive/ARCHIVE.md`). Still runnable:
`.venv/bin/python archive/run_recon3d.py`, `archive/modal_recon3d.py`.

Standalone sensitivity pipeline (SliceGAN-style GAN + 3-D FDM solve),
independent of both pipelines above but measuring under the SAME
reference-only ruler as micro2dfn's `recipe.json` (thresholds fitted
on the baseline batch only — never pooled over incoming batches):

```bash
.venv/bin/python -m modal run modal_recon3d.py            # full: 3 batches x 3 seeds x 4 vols (CPU)
RECON3D_GPU=1 .venv/bin/python -m modal run modal_recon3d.py --gpu   # A10G path
.venv/bin/python -m modal run modal_recon3d.py --fast     # dev pass
.venv/bin/python run_recon3d.py --fast                    # local CPU smoke (sanity only)
.venv/bin/python run_recon3d.py --analyze-only            # redo solves+report on cached volumes
```

Outputs in `recon3d_output/` — `report.md` (start here),
`transport_metrics.csv` (per-volume τ, percolation, Si accessibility),
`validation.csv` (real-vs-generated 2-D statistics), `volumes/*.npz`
(label volumes, stamped `__V{V}i{iters}__r{ruler}`), `models/*.csv`
(loss histories), `masks_{ruler}_ds{n}.npz` (frozen training data),
`recipe.json` + `masks_manifest.json` (ruler provenance), `figs/`.
`legacy_pooled_ruler/` archives the first ensemble built under the
deprecated pooled-all-images ruler — kept for provenance, superseded.

Key facts:
- Ruler provenance: `recon3d/data.py::load_or_fit_recipe` prefers
  `dfn_output/recipe.json` if present, else fits multi-Otsu on the
  baseline batch alone (tag `cb9831`: t_pore=0.7109, t_si=1.3354 on
  Batch_3). Every artifact carries the ruler hash so ensembles built
  under different rulers can't silently mix.
- SliceGAN: 3-D generator / 2-D critic, WGAN-GP; critic sees ONLY
  z-containing slices ((z,x) and (z,y)) → through-plane statistics are
  learned, the two in-plane directions are ASSUMED statistically
  equivalent (single-view limitation — all sections share one
  orientation). Top-down slices are never judged.
- Compute reality: Modal validates GPU functions at DEPLOY — a
  payment method is required even with credits, so the A10G function
  only exists when `RECON3D_GPU=1` is set. CPU build: V=64³ @
  100 nm/voxel = 6.4 µm edge, G/D width 32, ~2 it/s, ~25–55 min/model,
  9 parallel containers. V=128³ @ 50 nm config exists but needs GPU.
- Transport: Laplace solve on the spanning pore cluster (scipy.sparse
  CG + Jacobi) → τ through-plane & in-plane, spanning/source-connected
  pore fractions, Si-family accessibility (bright voxels adjacent to
  the spanning cluster). Non-spanning realisations report τ=∞ — kept
  as evidence of poor connectivity, not dropped.
- Validation gate: generated z-slices must reproduce real phase
  fractions, S2 two-point correlations and chord distributions before
  transport numbers are quotable (patch-level, reported per batch).
- Acquisition control (`acquisition_control.csv` + report section):
  per-image mask stats joined to `image_acquisition.csv` session
  groups. Pore-fragmentation statistics are SESSION-dominated
  (pore_comp_per_1000vox: 4% residual after group vs 54% after batch);
  within every session shared by >1 batch the batches are nearly
  identical. Per-batch ensemble differences are session-confounded
  candidates, not material findings — session-robust result is only
  "resolved pores rarely span anywhere".
- Report framing: ensembles are statistical realisations, NOT the real
  3-D structure; resolved-phase τ is an upper bound (sub-resolution
  pores invisible); batch comparison of ensembles only. Headline
  metrics are spanning rate + spanning pore share; conditional τ on
  the spanning subset is indicative only (small n).
