# Measurement Foundation — what the numbers mean, and what we still owe them

External review verdict (adopted): the analysis is only as strong as its
first link. A brightness cutoff can define a *repeatable category of
bright objects*; calling that category "silicon," calling a fraction of
it "active," and predicting mechanics are three successive assumptions.
A sophisticated simulator cannot repair an uncertain label.

This document records (a) the assumption chain, honestly annotated, and
(b) the five foundation priorities with their current status.

## The assumption chain — measured vs assumed

| link | what we actually have | status |
|---|---|---|
| brightness → object map | frozen 3-class map (pore/bulk/bright) fitted on Batch_3 only | **measured** — repeatable, recipe.json + run_manifest.json |
| bright objects → "silicon" | object classifier + physics priors (Z-contrast: Si is the brightest major phase expected) | **labelled proxy** — chemistry unproven without EDS |
| classified Si → "active Si" | coverage/enclosure heuristic → `si_accessible_frac`; bracketed by all-uncertain-bright upper bound | **assumption** — no electrical measurement exists in the data |
| active Si → stress | PyBaMM Bonkile2024 mechanics on composite DFN | **model output** — depends on bracket + swept parameters |
| stress → defect/lifetime | — | **not claimed** — 5 cycles can't resolve it |

Everything downstream of link 2 inherits the label uncertainty. The
bracket design (classified-Si → all-bright) is the honest answer we can
give from images alone; EDS on a reviewed subset is the answer we cannot
give yet.

## Priority 1 — frozen reproducible pipeline — DONE (this round)

- `run_dfn.py` fitted pooled thresholds + pooled classifier on all
  discovered images; adding a batch silently re-scaled prior results.
  Measured impact: `pore_count_per_mpx` moved up to **24.9 units (~22%)**
  on a single image between the pooled and frozen ruler.
- **Fix**: recipe (t_pore, t_si, t_core) is fitted on `--baseline
  Batch_3` only, saved to `recipe.json`, reused verbatim on every later
  run (`--recipe` to point at a different frozen recipe).
- Cross-check: the baseline-fitted values are **identical** to
  vcompare's frozen recipe (t_pore=0.710865, t_si=1.335364,
  t_core=1.549544) — two independent pipelines converging on the same
  ruler is evidence the fit is stable, not batch-dependent noise.
- `run_manifest.json`: recipe md5 + code md5s + every input image md5 —
  each report now names exactly which run produced it.
- `--reuse`: skip re-extraction; reuse cached markers under the frozen
  recipe (60 min → 5 s locally).
- polaron_qc's floating ruler remains as an explicitly labelled
  *diagnostic* (`floating_thresholds`), per `fig_frozen_floating.png`.

## Acquisition confound — the central data-quality result (2026-10-04)

`image_acquisition.csv`: the 31 photos fall into **13 inferred
acquisition groups** (frame size/pixel size/detectors/noise/contrast).
Variance partition (`marker_session_partition.csv`, chance levels
~0.60 after group / ~0.93 after batch — always quote them together):

- **Count-type markers read the session** — pores/mpx residual 0.04,
  noise-adj count 0.06, pore D50 0.06 after group (vs 0.4–0.6 after
  batch). Within a session, counts are nearly batch-independent.
- **Fractions are the least session-sensitive class** (residual
  0.19–0.31) — but groups still explain a substantial share; they are
  primary because they are area-conserving AND least fragile, not
  because they are session-free.
- Working rule: **fractions primary; counts are within-session
  diagnostics only.** All unblocked batch "differences" were
  count-type; zero survive within-session blocking (B1–B3 share one
  session with one photo each — the decisive pair is nearly untestable).
- "Can't tell" ≠ "identical": the blocked test is under-powered.
  Resolution path: acquisition metadata or interleaved re-imaging.

## Multi-detector check (InLens + ETD) — exploratory layer

`analyze_channels.py` → `channel_features.csv`, `channel_report.md`.
InLens gain differs by batch (89 vs ~110) — every channel feature
inherits the confound.

- **InLens pore contrast — possible, n=3** — Batch_1's pores darkest in
  all 3 shared sessions; the only direction surviving blocking, but on
  thin evidence with batch-correlated gain. Follow-up, not finding.
- **Uncertain-bright is bright material** — surface-bright (≥ Si-level)
  but high-texture (edge-rich) — validates the separate label; does NOT
  establish silicon identity.
- Registration: 0-px shift on all 34 fields.
- **REMOVED:** FFT "vertical texture" + curtaining test — the score
  measured frame aspect ratio (noise = +0.70 at real shape); direct
  autocorrelation shows slightly *horizontal* features. See
  `archive/ARCHIVE.md`.

## Priority 2 — independent label check — OPEN (partially prepped)

- `vcompare` already emits review artifacts: `seg_*.png` overlays,
  disputed-region flags (`img_4ih2ggld`, `img_5n1q8atc` —
  `img_f1vzngrs` review flag), problem-photo list.
- Needed next: a small reviewed panel — ordinary + disputed + each
  batch — with expert outline assessment; EDS spot-checks to confirm
  bright objects are Si (vs SiOx/other) and to estimate how much
  uncertain-bright is real Si. Until then "silicon" stays "bright,
  Si-consistent" in every report.

## Priority 3 — measurement repeatability — OPEN (partially)

- We already compute within-image repeatability proxies
  (`*_tile_err` columns): split-image re-measurement floors.
- Missing: re-imaging the *same material* under controlled conditions to
  bound imaging-only metric drift. The acquisition-group analysis above
  now shows this is THE binding constraint: even the noise-adjusted pore
  count is session-dominated (residual 0.06 after group). A controlled
  re-imaging run (interleaved batches per session) is the decisive
  experiment.

## Priority 4 — blind prospective test — PROTOCOL READY

- The three New_Images_Batch calls were made and saved before any
  ground truth: `new_image_assignment/assignments.csv` +
  `assignment_report.md`, method = envelope coverage under the frozen
  vcompare recipe. That IS the blind-test pattern: prediction written,
  then checked. **All three calls are "can't tell"** — each photo's
  frame height occurs in only one batch's group, so the assignment
  cannot separate session-match from batch-match.
- For the next incoming batch: same pattern — freeze recipe, run
  assignment, archive `run_manifest.json`, *then* compare with
  production records. Success criteria stated up front: false alarms,
  missed differences, and correct "inconclusive" outputs.

## Priority 5 — does the DFN add predictive value? — OPEN

- Honest position today: the DFN is a *consequence model* — demoted to
  an appendix in the paper. Its one threshold-exceeding indicator
  (peak Si stress ordering, −0.95/−1.60 MPa) is carried by a *modelled*
  proxy (reachable-Si share, ρ≈−0.93), is assumption-dominated (label
  bracket alone moves stress up to 1.8 MPa — more than the batch
  difference), and inherits the session confound through its inputs.
  A hypothesis for the image layer, not a measured fact.
- Still owed: a comparison against a *simpler* predictor (batch-median
  markers → linear/summary statistics) on independent outcomes, to test
  whether modelled indicators beat plain image summaries at all.

## Focus metrics (per review)

Primary: image quality (`noise_mad`, `si_bulk_contrast`), resolved pore
fraction + count, `si_candidate_frac` vs `uncertain_bright_frac`,
particle size (`si_d50_um`). Secondary/spatial kept as exploratory
layers with declared confounds (e.g. `si_clustering_R` carries a
brightness dependence ρ=−0.67 — documented, not yet corrected).

## Literature anchors

- Kench & Cooper 2021, *Generating 3D structures from a 2D slice with
  GAN-based dimensionality expansion* (arXiv:2102.07708) — the recon3d
  method (branch archived — `archive/recon3d/`); their own fidelity
  framing (statistical match, not ground truth) is exactly the caution
  our report applies.
- Eyre & Kench 2022, *MicroLib* (arXiv:2210.06541) — 2D-statistics
  validation protocol we mirror in `archive/recon3d/validate.py`
  before quoting transport numbers.
- Chen2020 composite DFN, O'Kane2022 degradation, Bonkile2024 Si-Gr
  mechanics — model inputs already cited in `paper/micro2dfn_paper.tex`.
- Note on BSE: atomic-number contrast identifies *heavy-element-rich*
  regions, not compounds; Si vs SiOx vs other bright phases is not
  decidable from BSE alone — the formal reason EDS is the label gate.
