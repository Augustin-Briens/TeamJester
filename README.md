# TeamJester — SEM batch analysis & classifier

Analysis of the `Hackathon-Polaron` SEM dataset: 3 batches of electrode
cross-section images (BSE / Inlens / ETD detectors), ~7–8 imaged locations
per batch. **Batch_3 is the supplier's promised baseline**; Batch_1 and
Batch_2 are subsequent deliveries tested as deltas against it. Everything
is explainable — no black-box models.

## Repo layout

```
analysis/            the analysis pipeline (run order below)
  segment.py         Phase 1 — BSE segmentation into pore / graphite / silicon
                     (+ Inlens fine-crack + ETD topography channels)
  features.py        Phase 2 — ~60 explainable features with dictionary
                     (feature_meta.csv)
  baseline.py        Phase 3 — Batch_3 baseline stats + robust-Mahalanobis
                     envelope (baseline_stats.csv, envelope.json)
  deltas.py          Phase 5 — Batch_1/2 deltas vs Batch_3 (mean + bootstrap
                     + Holm)
  categorise.py      Phase 6 — batch classifier: nearest-centroid on
                     robust-Mahalanobis distances, IN/OUT envelope, LOO
                     validation, group-size calibration
  dfn.py             Phase 4 — PyBaMM BasicDFN discharge simulations from
                     image-derived electrode parameters
  reports.py         Phase 7 — evidence figures + PDF reports
  run_all.py         one-command regeneration of every table/figure/report
  make_latex.py      builds outputs/reports/final_report.pdf (tectonic)
  NOTES.md           decisions, findings, follow-ups log
  outputs/
    tables/          every CSV the pipeline produces
    figures/         evidence + report figures
    reports/         PDFs: 00_summary … 05_causes + final_report.pdf
    xlsx/            Excel workbooks per phase
webapp/              the batch-classifier web app (see below)
Hackathon-Polaron/   the dataset (untouched)
```

## The pipeline steps

1. **Segment** each BSE image into silicon (bright), pore/crack (black),
   graphite bulk — validated against manual point-counting; flagged images
   listed in `quality_guards.csv`.
2. **Features**: interpretable measurements only — area fractions, particle
   sizes, clustering (Clark–Evans), correlation lengths, texture — gated on
   left/right-half repeatability and imaging-noise decorrelation.
3. **Baseline**: Batch_3 robust centre/scale per feature → Mahalanobis
   envelope (acceptance threshold at the batch's own 95% spread).
4. **Deltas**: Batch_1/Batch_2 mean shifts with bootstrap CIs and Holm
   correction → ranked delta tables + figures.
5. **Categorise**: nearest batch centroid in robust-z space → IN/OUT +
   most-like batch + softmax surety; validated leave-one-out and by group
   size (group calls of 3+ images are the reliable unit).
6. **DFN simulation**: image-derived porosity/particle stats feed PyBaMM
   discharge curves for each batch.
7. **Causes/reports**: evidence figures and PDFs tying each delta to a
   physical story.

**Removed metrics**: `si_d10_um` (smallest silicon particle) is
stereologically broken — a 2D cut through tightly packed particles reports
glancing chord-cuts as small particles (~0.37 µm floor on every batch =
the sectioning geometry, not the material). A DINOv2 embedding variant was
evaluated and removed; the product is 100% explainable.

## The web app (`webapp/`)

Upload SEM photo(s) → it segments with the identical pipeline, computes the
features, and returns:

- **IN/OUT** of the Batch_3 promised envelope (distance vs threshold gauge),
- **most-like batch + % surety** (share of e^(−distance) across the three
  centroids — the exact math is shown),
- **top drivers in plain words** (z-scored vs Batch_3),
- **per-detector contribution cards** (what BSE / Inlens / ETD each told
  the model),
- a **B2-vs-B3 focused** column (dedicated logistic boundary),
- the segmentation overlay as evidence.

Reliability: single images ~45% LOO — upload 5+ locations of a sample for
the ~100% group call.

Run locally:

```bash
cd webapp
pip install -e .           # or: pip install fastapi uvicorn python-multipart ...
uvicorn app.main:app --port 8811
```

## Reproduce the analysis

```bash
cd analysis
python run_all.py          # rebuilds all tables/figures/reports from the images
python categorise.py --validate   # LOO validation only
python categorise.py <folder>     # classify an unseen image folder
python make_latex.py       # rebuild final_report.pdf (needs tectonic)
```
