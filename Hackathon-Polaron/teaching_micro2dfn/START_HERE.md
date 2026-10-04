# micro2dfn masterclass — START HERE

A teaching + audit pack for the Polaron anode-cross-section project:
how a BSE image becomes labels, labels become a batch comparison and
model inputs, and exactly where the evidence stops.

Everything is built from **this repository's real outputs** — nothing
is invented, staged, or drawn from memory. Every quantitative claim in
the deck is traceable to a file via `claim_evidence.csv`.

---

## Read this first (2 minutes)

The dataset is **34 back-scattered-electron (BSE) fields of view** of
graphite+silicon anode cross-sections: Batch_1 (7), Batch_2 (7),
Batch_3 (17 — the shipment-approved **reference**), plus 3 unlabelled
new photos. Two systems work on the same pixels:

- **polaron_qc** — quality-control (QC) anomaly detection. Asks:
  *does this batch break the reference's own envelope?* Verdicts
  (ACCEPT / INVESTIGATE / REJECT) are **similarity statements**, not
  chemistry or performance.
- **micro2dfn** — structural measurement → exploratory consequence
  indicators from a standard battery-physics model (a
  Doyle–Fuller–Newman cell simulation, run by the PyBaMM package).
  Asks: *what structure is resolved, and what could it do in a
  physics model under declared assumptions?*

## The honest headline (what is established)

| Statement | Status |
|---|---|
| Batch_1's resolved **pore count** is higher than Batch_3's — uniform across all 7 images, survives noise adjustment and leave-one-out | **supported** (exact p = 0.0005 raw, 0.0019 adjusted; `validated_comparison/pairwise_comparisons.csv`) |
| 3 of 7 Batch_1 regions carry extra bright material; six pixel-level tests support **likely fine silicon** | **best-supported interpretation** — chemistry unconfirmed without EDS (elemental spectroscopy) |
| Both pipelines flag the *same* regions; they only disagree on what to call them (classified Si r = 0.27) | established (`reconcile_output/`) |
| DFN paired differences: capacity/swelling below declared thresholds; peak Si stress differs | **model-conditional only** — assumption spread dwarfs batch differences on capacity |
| "Batch_3 is defect-free" | **not claimed** — it is a selected reference, not ground truth |
| Any measured cell performance, lifetime, or safety consequence | **absent by design** — no electrochemical data exist |

## The 10-slide route

For a ~15-minute pass, follow only the slides tagged **ROUTE**
(slides 1, 3, 4, 7, 10, 13, 16, 29, 31, 35 — exactly ten).
Everything else is the teaching path.

## Files

| File | What it is |
|---|---|
| `micro2dfn_masterclass.pptx` / `.pdf` | the deck (41 slides incl. appendix) + matching PDF |
| `presenter_script.md` | verbatim narration for the route |
| `metric_dictionary.csv` | every metric: units, equation, limits, evidence class |
| `claim_evidence.csv` | 50 claims → source file + evidence class |
| `dataset_manifest.csv` | all 34 FOVs, channels, pixel pitch, md5 |
| `metric_watchlist.md` | the one-page "watch these" list |
| `GLOSSARY.md` | acronyms and jargon in plain words |
| `roadmap.md` | what a prospective validation protocol requires |
| `knowledge_check.md` | 8 questions with answers |
| `expert_review_sheet.csv` | what a human expert should verify |
| `annotation_provenance.md` | where every annotation came from (agent, not expert) |
| `version_manifest.txt` | environment + recipe pins |
| `BUILD.md` | one-command rebuild of figures → deck → PDF |
| `figures/` | all generated figures |
| `annotated_images/` | annotated reference crops (agent annotations) |
| `claim_audit.md` / `.csv` | live audit: every file-checkable claim vs current sources |
| `sync_log.md` | append-only history: what changed, what was rebuilt |
| `build_state.json` | input fingerprints + last-sync stamp |
| `scripts/` | `sync.py` (one-command update), `make_manifest.py`, `build_figures.py`, `build_deck.py`, `audit_claims.py` |

## Ground rules this pack honours

- **New data never recalibrates the ruler.** Thresholds, the noise
  model, and the classifier are fitted on Batch_3 and frozen.
- **The image is the independent unit** — not the pixel, not the
  particle. n = 7/7/17 is small; every "no clear difference" carries a
  minimum-detectable-difference.
- **Measured / inferred / assumed / simulated / illustrative** are
  labelled everywhere. Model outputs are conditional indicators.
- **Verdicts are not verdicts about cells.** REJECT meant "beyond the
  baseline envelope" under a calibration that was later corrected —
  today the defensible label for Batch_1 is INVESTIGATE.
- **Agent annotations are not expert labels.** Every annotation in
  this pack was produced by the pipeline/agent and marked as such;
  the expert-review sheet lists what a human must confirm.

## Keeping this pack current (it updates itself)

The pack tracks the repository state. Run:

```bash
.venv/bin/python teaching_micro2dfn/scripts/sync.py
```

`sync.py` fingerprints every input (images, output CSVs, recipe files,
pipeline code, pack scripts), rebuilds **only the stages the changes
touch** (manifest → figures → deck → PDF → rendered pages), regenerates
`version_manifest.txt`, and re-audits every machine-checkable claim in
`claim_evidence.csv` against the current files.

- `sync.py --check` — report drift + audit, build nothing.
- `sync.py --force` — rebuild everything.
- A **STALE** or **MISSING** claim means the files on disk no longer
  say what the deck/docs say — fix the text or the claim before
  trusting that statement again. See `claim_audit.md`.
- Every run appends to `sync_log.md`, and the deck's appendix A5
  prints a build stamp (sync time + input count) so a PDF reader can
  tell which repository state produced it.

**Important**: if the upstream pipelines are re-run (`run_dfn.py`,
`run_qc.py`, `run_reconcile.py`, `assign_new_images.py`), run
`sync.py` afterwards — the audit will show exactly which claims the
new outputs changed.

## Reproduce (manual stages)

```bash
cd /Users/raduvadanici/Downloads/Hackathon-Polaron
.venv/bin/python teaching_micro2dfn/scripts/make_manifest.py
.venv/bin/python teaching_micro2dfn/scripts/build_figures.py
.venv/bin/python teaching_micro2dfn/scripts/build_deck.py
/Applications/LibreOffice.app/Contents/MacOS/soffice --headless \
  --convert-to pdf --outdir teaching_micro2dfn \
  teaching_micro2dfn/micro2dfn_masterclass.pptx
```
