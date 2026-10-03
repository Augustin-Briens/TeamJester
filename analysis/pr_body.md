## Summary

Full baseline/delta analysis of the `Hackathon-Polaron` SEM dataset (7/7/17 locations, detectors verified pixel-aligned, 25 nm/px). Batch_3 is characterised as the baseline; Batch_1 and Batch_2 are expressed as deltas with bootstrap intervals, Holm-corrected p-values and per-feature minimum detectable differences; `categorise.py` classifies unseen image groups against the Batch_3 envelope.

Key findings:
- **Imaging vs physical signal**: the two strongest "differences" (`bse_bulk_texture`, `etd_roughness`) correlate >0.85 with the noise guard — they are acquisition signatures, handled via an artefact-safe feature set (`ARTEFACT_RISK`) reported alongside.
- **Batch_1**: `bse_bulk_texture` +18% survives Holm (p=0.014); pore_frac −13.6% and large_gap_frac −71% are suggestive (CI excludes 0). Pore chords shrink ~5% in BOTH directions → calendering rejected, denser packing preferred (labelled inference in `05_causes.pdf`).
- **Batch_2**: nothing survives Holm; gr_st_coherence −8.4%, etd_roughness +9.8% suggestive. Group-level DFN at 2C/measured-porosity predicts ~3% capacity loss for B1/B2 (CI excludes 0); the corrected-0.30 porosity run makes batches indistinguishable.
- **Categorisation honesty**: single-image LOO is only ~45% (documented); group-of-5+ images reaches ~97-100% for B1, ~70-100% for B2 — `images_needed.csv` quantifies it; `categorise.py` classifies a folder as ONE group accordingly. Verified end-to-end on a Batch_2-as-unseen folder.

## Deliverables (`analysis/`)

- `segment.py` `features.py` `baseline.py` `deltas.py` `dfn.py` `categorise.py` `score_pointcount.py` `run_all.py` (cached; `--clean`/`--fast` flags; verified the entry point runs all stages)
- `outputs/xlsx/`: 6 workbooks (validation, features, baseline, deltas, dfn, categorisation) — PDFs read the same CSVs so numbers match by construction
- `outputs/reports/`: `00_summary.pdf` (2pp) + `01`-`06` per-phase PDFs, each opening with a one-paragraph answer
- `outputs/figures/`, `outputs/overlays/` (31 overlays + 3 contact sheets), `outputs/pointcount/` (300-pt sheet + crops, human column left empty for organisers + scorer script), `outputs/masks/`
- `NOTES.md`: every assumption, the 2 flagged Batch_1 images (4ih2ggld, 5n1q8atc), all 33 dropped features with reasons

## Verification vs prior findings

- Silicon fraction same across batches: **agree** (B1 −2.4% with flagged images excluded; the raw +45% is the two bad images)
- B1 ~18% less pore: **directional agree** (−13.6%, suggestive)
- B2 fewer elongated cracks: **not confirmed** — crack_len_density was non-repeatable (dropped); the visible difference is in B1, not B2
- The 2 flagged images: **confirmed** by absolute quality guards (Si/bulk contrast + noise)

## Limits (all stated in outputs)

Assumed graphite bulk / SiOx bright particles / black pore is unconfirmed; visible porosity ~10% < true; DFN absolutes unvalidated (differences only); causes are inferences not proof; no binder class claimed (not separable); all methods identical across batches.
