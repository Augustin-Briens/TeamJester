# Annotation provenance

Every annotation in this pack — the coloured overlays, circled
particles, "ambiguous bright" regions, crop rectangles, and
annotated crops — was produced by the **pipeline and the agent**, not
by a human expert. **None of it is a ground-truth label.**

## Where each annotation came from

| Asset | Produced by | Basis | Status |
|---|---|---|---|
| `annotated_images/ref_raw_crop.png` | `build_figures.py` | centre crop of img_hawkfj64 | mechanical crop |
| `annotated_images/ref_flat_crop.png` | same | flat-fielded crop | mechanical crop |
| `annotated_images/ref_seg_crop.png` | same | frozen-recipe segmentation | algorithmic labels |
| overlays in `fig_seg_steps`, `fig_watershed`, `fig_classifier`, `fig_measure` | segmentation + object classifier | frozen thresholds + classifier rules | algorithmic labels |
| `fig_disputed3` regions | QC flag machinery | consensus-exceedance detection | algorithmic flag |
| `fig_measure` circled particle | scripted pick: ~3 µm object in crop | teaching choice, documented in script | mechanical selection |
| `annotations.csv` (10 rows) | `build_figures.py` `note()` calls | what each figure marks, by rule | algorithmic |

## What these annotations are NOT

- **Not expert labels.** No SEM scientist or materials expert has
  reviewed these crops. Blind expert annotation is a *roadmap item*
  precisely because none exists.
- **Not chemical identity.** "si_particle" / "bright_fine" are
  classifier names for brightness+morphology patterns; EDS would be
  needed for species.
- **Not a benchmark.** They cannot be used to score "accuracy" of
  any pipeline — using pipeline output as ground truth is circular.

## If a human expert annotates later

1. Annotate **blind to batch labels and to pipeline output**.
2. Record: annotator id, date, instructions version, object-level
   labels, confidence.
3. Compare to pipeline labels at the *object* level (precision/recall
   on si_particle), report disagreement — do not "correct" the data.
4. The comparison, not the annotation alone, is the deliverable.
