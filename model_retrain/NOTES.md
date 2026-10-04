# model_retrain - NOTES

## What this is
A NEW RandomForest pixel classifier trained to reproduce the existing
qc_real.py multi-Otsu labels on BSE tiles. There was no pre-existing trained
model or saved weights in the repo (qc_real.py is rule-based); the user
confirmed training a new model on the Otsu masks.

## IMPORTANT interpretation caveat
Labels are produced by a threshold pipeline, not a person. Every accuracy
figure = agreement with multi-Otsu + morphology, not ground truth.
Bright particles are assumed silicon-based, grey bulk graphite - appearance
only, stated per standing instruction.

## Data
- 31 locations, 7/7/17 per batch (matches earlier count, no mismatch).
- 2x2 tiling -> 124 tiles; tile size ~3500 x ~1000 px (all >> 200 px).
- Detectors BSE/Inlens/ETD share identical dims per location; `_SE.tif`
  treated as the side detector where `_ETD.tif` is absent.
- Tile NPZs and predictions are local artifacts (gitignored); only the
  model file, splits.json and report/xlsx/figures are committed.

## Splits (seed 20241003)
- test: 0grcilhi, 4ih2ggld, hawkfj64, kbdh4tri, r17byphk, rxax5ozo, uhdslk0o
- folds: [["5n1q8atc", "71vgq3fw", "i9jiqjwl", "tuy3zymq", "x77cy643"], ["avn74qx1", "hzumfsms", "iv6g2oq0", "x7u69zsw", "xgj4xftb"], ["b3esycq1", "f1vzngrs", "mgxahqnk", "ufdvpb81", "vc2whyaq"], ["epqdaau9", "ffwubibz", "pl8uabbv", "ptg8lmto", "utfgcjfa"], ["3806gxp0", "9luzk4jm", "cfe5vt7s", "fzrt2k6r"]]
- Leakage assertion passed: no image ID in more than one split.

## Headline numbers
- Test accuracy 0.9892 [95% CI 0.9838-0.9937]
  (image bootstrap). Target 95%: REACHED.
- CV accuracy 0.9774 +/- 0.0154.

## Sanity check results
- 1_leakage_assert: PASS - all image IDs appear in exactly one split
- 2_leakage_demonstration_WRONG: PASS - WRONG split (random tiles): 16/20 val tiles share a parent image with train; acc=0.9889 bal=0.9752
- 3_shuffled_labels: PASS - shuffled-label acc=0.8120 (majority fraction ~0.832); expected collapse to baseline
- 4_baselines: PASS - majority-class acc mean=0.8161; raw per-tile multi-Otsu agreement mean=0.9774
- 5_overfit_train_vs_val: PASS - see fold table: f0 tr=0.990 va=0.980; f1 tr=0.987 va=0.987; f2 tr=0.988 va=0.984; f3 tr=0.987 va=0.950; f4 tr=0.988 va=0.986
- 6_tile_edge: PASS - edge acc 0.9812 vs interior 0.9839 (mean gap 0.0026)
- 7_reassembly: PASS - stitched-vs-whole mismatch mean 0.0092
- 8_reproducibility: PASS - same-seed predictions identical: True; 3-seed acc spread: [0.9707, 0.9705, 0.9702]
- 9_heldout_batch: PASS - Batch_1: 0.9851; Batch_2: 0.9599; Batch_3: 0.9763
- 10_eyeball_examples: PASS - side_by_side.png + worst_tiles.png saved; worst tile bal-acc=0.918

## Assumptions / known issues
- Feature windows are truncated at tile edges -> tiny stitch-vs-whole
  mismatch (reported in check 7).
- '95%' is measured at the pixel level against Otsu labels; a human-labelled
  ground truth does not exist, so correctness vs reality is unknown.
- Training pixels are subsampled (30k/tile) for runtime; evaluation on the
  test images uses every pixel.
- The two low-contrast Batch_1 outliers (4ih2ggld, 5n1q8atc) are kept and
  flagged; they follow the same grouped-split rules.
