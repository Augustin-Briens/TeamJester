# Multi-detector channel analysis (InLens + ETD) — EXPLORATORY

Not part of the frozen QC recipe. BSE is atomic-number contrast; InLens/ETD
are secondary-electron (surface-sensitive) channels. Registration verified
(0 px phase-correlation shift on every image).

**Session caveat (primary):** InLens gain differs by batch (median
brightness ≈89 for Batch_3 vs ~110 for Batches 1–2), so every channel
feature inherits the acquisition confound. These are exploratory
follow-up signals, not batch markers.

## Batch medians

| feature | Batch_1 | Batch_2 | Batch_3 | New_Images_Batch |
|---|---|---|---|---|
| inl_pore_mean | 0.528 | 0.684 | 0.840 | 0.662 |
| inl_bulk_mean | 1.046 | 1.059 | 1.223 | 1.065 |
| inl_si_mean | 1.471 | 1.460 | 1.720 | 1.451 |
| inl_unc_mean | 1.404 | 1.784 | 2.109 | 1.689 |
| inl_pore_darkness | 0.495 | 0.656 | 0.689 | 0.621 |
| inl_unc_si_like | 1.253 | 1.287 | 2.254 | 1.730 |
| inl_edge_density | 0.100 | 0.108 | 0.125 | 0.105 |
| etd_pore_mean | 0.204 | 0.211 | 0.195 | 0.202 |
| etd_bulk_mean | 1.026 | 1.030 | 1.026 | 1.030 |
| etd_si_mean | 1.505 | 1.566 | 1.564 | 1.538 |
| etd_unc_mean | 1.423 | 1.446 | 1.604 | 1.576 |
| etd_pore_darkness | 0.198 | 0.204 | 0.188 | 0.192 |
| etd_unc_si_like | 0.740 | 0.731 | 1.089 | 1.078 |
| etd_edge_density | 0.107 | 0.107 | 0.098 | 0.103 |
| inl_std_si | 0.145 | 0.146 | 0.173 | 0.167 |
| inl_std_unc | 0.233 | 0.336 | 0.328 | 0.313 |
| inl_std_bulk | 0.151 | 0.173 | 0.219 | 0.169 |

## What the channels say

1. **InLens pore-contrast — possible, n=3.** Batch_1's pores read
   darkest in InLens in all three sessions it shares with another batch
   (`inl_pore_darkness` 0.49–0.61 vs 0.51–0.69 for same-session partners).
   The only candidate signal whose direction survived blocking — but on
   three matched observations, with batch-correlated detector gain. A
   controlled re-imaging follow-up signal, NOT a measured pore depth or
   "more open voids".
2. **Uncertain-bright material is bright material, not bulk misread** —
   in both surface channels it sits at or above classified-Si brightness
   (`inl_unc_si_like` 1.25–2.25), with ~2× the local texture. Consistent
   with thin/edge-rich bright material rather than solid grains. This
   validates keeping it as a separate uncertain label; it does NOT
   establish silicon identity (edges and topography also read bright in
   surface channels) — the all-bright upper bracket stays a bracket.
3. **Removed:** the FFT "vertical texture"/curtaining test. The
   directionality score measured the frame's aspect ratio (noise scores
   +0.70 at the real 1034×3500 shape); direct autocorrelation shows
   features slightly *horizontally* elongated. See archive/ARCHIVE.md.

## What the channels still cannot do

- chemical identity (still needs EDS)
- absolute calibration across sessions without acquisition metadata
  (gain/brightness settings not in TIFF tags)
- they corroborate the frozen-ruler labels but do not replace them
