# NOTES - pore space analysis

## Image inventory
- Batch_1: 7 BSE images; Batch_2: 7; Batch_3: 17. Matches the earlier
  run's count of 7 / 7 / 17 - no mismatch.
- Some locations store the third detector image as `*_SE.tif` instead of
  `*_ETD.tif` (Batch_2: rxax5ozo; Batch_3: utfgcjfa, vc2whyaq, x77cy643).
  BSE counts are unaffected.

## Assumptions
- Bright BSE particles are ASSUMED silicon-based; grey bulk ASSUMED
  graphite - from image appearance only, no compositional confirmation.
- Pixel size ~25.0 nm/px in every BSE image (TIFF resolution tags; range
  24.999-25.001 nm). No scale bar and no instrument metadata.
- Labels: 0 pore / 1 bulk / 2 bright; bright wins where masks overlap.

## Method notes
- Pore threshold = LOWER cut of 3-class multi-Otsu on the smoothed,
  normalised image (same settings everywhere); objects <20 px removed.
- Bright segmentation identical to `silicon_size` analysis (histogram
  valley, multi-Otsu fallback flagged per image in `bright_method`).
- Pore class precedence: large_gap (area > 0.5% of image) > crack_slit
  (aspect > 3) > compact.
- Ring measures use distance transforms; every non-bright pixel is
  attributed to its nearest particle. Contact = share of inner-boundary
  pixels within 3 px of a pore (dilation approach, not
  direct touching - smoothing leaves a thin grey band between bright and
  dark regions, so direct contact counts are ~0 by construction).
- 'Confined' = zero pore pixels inside the ring; reported at 5/10/20 px.

## Pore checks (STEP 3)
- Empty-black vs resin-filled: median share of pore pixels with raw grey
  level <30/255 is 78%; pores are essentially black/empty,
  not resin-dark, so the dark class is dominated by true voids.
- Bright material inside voids (back walls): essentially absent -
  2 of 31 images contain any such object (max 1;
  columns `n_bright_in_voids`, `bright_in_voids_area_frac` in
  Per_image). Where present those pixels are classified bright, which
  keeps them out of the pore count.
- Pore space next to silicon is LOWER than the image average in every
  batch (local/whole ratio 0.62-0.73 at the 10-px
  ring): the assumed-Si particles sit in comparatively dense regions.
- Batch ranking on mean porosity is stable under +/-10% threshold
  shifts; the Batch_1 < Batch_2 < Batch_3 order is descriptive but only
  marginally significant (KW p=0.054 all images, p=0.116
  excluding outliers), and the Batch_1 vs Batch_3 pairwise contrast
  (Holm p=0.039) is not robust to outlier removal
  (p=0.116). No batch difference on any silicon-adjacent pore
  measure (local porosity, confined share, boundary contact: KW
  p=0.872/0.241/0.801).
- Threshold sensitivity (+/-10% on the pore cut): see the Sensitivity
  sheet. The batch ranking on mean porosity is stable under +/-10% pore-threshold shifts.
- Outlier images 4ih2ggld and 5n1q8atc (Batch_1, low contrast): all
  results reported with (`all_images`) and without
  (`excluding_outlier_images`).

## Cross-check vs earlier rough numbers
- Earlier rough total porosity ~0.089 / 0.100 / 0.109 for Batch_1/2/3
  (weak test, p~0.07). This run: Batch_1: ours 0.088 vs earlier 0.089; Batch_2: ours 0.099 vs earlier 0.100; Batch_3: ours 0.108 vs earlier 0.109. Residual differences
  vs the rough pass are consistent with a different threshold rule plus
  the <20 px cleaning and bright-wins overlap handling applied here.

## Statistics
- IMAGE is the independent sample (n = 7, 7, 17), not the particle or
  the pore. Batch_3's larger image count gives pairwise tests involving
  it more power than the Batch_1-Batch_2 comparison.
- Bootstrap CIs (2000 resamples) resample images and recompute the
  per-image-mean statistic each time.

## Paper access
- Impact of Silicon/Graphite Composite Electrode Porosity on the Cycle Life of 18650 Lithium-Ion Cell, DOI 10.1021/acsaem.0c01999. The full text is paywalled; no open-access
  copy was found (ACS purchase-only page; no OA location on OpenAlex /
  Unpaywall / HAL; figshare hosts only the 5-page SI). All quotes in the
  Paper sheet and report are verbatim from the publisher-hosted abstract
  and the free SI (ae0c01999_si_001.pdf). If the main text is provided,
  the Paper sheet should be extended with main-text quotes.

## Known issues / what failed
- Reading the TIFFs requires `imagecodecs` (LZW decode) - a dependency
  of tifffile, not an extra analysis library.
- Overlays are saved at 50% resolution to keep file sizes reasonable.
- Direct-touch pore contact returns ~0 because smoothing leaves a thin
  grey band; the 3-px distance measure is used instead (non-zero as
  designed - see `contact_frac_mean` in Per_image).
